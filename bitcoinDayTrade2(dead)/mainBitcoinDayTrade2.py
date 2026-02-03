from __future__ import annotations

import os
import time
import pickle
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api import TimeFrame
from alpaca_trade_api.rest import TimeFrameUnit

from gmail import sendReport

load_dotenv()

# ---------------- CONFIG ----------------
LOG_FILE = "bitcoin_daytrade_v2.log"
STATE_FILE = "bitcoin_daytrade_v2_state.pkl"

# Timing
CHECK_SECONDS = 60                      # 1 check per minute
SAVE_EVERY_SECONDS = 15 * 60            # every 15 minutes
REPORT_EVERY_SECONDS_OPEN = 2 * 60 * 60 # every 2 hours while market open

# Email toggles
ENABLE_REPORTS = True
ENABLE_STARTUP_REPORT = True

# Symbols
BTC_TRADE_SYMBOL = "BTC/USD"   # for submit_order + crypto bars
BTC_POS_SYMBOLS = ["BTCUSD", "BTC/USD"]  # try both for get_position
BITI_SYMBOL = "BITI"

# Strategy parameters
UPTREND_WINDOW = 3             # last 3 one-minute closes
MIN_MOVE_PCT = 0.00055          # 0.04% over 3 minutes to be considered "strong"
COOLDOWN_SECONDS = 60 * 2      # 2 minutes cooldown after any trade

# Risk / sizing
STARTING_MONEY_USD = 100_000.00
MAX_TRADE_USD = STARTING_MONEY_USD * 1.00
MIN_TRADE_USD = 1.00

# BITI handling
SELL_BITI_BEFORE_CLOSE_MIN = 5  # minutes before close: force BITI exit + disable BITI buys

# If no holdings and signal is FLAT:
DEFAULT_WHEN_FLAT_MARKET_OPEN = "CASH"   # "BTC" or "CASH"
DEFAULT_WHEN_FLAT_MARKET_CLOSED = "BTC"  # "BTC" or "CASH"

# ---------------- ALPACA SETUP ----------------
API_KEY = os.getenv("alpacaKey")
API_SECRET = os.getenv("alpacaSecret")
BASE_URL = os.getenv("alpacaBaseURL", "https://paper-api.alpaca.markets")
alpaca = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")


# ---------------- LOGGING (FILE ONLY) ----------------
def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def log_line(msg: str) -> None:
    ts = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


# ---------------- BST FOR PRICE TRACKING ----------------
@dataclass
class BSTNode:
    key: float            # unix timestamp
    value: float          # price
    left: Optional["BSTNode"] = None
    right: Optional["BSTNode"] = None

class PriceBST:
    def __init__(self) -> None:
        self.root: Optional[BSTNode] = None

    def insert(self, key: float, value: float) -> None:
        def _ins(node: Optional[BSTNode], k: float, v: float) -> BSTNode:
            if node is None:
                return BSTNode(k, v)
            if k < node.key:
                node.left = _ins(node.left, k, v)
            elif k > node.key:
                node.right = _ins(node.right, k, v)
            else:
                node.value = v
            return node

        self.root = _ins(self.root, key, value)

    def last_value(self) -> Optional[float]:
        node = self.root
        if node is None:
            return None
        while node.right is not None:
            node = node.right
        return node.value


# ---------------- STATE PERSISTENCE ----------------
def default_state() -> Dict[str, Any]:
    return {
        "trees": {
            "BTC": PriceBST(),
            "BITI": PriceBST(),
        },
        "meta": {
            "last_bar_time": None,           # iso str
            "last_trade_time": None,         # iso str
            "last_report_time": None,        # iso str
            "last_report_prices": {},        # {"BTC": x, "BITI": y}
            "price_window": [],              # last 3 BTC 1m closes
            "last_signal": "INIT",           # UP/DOWN/FLAT
            "last_move_pct": 0.0,
        },
        "exit": {
            "active": False,
            "asset": None,                 # "BTC" / "BITI" etc
            "exit_price": None,
            "lowest_since_exit": None,
            "post_exit_window": [],        # recent prices since exit
            "exit_time": None
}

    }

def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, STATE_FILE)

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return default_state()
    try:
        with open(STATE_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        log_line(f"ERROR: load_state failed -> {repr(e)} (starting fresh)")
        return default_state()


# ---------------- ACCOUNT / MARKET HELPERS ----------------
def market_is_open() -> bool:
    c = alpaca.get_clock()
    return bool(c.is_open)

def minutes_to_close() -> Optional[float]:
    """
    Returns minutes to next_close if market is open, else None.
    """
    try:
        c = alpaca.get_clock()
        if not c.is_open:
            return None
        close_dt = c.next_close
        now = now_utc()
        delta = close_dt - now
        return delta.total_seconds() / 60.0
    except Exception:
        return None

def account_cash_and_bp() -> Tuple[float, float]:
    acct = alpaca.get_account()
    cash = float(getattr(acct, "cash", 0.0))
    bp = float(getattr(acct, "buying_power", 0.0))
    return cash, bp

def spendable_usd() -> float:
    """
    Rule:
      - if buying_power <= cash -> follow buying_power
      - if buying_power > cash -> follow cash (avoid debt)
    Equivalent: min(cash, buying_power)
    """
    cash, bp = account_cash_and_bp()
    usable = bp if bp <= cash else cash
    return max(0.0, usable - 5.00)  # buffer

def iso_to_dt(x: Optional[str]) -> Optional[dt.datetime]:
    if not x:
        return None
    try:
        return dt.datetime.fromisoformat(x)
    except Exception:
        return None

def dt_to_iso(x: Optional[dt.datetime]) -> Optional[str]:
    return x.isoformat() if x else None


# ---------------- BITI MARKET-CLOSE RULES ----------------
def biti_allowed_now() -> bool:
    """
    BITI is allowed only when:
      - market is open
      - AND we are NOT inside the forced-close window
    """
    if not market_is_open():
        return False
    m = minutes_to_close()
    if m is None:
        return True
    return m > SELL_BITI_BEFORE_CLOSE_MIN

def can_trade_biti_now() -> bool:
    return biti_allowed_now()


# ---------------- DATA FETCHING ----------------
def get_latest_btc_close_1m() -> Tuple[dt.datetime, float]:
    """
    Pulls latest 1-minute crypto bar close from Alpaca for BTC/USD.
    """
    end = now_utc()
    start = end - dt.timedelta(minutes=15)

    bars = alpaca.get_crypto_bars(
        BTC_TRADE_SYMBOL,
        TimeFrame(1, TimeFrameUnit.Minute),
        start.isoformat(),
        end.isoformat(),
    ).df

    if bars is None or bars.empty:
        raise RuntimeError("No BTC 1m bars returned")

    bar_time = bars.index[-1].to_pydatetime()
    if bar_time.tzinfo is None:
        bar_time = bar_time.replace(tzinfo=dt.timezone.utc)

    price = float(bars["close"].iloc[-1])
    return bar_time, price

def get_biti_last_price() -> Optional[float]:
    """
    Best-effort: get latest trade for BITI from Alpaca.
    Only valid during market hours.
    """
    try:
        t = alpaca.get_latest_trade(BITI_SYMBOL)
        return float(getattr(t, "price", 0.0))
    except Exception:
        return None


# ---------------- POSITIONS / ORDERS ----------------
def safe_get_position_qty(symbol: str) -> float:
    try:
        pos = alpaca.get_position(symbol)
        return float(pos.qty)
    except Exception:
        return 0.0

def btc_qty() -> float:
    for sym in BTC_POS_SYMBOLS:
        q = safe_get_position_qty(sym)
        if q > 0:
            return q
    return 0.0

def biti_qty() -> float:
    return safe_get_position_qty(BITI_SYMBOL)

def has_open_order_for(symbols: set[str]) -> bool:
    try:
        orders = alpaca.list_orders(status="open", limit=200)
        for o in orders:
            sym = getattr(o, "symbol", "")
            if sym in symbols:
                return True
        return False
    except Exception:
        # If order check fails, be conservative and assume "yes" to avoid duplicates
        return True


# ---------------- ORDER PLACEMENT ----------------
def place_crypto_buy_notional(usd: float) -> str:
    o = alpaca.submit_order(
        symbol=BTC_TRADE_SYMBOL,
        notional=round(usd, 2),
        side="buy",
        type="market",
        time_in_force="gtc",
    )
    return getattr(o, "id", "")

def place_crypto_sell_qty(qty: float) -> str:
    o = alpaca.submit_order(
        symbol=BTC_TRADE_SYMBOL,
        qty=str(qty),
        side="sell",
        type="market",
        time_in_force="gtc",
    )
    return getattr(o, "id", "")

def place_equity_buy_notional(symbol: str, usd: float) -> str:
    o = alpaca.submit_order(
        symbol=symbol,
        notional=round(usd, 2),
        side="buy",
        type="market",
        time_in_force="day",
    )
    return getattr(o, "id", "")

def place_equity_sell_qty(symbol: str, qty: float) -> str:
    o = alpaca.submit_order(
        symbol=symbol,
        qty=str(qty),
        side="sell",
        type="market",
        time_in_force="day",
    )
    return getattr(o, "id", "")


# ---------------- SIGNAL ----------------
@dataclass(frozen=True)
class Signal:
    direction: str  # "UP", "DOWN", "FLAT"
    move_pct: float

def update_price_window(state: Dict[str, Any], new_price: float) -> None:
    w: List[float] = state["meta"].get("price_window", [])
    w.append(float(new_price))
    while len(w) > UPTREND_WINDOW:
        w.pop(0)
    state["meta"]["price_window"] = w

def compute_signal_from_window(window: List[float]) -> Signal:
    if len(window) < 3:
        return Signal("FLAT", 0.0)
    p0, p1, p2 = window[-3], window[-2], window[-1]

    if p0 < p1 < p2:
        direction = "UP"
    elif p0 > p1 > p2:
        direction = "DOWN"
    else:
        direction = "FLAT"

    move_pct = abs((p2 - p0) / p0) if p0 != 0 else 0.0
    return Signal(direction, move_pct)


# ---------------- TRADING LOGIC ----------------
def compute_buy_usd() -> float:
    usable = min(spendable_usd(), STARTING_MONEY_USD, MAX_TRADE_USD)
    return usable

def force_exit_biti_if_close_soon() -> Tuple[bool, bool]:
    """
    Returns (sold_biti, in_close_window).

    in_close_window means: market is open AND minutes_to_close <= SELL_BITI_BEFORE_CLOSE_MIN
    When in_close_window is True, the main loop should NOT buy BITI and should skip
    signal trading for this iteration to avoid re-buying after a forced sell.
    """
    if not market_is_open():
        return (False, False)

    m = minutes_to_close()
    if m is None:
        return (False, False)

    in_close_window = (m <= SELL_BITI_BEFORE_CLOSE_MIN)

    if in_close_window and biti_qty() > 0:
        qty = biti_qty()
        oid = place_equity_sell_qty(BITI_SYMBOL, qty)
        log_line(f"FORCE: SELL BITI before close (T-{m:.1f}m) qty={qty:.6f} order_id={oid}")
        return (True, True)

    return (False, in_close_window)

def rotate_to_btc(price: float) -> bool:
    """
    Ensure we hold BTC and NOT BITI.
    """
    if has_open_order_for({BTC_TRADE_SYMBOL, BITI_SYMBOL, "BTCUSD"}):
        log_line("SKIP: open order exists (BTC/BITI) -> no rotate_to_btc")
        return False

    did = False

    # sell BITI first ONLY if allowed
    if can_trade_biti_now():
        q_biti = biti_qty()
        if q_biti > 0:
            oid = place_equity_sell_qty(BITI_SYMBOL, q_biti)
            log_line(f"ACTION: SELL BITI ALL (bullish) qty={q_biti:.6f} order_id={oid}")
            did = True

    # if already holding BTC, do nothing
    q_btc = btc_qty()
    if q_btc > 0:
        log_line(f"INFO: already holding BTC qty={q_btc:.8f} -> no buy")
        return did

    buy_usd = compute_buy_usd()
    cash, bp = account_cash_and_bp()
    log_line(f"DEBUG: rotate_to_btc cash=${cash:.2f} bp=${bp:.2f} spendable=${spendable_usd():.2f} buy_usd=${buy_usd:.2f}")

    if buy_usd >= MIN_TRADE_USD:
        oid = place_crypto_buy_notional(buy_usd)
        log_line(f"ACTION: BUY BTC notional=${buy_usd:.2f} order_id={oid} price~{price:.2f}")
        return True

    log_line("INFO: not enough spendable USD to buy BTC.")
    return did

def rotate_to_biti() -> bool:
    """
    Ensure we hold BITI and NOT BTC (market hours only, and not in close-window).
    """
    if not can_trade_biti_now():
        log_line("INFO: BITI disabled (market closed or close-window) -> skipping rotate_to_biti")
        return False

    if has_open_order_for({BTC_TRADE_SYMBOL, BITI_SYMBOL, "BTCUSD"}):
        log_line("SKIP: open order exists (BTC/BITI) -> no rotate_to_biti")
        return False

    did = False

    # sell BTC first (crypto is allowed anytime)
    q_btc = btc_qty()
    if q_btc > 0:
        oid = place_crypto_sell_qty(q_btc)
        log_line(f"ACTION: SELL BTC ALL (bearish->BITI) qty={q_btc:.8f} order_id={oid}")
        did = True

    # if already holding BITI, do nothing
    q_biti = biti_qty()
    if q_biti > 0:
        log_line(f"INFO: already holding BITI qty={q_biti:.6f} -> no buy")
        return did

    buy_usd = compute_buy_usd()
    cash, bp = account_cash_and_bp()
    log_line(f"DEBUG: rotate_to_biti cash=${cash:.2f} bp=${bp:.2f} spendable=${spendable_usd():.2f} buy_usd=${buy_usd:.2f}")

    if buy_usd >= MIN_TRADE_USD:
        oid = place_equity_buy_notional(BITI_SYMBOL, buy_usd)
        log_line(f"ACTION: BUY BITI notional=${buy_usd:.2f} order_id={oid}")
        return True

    log_line("INFO: not enough spendable USD to buy BITI.")
    return did

def rotate_to_cash() -> bool:
    """
    Sell BTC (and BITI if possible) and hold cash.
    """
    if has_open_order_for({BTC_TRADE_SYMBOL, BITI_SYMBOL, "BTCUSD"}):
        log_line("SKIP: open order exists (BTC/BITI) -> no rotate_to_cash")
        return False

    did = False

    q_btc = btc_qty()
    if q_btc > 0:
        oid = place_crypto_sell_qty(q_btc)
        log_line(f"ACTION: SELL BTC ALL -> qty={q_btc:.8f} order_id={oid}")
        did = True

    # sell BITI only if allowed
    if can_trade_biti_now():
        q_biti = biti_qty()
        if q_biti > 0:
            oid = place_equity_sell_qty(BITI_SYMBOL, q_biti)
            log_line(f"ACTION: SELL BITI ALL -> qty={q_biti:.6f} order_id={oid}")
            did = True

    return did


# ---------------- REPORTING ----------------
def should_send_report_open(state: Dict[str, Any]) -> bool:
    if not ENABLE_REPORTS:
        return False
    if not market_is_open():
        return False
    last_iso = state["meta"].get("last_report_time")
    last = iso_to_dt(last_iso)
    if last is None:
        return True
    return (now_utc() - last).total_seconds() >= REPORT_EVERY_SECONDS_OPEN

def make_report(state: Dict[str, Any], btc_price: float, biti_price: Optional[float]) -> str:
    cash, bp = account_cash_and_bp()
    sp = spendable_usd()

    w = state["meta"].get("price_window", [])
    sig = state["meta"].get("last_signal", "NA")
    mv = float(state["meta"].get("last_move_pct", 0.0))

    last_rep = state["meta"].get("last_report_prices", {})
    last_btc = last_rep.get("BTC")
    last_biti = last_rep.get("BITI")

    def pct(a: Optional[float], b: Optional[float]) -> str:
        if a is None or b is None or a == 0:
            return "NA"
        return f"{((b-a)/a*100):+.3f}%"

    lines = []
    lines.append(f"Signal: {sig} | move(3m): {mv*100:.3f}% | window={','.join([f'{x:.2f}' for x in w])}")
    lines.append(f"Holdings: BTC_QTY={btc_qty():.8f} | BITI_QTY={biti_qty():.6f}")
    lines.append(f"Account: cash=${cash:.2f} | buying_power=${bp:.2f} | spendable=${sp:.2f} (min(cash,bp))")
    lines.append("")
    lines.append("ASSET | LAST_REPORT | CURRENT | %vsLAST_REPORT")
    lines.append("------|------------|---------|--------------")
    lines.append(f"BTC   | {(f'{last_btc:.2f}' if last_btc is not None else 'NA'):>10} | {btc_price:>7.2f} | {pct(last_btc, btc_price):>12}")
    if biti_price is None:
        lines.append(f"BITI  | {(f'{last_biti:.2f}' if last_biti is not None else 'NA'):>10} | {'NA':>7} | {'NA':>12}")
    else:
        lines.append(f"BITI  | {(f'{last_biti:.2f}' if last_biti is not None else 'NA'):>10} | {biti_price:>7.2f} | {pct(last_biti, biti_price):>12}")
    return "\n".join(lines)

def update_last_report_prices(state: Dict[str, Any], btc_price: float, biti_price: Optional[float]) -> None:
    state["meta"]["last_report_prices"]["BTC"] = btc_price
    if biti_price is not None:
        state["meta"]["last_report_prices"]["BITI"] = biti_price
    state["meta"]["last_report_time"] = dt_to_iso(now_utc())


# ---------------- MAIN LOOP ----------------
def main() -> None:
    state = load_state()
    log_line("=== BITCOIN DAYTRADE BOT v2 START ===")
    log_line(f"Config: CHECK={CHECK_SECONDS}s SAVE={SAVE_EVERY_SECONDS}s REPORT_OPEN={REPORT_EVERY_SECONDS_OPEN}s")
    log_line(f"Symbols: BTC_TRADE={BTC_TRADE_SYMBOL} BTC_POS={BTC_POS_SYMBOLS} BITI={BITI_SYMBOL}")
    log_line(f"Sizing: STARTING=${STARTING_MONEY_USD:.2f} MAX_TRADE=${MAX_TRADE_USD:.2f}")
    log_line(f"BITI close rule: SELL_BITI_BEFORE_CLOSE_MIN={SELL_BITI_BEFORE_CLOSE_MIN} (BITI disabled during close-window)")

    # startup email
    if ENABLE_REPORTS and ENABLE_STARTUP_REPORT:
        try:
            btc_time, btc_price = get_latest_btc_close_1m()
            biti_price = get_biti_last_price() if market_is_open() else None
            body = "Bot initialized and running.\n\n" + make_report(state, btc_price, biti_price)
            sendReport(body=body, subject_prefix="Startup BTC DayTrade v2")
            log_line("EMAIL: startup report sent")
        except Exception as e:
            log_line(f"EMAIL ERROR (startup): {repr(e)}")

    last_save_time = now_utc()

    while True:
        try:
            # --- MARKET CLOSE WINDOW LOGIC (GLOBAL GUARD) ---
            sold_biti, in_close_window = (False, False)
            try:
                sold_biti, in_close_window = force_exit_biti_if_close_soon()
            except Exception as e:
                log_line(f"ERROR: force_exit_biti_if_close_soon -> {repr(e)}")

            # If we're in the close window, NEVER trade BITI and also skip signal trading
            # for this minute to prevent any re-buy after a forced sell.
            if in_close_window:
                log_line("INFO: In market-close window -> BITI disabled. Skipping signal trading this minute.")

                # Reporting still allowed (market open), saving still allowed
                if should_send_report_open(state):
                    try:
                        btc_time, btc_price = get_latest_btc_close_1m()
                        body = make_report(state, btc_price, None)  # BITI price not meaningful in close-window
                        sendReport(body=body, subject_prefix="BTC DayTrade v2 Report (Close Window)")
                        log_line("EMAIL: close-window report sent")
                        update_last_report_prices(state, btc_price, None)
                    except Exception as e:
                        log_line(f"EMAIL ERROR (report close-window): {repr(e)}")

                if (now_utc() - last_save_time).total_seconds() >= SAVE_EVERY_SECONDS:
                    try:
                        save_state(state)
                        last_save_time = now_utc()
                        log_line("STATE: saved")
                    except Exception as e:
                        log_line(f"ERROR: save_state -> {repr(e)}")

                time.sleep(CHECK_SECONDS)
                continue

            # If any open order exists, don't place new ones
            if has_open_order_for({BTC_TRADE_SYMBOL, "BTCUSD", BITI_SYMBOL}):
                log_line("SKIP: open order exists (BTC/BITI)")
            else:
                bar_time, btc_price = get_latest_btc_close_1m()

                # sliding window should advance only on new minute bar
                last_bar_iso = state["meta"].get("last_bar_time")
                last_bar_dt = iso_to_dt(last_bar_iso)
                if last_bar_dt is not None and bar_time <= last_bar_dt:
                    log_line(f"SKIP: already processed BTC 1m bar @ {bar_time.isoformat()} price={btc_price:.2f}")
                else:
                    state["meta"]["last_bar_time"] = dt_to_iso(bar_time)

                    # record prices in BSTs
                    ts = time.time()
                    state["trees"]["BTC"].insert(ts, btc_price)

                    # Only track BITI price when market open
                    biti_price = get_biti_last_price() if market_is_open() else None
                    if biti_price is not None:
                        state["trees"]["BITI"].insert(ts, biti_price)

                    # update sliding 3-minute window and compute signal
                    update_price_window(state, btc_price)
                    sig = compute_signal_from_window(state["meta"]["price_window"])
                    state["meta"]["last_signal"] = sig.direction
                    state["meta"]["last_move_pct"] = sig.move_pct

                    # log snapshot (file only)
                    cash, bp = account_cash_and_bp()
                    log_line(
                        f"BTC 1m close={btc_price:.2f} bar={bar_time.isoformat()} | "
                        f"signal={sig.direction} move={sig.move_pct*100:.3f}% | "
                        f"market_open={market_is_open()} | BITI_allowed={biti_allowed_now()} | "
                        f"BTC_QTY={btc_qty():.8f} | BITI_QTY={biti_qty():.6f} | "
                        f"CASH=${cash:.2f} | BP=${bp:.2f} | spendable=${spendable_usd():.2f}"
                    )

                    # cooldown
                    last_trade_dt = iso_to_dt(state["meta"].get("last_trade_time"))
                    if last_trade_dt is None:
                        last_trade_dt = now_utc()
                        state["meta"]["last_trade_time"] = dt_to_iso(last_trade_dt)

                    since_trade = (now_utc() - last_trade_dt).total_seconds()
                    if since_trade < COOLDOWN_SECONDS:
                        log_line(f"COOLDOWN: {int(since_trade)}s < {COOLDOWN_SECONDS}s -> HOLD")
                    else:
                        strong = sig.move_pct >= MIN_MOVE_PCT

                        if sig.direction == "UP" and strong:
                            did = rotate_to_btc(btc_price)
                            if did:
                                state["meta"]["last_trade_time"] = dt_to_iso(now_utc())

                        elif sig.direction == "DOWN" and strong:
                            if can_trade_biti_now():
                                did = rotate_to_biti()
                                if did:
                                    state["meta"]["last_trade_time"] = dt_to_iso(now_utc())
                            else:
                                # BITI not allowed (close-window) OR market closed: rotate to CASH
                                log_line("INFO: bearish but BITI not allowed -> rotating to CASH")
                                did = rotate_to_cash()
                                if did:
                                    state["meta"]["last_trade_time"] = dt_to_iso(now_utc())

                        else:
                            # FLAT or not strong: initial allocation logic if nothing held
                            if btc_qty() == 0 and biti_qty() == 0:
                                if market_is_open():
                                    if DEFAULT_WHEN_FLAT_MARKET_OPEN == "BTC":
                                        did = rotate_to_btc(btc_price)
                                        if did:
                                            state["meta"]["last_trade_time"] = dt_to_iso(now_utc())
                                    else:
                                        log_line("HOLD: FLAT/no-strong signal; default is CASH (market open).")
                                else:
                                    if DEFAULT_WHEN_FLAT_MARKET_CLOSED == "BTC":
                                        did = rotate_to_btc(btc_price)
                                        if did:
                                            state["meta"]["last_trade_time"] = dt_to_iso(now_utc())
                                    else:
                                        log_line("HOLD: FLAT/no-strong signal; default is CASH (market closed).")
                            else:
                                log_line("HOLD: signal not clean/strong enough.")

                    # Reporting (market open only)
                    if should_send_report_open(state):
                        try:
                            biti_price_for_report = get_biti_last_price() if market_is_open() else None
                            body = make_report(state, btc_price, biti_price_for_report)
                            sendReport(body=body, subject_prefix="BTC DayTrade v2 Report")
                            log_line("EMAIL: open-market report sent")
                            update_last_report_prices(state, btc_price, biti_price_for_report)
                        except Exception as e:
                            log_line(f"EMAIL ERROR (report): {repr(e)}")

            # periodic save
            if (now_utc() - last_save_time).total_seconds() >= SAVE_EVERY_SECONDS:
                try:
                    save_state(state)
                    last_save_time = now_utc()
                    log_line("STATE: saved")
                except Exception as e:
                    log_line(f"ERROR: save_state -> {repr(e)}")

        except KeyboardInterrupt:
            log_line("INFO: KeyboardInterrupt -> saving state and exiting")
            try:
                save_state(state)
                log_line("STATE: saved on exit")
            except Exception as e:
                log_line(f"ERROR: save on exit -> {repr(e)}")
            break
        except Exception as e:
            log_line(f"ERROR: main loop -> {repr(e)}")

        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    main()
