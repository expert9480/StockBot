from __future__ import annotations

import os
import time
import pickle
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import yfinance as yf

# ----------------- USER MODULES -----------------
# Your gmail.py should expose sendReport(body=..., subject_prefix=...)
from gmail import sendReport  # keep your existing function


# ----------------- CONFIG -----------------
load_dotenv()

LOG_FILE = "index_vix_bot.log"
STATE_FILE = "index_vix_state.pkl"

# Toggles
ENABLE_REPORTS = True
ENABLE_STARTUP_REPORT = True

# Timing
CHECK_SECONDS = 60  # 1 check per minute
SAVE_EVERY_SECONDS = 15 * 60
REPORT_EVERY_SECONDS_OPEN = 4 * 60 * 60  # every 4 hours while market open (changeable)

# Strategy thresholds
VIX_SELL_LEVEL = 28.0
VIX_REBUY_LEVEL = 40.0

# "3 checks increasing" requirement
UPTREND_WINDOW = 3  # 3 checks

STARTING_MONEY_USD = 100_000.00
MAX_TRADE_USD_PER_SYMBOL = 50_000.00  # safety cap per ETF buy; set higher if desired

# Watched ETFs (edit as needed)
# Using SPY + VOO + VT as "S&P 500 + S&P 500 + Total World"
WATCH_ETFS = ["SPY", "VOO", "VT"]
VIX_SYMBOL = "^VIX"

# Alpaca env vars
API_KEY = os.getenv("alpacaKey")
API_SECRET = os.getenv("alpacaSecret")
BASE_URL = os.getenv("alpacaBaseURL", "https://paper-api.alpaca.markets")


# ----------------- LOGGING (FILE ONLY) -----------------
def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def log_line(msg: str) -> None:
    ts = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ----------------- BST FOR INTRADAY PRICES -----------------
@dataclass
class BSTNode:
    key: float  # unix timestamp
    value: float  # price
    left: Optional["BSTNode"] = None
    right: Optional["BSTNode"] = None


class PriceBST:
    """
    A simple BST keyed by timestamp (float). Stores (timestamp -> price).
    Not self-balancing, but fine for intraday use.
    """
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
                # same timestamp: overwrite
                node.value = v
            return node

        self.root = _ins(self.root, key, value)

    def inorder(self) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []

        def _walk(node: Optional[BSTNode]) -> None:
            if node is None:
                return
            _walk(node.left)
            out.append((node.key, node.value))
            _walk(node.right)

        _walk(self.root)
        return out

    def last_value(self) -> Optional[float]:
        node = self.root
        if node is None:
            return None
        while node.right is not None:
            node = node.right
        return node.value


# ----------------- STATE -----------------
def default_state() -> Dict[str, Any]:
    return {
        "day": now_utc().date().isoformat(),
        "trees": {sym: PriceBST() for sym in (WATCH_ETFS + [VIX_SYMBOL])},
        "meta": {
            "mode": "INVESTED",  # INVESTED or SOLD_OUT
            "sell_snapshot": {},  # {sym: sell_price}
            "sell_time": None,    # iso timestamp
            "last_report_prices": {},  # {sym: price_at_last_report}
            "day_open_prices": {},     # {sym: open_price_for_day}
            "last_report_time": None,  # iso timestamp
            "last_save_time": None,    # iso timestamp
            "last_check_time": None,   # iso timestamp
            "uptick_buffer": {sym: [] for sym in WATCH_ETFS},  # last N prices per sym
        },
    }


def save_state(state: Dict[str, Any]) -> None:
    # Pickle whole object (BST included)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, STATE_FILE)
    state["meta"]["last_save_time"] = now_utc().isoformat()


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return default_state()

    try:
        with open(STATE_FILE, "rb") as f:
            s = pickle.load(f)

        # Day rollover -> reset trees/open prices but keep meta modes if you want.
        today = now_utc().date().isoformat()
        if s.get("day") != today:
            s = default_state()
        return s
    except Exception as e:
        log_line(f"ERROR: failed to load state -> {repr(e)}; starting fresh.")
        return default_state()


# ----------------- MARKET DATA -----------------
def yf_last_price(symbol: str) -> float:
    """
    Uses yfinance fast path; falls back safely.
    """
    t = yf.Ticker(symbol)
    # fast_info is usually present; if not, fallback to history
    price = None
    try:
        price = t.fast_info.get("last_price", None)
    except Exception:
        price = None

    if price is None:
        hist = t.history(period="1d", interval="1m")
        if hist is None or hist.empty:
            raise RuntimeError(f"No yfinance data for {symbol}")
        price = float(hist["Close"].iloc[-1])

    return float(price)


def yf_day_open_price(symbol: str) -> float:
    """
    Gets today's open from 1d data.
    """
    t = yf.Ticker(symbol)
    hist = t.history(period="1d", interval="1d")
    if hist is None or hist.empty:
        # fallback
        hist = t.history(period="5d", interval="1d")
        if hist is None or hist.empty:
            raise RuntimeError(f"No open price data for {symbol}")
        # use most recent row
    return float(hist["Open"].iloc[-1])


# ----------------- ALPACA -----------------
alpaca = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")


def market_is_open() -> bool:
    c = alpaca.get_clock()
    return bool(c.is_open)


def account_cash_and_bp() -> Tuple[float, float]:
    acct = alpaca.get_account()
    cash = float(getattr(acct, "cash", 0.0))
    bp = float(getattr(acct, "buying_power", 0.0))
    return cash, bp


def spendable_usd() -> float:
    """
    If buying_power <= cash -> use buying_power
    If buying_power > cash -> use cash (avoid debt)
    """
    cash, bp = account_cash_and_bp()
    usable = bp if bp <= cash else cash
    return max(0.0, usable - 5.00)  # small buffer


def safe_position_qty(symbol: str) -> float:
    try:
        pos = alpaca.get_position(symbol)
        return float(pos.qty)
    except Exception:
        return 0.0


def list_open_orders_for(symbols: set[str]) -> List[Any]:
    try:
        orders = alpaca.list_orders(status="open", limit=200)
        return [o for o in orders if getattr(o, "symbol", "") in symbols]
    except Exception:
        return []


def submit_sell_all(symbol: str) -> bool:
    qty = safe_position_qty(symbol)
    if qty <= 0:
        return False
    o = alpaca.submit_order(
        symbol=symbol,
        qty=str(qty),
        side="sell",
        type="market",
        time_in_force="day",
    )
    log_line(f"ACTION: SELL {symbol} ALL qty={qty:.6f} order_id={getattr(o,'id','')}")
    return True


def submit_buy_notional(symbol: str, usd: float) -> bool:
    if usd <= 0:
        return False
    o = alpaca.submit_order(
        symbol=symbol,
        notional=round(usd, 2),
        side="buy",
        type="market",
        time_in_force="day",
    )
    log_line(f"ACTION: BUY {symbol} notional=${usd:.2f} order_id={getattr(o,'id','')}")
    return True


# ----------------- STRATEGY HELPERS -----------------
def update_uptick_buffer(state: Dict[str, Any], sym: str, price: float) -> None:
    buf: List[float] = state["meta"]["uptick_buffer"].setdefault(sym, [])
    buf.append(price)
    # keep last UPTREND_WINDOW prices
    while len(buf) > UPTREND_WINDOW:
        buf.pop(0)


def is_upticking_3(state: Dict[str, Any]) -> bool:
    """
    True if EACH ETF has 3 consecutive increasing checks (p0<p1<p2).
    If you want "any ETF" instead, change logic.
    """
    for sym in WATCH_ETFS:
        buf: List[float] = state["meta"]["uptick_buffer"].get(sym, [])
        if len(buf) < UPTREND_WINDOW:
            return False
        if not (buf[0] < buf[1] < buf[2]):
            return False
    return True


def all_etfs_below_sell_snapshot(state: Dict[str, Any], current_prices: Dict[str, float]) -> bool:
    snap: Dict[str, float] = state["meta"].get("sell_snapshot", {})
    if not snap:
        return False
    for sym in WATCH_ETFS:
        if sym not in snap:
            return False
        if current_prices[sym] >= snap[sym]:
            return False
    return True


def compute_per_symbol_buy_usd(state: Dict[str, Any]) -> Dict[str, float]:
    """
    Allocate spendable evenly across ETFs, capped per symbol and by STARTING_MONEY_USD.
    """
    usable = min(spendable_usd(), STARTING_MONEY_USD)
    if usable <= 0:
        return {sym: 0.0 for sym in WATCH_ETFS}

    per = usable / max(1, len(WATCH_ETFS))
    per = min(per, MAX_TRADE_USD_PER_SYMBOL)
    return {sym: per for sym in WATCH_ETFS}


# ----------------- REPORTING -----------------
def format_report(state: Dict[str, Any], prices: Dict[str, float]) -> str:
    """
    Includes: open, last-report, current, % changes
    """
    opens = state["meta"].get("day_open_prices", {})
    last_rep = state["meta"].get("last_report_prices", {})

    lines = []
    lines.append(f"Mode: {state['meta'].get('mode')}")
    cash, bp = account_cash_and_bp()
    lines.append(f"Account: cash=${cash:.2f} buying_power=${bp:.2f} spendable=${spendable_usd():.2f}")
    lines.append("")
    lines.append("SYMBOL | OPEN | LAST_REPORT | CURRENT | %vsOPEN | %vsLAST_REPORT")
    lines.append("-------|------|------------|---------|--------|--------------")

    for sym in (WATCH_ETFS + [VIX_SYMBOL]):
        o = opens.get(sym)
        lr = last_rep.get(sym)
        cur = prices.get(sym)

        def pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
            if a is None or b is None or a == 0:
                return None
            return (b - a) / a * 100.0

        p_open = pct(o, cur)
        p_lr = pct(lr, cur)

        lines.append(
            f"{sym:5} | "
            f"{(f'{o:.2f}' if o is not None else 'NA'):>6} | "
            f"{(f'{lr:.2f}' if lr is not None else 'NA'):>10} | "
            f"{(f'{cur:.2f}' if cur is not None else 'NA'):>7} | "
            f"{(f'{p_open:+.2f}%' if p_open is not None else 'NA'):>7} | "
            f"{(f'{p_lr:+.2f}%' if p_lr is not None else 'NA'):>12}"
        )

    return "\n".join(lines)


def maybe_send_report(state: Dict[str, Any], prices: Dict[str, float]) -> None:
    if not ENABLE_REPORTS:
        return
    if not market_is_open():
        return

    now = now_utc()
    last_iso = state["meta"].get("last_report_time")
    last = dt.datetime.fromisoformat(last_iso) if last_iso else None

    if last is not None:
        if (now - last).total_seconds() < REPORT_EVERY_SECONDS_OPEN:
            return

    # build + send
    body = format_report(state, prices)
    try:
        sendReport(body=body, subject_prefix="Index/VIX Report")
        log_line("EMAIL: report sent")
    except Exception as e:
        log_line(f"EMAIL ERROR: {repr(e)}")

    # update last report prices + time
    for sym, p in prices.items():
        state["meta"]["last_report_prices"][sym] = p
    state["meta"]["last_report_time"] = now.isoformat()


# ----------------- MAIN LOOP -----------------
def main() -> None:
    state = load_state()

    # Ensure trees exist
    for sym in (WATCH_ETFS + [VIX_SYMBOL]):
        state["trees"].setdefault(sym, PriceBST())

    # Startup: record opens once per day
    if not state["meta"].get("day_open_prices"):
        try:
            for sym in (WATCH_ETFS + [VIX_SYMBOL]):
                state["meta"]["day_open_prices"][sym] = yf_day_open_price(sym)
        except Exception as e:
            log_line(f"ERROR: failed to fetch open prices -> {repr(e)}")

    # Startup report
    if ENABLE_REPORTS and ENABLE_STARTUP_REPORT:
        try:
            prices = {sym: yf_last_price(sym) for sym in (WATCH_ETFS + [VIX_SYMBOL])}
            body = "Bot initialized and running.\n\n" + format_report(state, prices)
            sendReport(body=body, subject_prefix="Startup")
            log_line("EMAIL: startup report sent")
        except Exception as e:
            log_line(f"EMAIL ERROR (startup): {repr(e)}")

    last_save = None
    if state["meta"].get("last_save_time"):
        try:
            last_save = dt.datetime.fromisoformat(state["meta"]["last_save_time"])
        except Exception:
            last_save = None

    while True:
        try:
            # no console output; all logging goes to file only

            # day rollover safeguard
            today = now_utc().date().isoformat()
            if state.get("day") != today:
                log_line("INFO: day rollover -> resetting state")
                state = default_state()

            # Fetch prices (yfinance)
            prices: Dict[str, float] = {}
            for sym in (WATCH_ETFS + [VIX_SYMBOL]):
                prices[sym] = yf_last_price(sym)

            # Insert into BSTs
            ts = time.time()
            for sym, p in prices.items():
                state["trees"][sym].insert(ts, p)

            # Update uptick buffers for ETFs
            for sym in WATCH_ETFS:
                update_uptick_buffer(state, sym, prices[sym])

            # Update open prices if missing
            for sym in (WATCH_ETFS + [VIX_SYMBOL]):
                if sym not in state["meta"]["day_open_prices"]:
                    try:
                        state["meta"]["day_open_prices"][sym] = yf_day_open_price(sym)
                    except Exception:
                        pass

            # Avoid trading if open orders exist
            open_orders = list_open_orders_for(set(WATCH_ETFS))
            if open_orders:
                log_line(f"SKIP: {len(open_orders)} open order(s) exist; waiting.")
            else:
                vix = prices[VIX_SYMBOL]
                mode = state["meta"].get("mode", "INVESTED")

                # SELL RULE
                if mode == "INVESTED" and vix >= VIX_SELL_LEVEL:
                    if market_is_open():
                        log_line(f"TRIGGER: VIX={vix:.2f} >= {VIX_SELL_LEVEL:.2f} -> SELL ALL")
                        any_sold = False
                        for sym in WATCH_ETFS:
                            any_sold = submit_sell_all(sym) or any_sold

                        # record snapshot even if you already had 0 shares
                        state["meta"]["sell_snapshot"] = {sym: prices[sym] for sym in WATCH_ETFS}
                        state["meta"]["sell_time"] = now_utc().isoformat()
                        state["meta"]["mode"] = "SOLD_OUT"

                        if not any_sold:
                            log_line("INFO: sell trigger hit but no positions were held (still switching to SOLD_OUT).")
                    else:
                        log_line(f"INFO: VIX sell trigger hit (VIX={vix:.2f}) but market closed; no action.")

                # REBUY RULE
                elif mode == "SOLD_OUT":
                    # Only consider rebuy when market open
                    if market_is_open():
                        snap_ok = all_etfs_below_sell_snapshot(state, prices)
                        if vix >= VIX_REBUY_LEVEL and snap_ok:
                            uptick = is_upticking_3(state)
                            if uptick:
                                log_line(
                                    f"TRIGGER: VIX={vix:.2f} >= {VIX_REBUY_LEVEL:.2f} "
                                    f"+ all ETFs below sell snapshot + 3-check uptick -> REBUY"
                                )
                                alloc = compute_per_symbol_buy_usd(state)
                                for sym in WATCH_ETFS:
                                    usd = alloc[sym]
                                    if usd >= 1.0:
                                        submit_buy_notional(sym, usd)

                                state["meta"]["mode"] = "INVESTED"
                                # reset sell snapshot after rebuy
                                state["meta"]["sell_snapshot"] = {}
                                state["meta"]["sell_time"] = None
                            else:
                                log_line(
                                    f"WAIT: rebuy conditions met except uptick. "
                                    f"VIX={vix:.2f} snap_ok={snap_ok} uptick={uptick}"
                                )
                        else:
                            log_line(
                                f"WAIT: SOLD_OUT. VIX={vix:.2f} (need >= {VIX_REBUY_LEVEL:.2f}) "
                                f"snap_ok={snap_ok}"
                            )

            # Reporting
            maybe_send_report(state, prices)

            # Periodic save
            now = now_utc()
            if last_save is None or (now - last_save).total_seconds() >= SAVE_EVERY_SECONDS:
                save_state(state)
                last_save = now
                log_line("STATE: saved")

        except KeyboardInterrupt:
            log_line("INFO: KeyboardInterrupt -> saving state and exiting")
            try:
                save_state(state)
                log_line("STATE: saved on exit")
            except Exception as e:
                log_line(f"ERROR: failed saving on exit -> {repr(e)}")
            break
        except Exception as e:
            # never crash silently
            log_line(f"ERROR: {repr(e)}")

        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    main()
