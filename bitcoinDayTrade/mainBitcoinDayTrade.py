from __future__ import annotations

import os
import time
import datetime as dt
from dataclasses import dataclass
from dotenv import load_dotenv

import alpaca_trade_api as tradeapi
from alpaca_trade_api import TimeFrame
from alpaca_trade_api.rest import TimeFrameUnit

from gmail import sendReport
from state import load_state, save_state, get_dt, set_dt

load_dotenv()

# ---------------- CONFIG ----------------
CHECK_SECONDS = .5 * 60  # 1 minute checks (bars are 5m, so you will see SKIPs between new bars)

BTC_TRADE_SYMBOL = "BTC/USD"   # submit_order + bars
BTC_POS_SYMBOL = "BTCUSD"      # get_position + list_orders filtering

BITI_SYMBOL = "BITI"

STARTING_MONEY_USD = 100000.00   # target max deployment
MAX_TRADE_USD = 25000.00         # hard cap per trade (SAFETY). Set to 100000 if you really want.

MIN_MOVE_PCT_3BARS = 0.0015      # 0.15% over last 3 bars
COOLDOWN_SECONDS = 60 * .5       # 1 minute cooldown (aggressive)
MIN_TRADE_USD = 1.00

LOG_FILE = "trade_log.txt"

REPORT_EVERY_SECONDS_OPEN = 2 * 60 * 60
SEND_STARTUP_REPORT = True

DEFAULT_ALLOCATION_WHEN_CASH_ONLY_MARKET_CLOSED = "BTC"
DEFAULT_ALLOCATION_WHEN_CASH_ONLY_MARKET_OPEN = "BTC"

# ---------------- ALPACA SETUP ----------------
API_KEY = os.getenv("alpacaKey")
API_SECRET = os.getenv("alpacaSecret")
BASE_URL = os.getenv("alpacaBaseURL", "https://paper-api.alpaca.markets")

alpaca = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")


# ---------------- UTILITIES ----------------
def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def log_line(line: str) -> None:
    stamp = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    out = f"[{stamp}] {line}"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(out + "\n")


def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def get_account_snapshot() -> tuple[float, float]:
    """
    Returns (cash, buying_power) as floats.
    """
    acct = alpaca.get_account()
    cash = safe_float(getattr(acct, "cash", 0.0))
    bp = safe_float(getattr(acct, "buying_power", 0.0))
    return cash, bp


def get_spendable_usd() -> float:
    """
    Use buying_power as the spendable amount because cash may not be spendable.
    """
    _, bp = get_account_snapshot()
    return max(0.0, bp - 5.00)  # small buffer


def market_is_open() -> bool:
    c = alpaca.get_clock()
    return bool(c.is_open)


def get_latest_btc_close_5m() -> tuple[dt.datetime, float]:
    end = now_utc()
    start = end - dt.timedelta(hours=2)

    bars = alpaca.get_crypto_bars(
        BTC_TRADE_SYMBOL,
        TimeFrame(5, TimeFrameUnit.Minute),
        start.isoformat(),
        end.isoformat(),
    ).df

    if bars is None or bars.empty:
        raise RuntimeError("No BTC 5m bars returned")

    bar_time = bars.index[-1].to_pydatetime()
    if bar_time.tzinfo is None:
        bar_time = bar_time.replace(tzinfo=dt.timezone.utc)

    price = float(bars["close"].iloc[-1])
    return bar_time, price


def get_last_n_btc_closes_5m(n: int) -> list[float]:
    end = now_utc()
    start = end - dt.timedelta(hours=3)

    bars = alpaca.get_crypto_bars(
        BTC_TRADE_SYMBOL,
        TimeFrame(5, TimeFrameUnit.Minute),
        start.isoformat(),
        end.isoformat(),
    ).df

    if bars is None or bars.empty or len(bars) < n:
        raise RuntimeError(f"Need at least {n} BTC 5m bars; got {0 if bars is None else len(bars)}")

    return [float(x) for x in bars["close"].iloc[-n:].tolist()]


def safe_get_position_qty(symbol: str) -> float:
    try:
        pos = alpaca.get_position(symbol)
        return float(pos.qty)
    except Exception:
        return 0.0


def has_any_holdings() -> bool:
    btc_qty = safe_get_position_qty(BTC_POS_SYMBOL)
    biti_qty = safe_get_position_qty(BITI_SYMBOL)
    return (btc_qty > 0) or (biti_qty > 0)


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


def list_open_orders_for(symbols: set[str]) -> list:
    try:
        orders = alpaca.list_orders(status="open", limit=200)
        return [o for o in orders if getattr(o, "symbol", "") in symbols]
    except Exception:
        return []


def build_report_since(ts: dt.datetime) -> str:
    if not os.path.exists(LOG_FILE):
        return "No log file yet."

    cutoff = ts.strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("[") and line[1:20] >= cutoff:
                lines.append(line.rstrip())

    return "No activity during this window." if not lines else "\n".join(lines[-500:])


@dataclass(frozen=True)
class Signal:
    direction: str
    move_pct: float


def compute_signal_from_last3(closes4: list[float]) -> Signal:
    c0, c1, c2, c3 = closes4
    d1 = c1 - c0
    d2 = c2 - c1
    d3 = c3 - c2

    if d1 < 0 and d2 < 0 and d3 < 0:
        direction = "DOWN"
    elif d1 > 0 and d2 > 0 and d3 > 0:
        direction = "UP"
    else:
        direction = "FLAT"

    move_pct = abs((c3 - c0) / c0) if c0 != 0 else 0.0
    return Signal(direction=direction, move_pct=move_pct)


# ---------------- TRADING LOGIC ----------------
def compute_buy_usd() -> float:
    """
    How much to deploy on a buy right now.
    Uses buying_power, capped by STARTING_MONEY_USD and MAX_TRADE_USD.
    """
    spendable = get_spendable_usd()
    buy_usd = min(spendable, STARTING_MONEY_USD, MAX_TRADE_USD)
    return buy_usd


def rotate_to_cash() -> bool:
    did = False

    btc_qty = safe_get_position_qty(BTC_POS_SYMBOL)
    if btc_qty > 0:
        oid = place_crypto_sell_qty(btc_qty)
        log_line(f"ACTION: SELL BTC ALL -> order_id={oid} qty={btc_qty:.8f}")
        did = True

    biti_qty = safe_get_position_qty(BITI_SYMBOL)
    if biti_qty > 0:
        oid = place_equity_sell_qty(BITI_SYMBOL, biti_qty)
        log_line(f"ACTION: SELL BITI ALL -> order_id={oid} qty={biti_qty:.6f}")
        did = True

    return did


def rotate_to_btc(price: float) -> bool:
    did = False

    # sell BITI first
    biti_qty = safe_get_position_qty(BITI_SYMBOL)
    if biti_qty > 0:
        oid = place_equity_sell_qty(BITI_SYMBOL, biti_qty)
        log_line(f"ACTION: SELL BITI ALL (bullish) -> order_id={oid} qty={biti_qty:.6f}")
        did = True

    cash, bp = get_account_snapshot()
    buy_usd = compute_buy_usd()

    log_line(f"DEBUG: rotate_to_btc cash=${cash:.2f} buying_power=${bp:.2f} buy_usd=${buy_usd:.2f}")

    if buy_usd >= MIN_TRADE_USD:
        oid = place_crypto_buy_notional(buy_usd)
        log_line(f"ACTION: BUY BTC notional=${buy_usd:.2f} -> order_id={oid} price~{price:.2f}")
        return True

    log_line("INFO: (bullish) Not enough buying power to buy BTC.")
    return did


def rotate_to_biti() -> bool:
    did = False

    # sell BTC first
    btc_qty = safe_get_position_qty(BTC_POS_SYMBOL)
    if btc_qty > 0:
        oid = place_crypto_sell_qty(btc_qty)
        log_line(f"ACTION: SELL BTC ALL (bearish->BITI) -> order_id={oid} qty={btc_qty:.8f}")
        did = True

    cash, bp = get_account_snapshot()
    buy_usd = compute_buy_usd()

    log_line(f"DEBUG: rotate_to_biti cash=${cash:.2f} buying_power=${bp:.2f} buy_usd=${buy_usd:.2f}")

    if buy_usd >= MIN_TRADE_USD:
        oid = place_equity_buy_notional(BITI_SYMBOL, buy_usd)
        log_line(f"ACTION: BUY BITI notional=${buy_usd:.2f} -> order_id={oid}")
        return True

    log_line("INFO: (bearish) Not enough buying power to buy BITI.")
    return did


# ---------------- MAIN LOOP ----------------
def main() -> None:
    log_line("=== DAY-ROTATION BOT START ===")
    log_line(f"BTC_TRADE={BTC_TRADE_SYMBOL} | BTC_POS={BTC_POS_SYMBOL} | BITI={BITI_SYMBOL} | STARTING=${STARTING_MONEY_USD:.2f}")

    state = load_state()

    last_bar_time = get_dt(state, "last_bar_time")
    last_trade_time = get_dt(state, "last_trade_time")
    last_report_time = get_dt(state, "last_report_time") or now_utc()
    last_open_report_time = get_dt(state, "last_open_report_time")

    if SEND_STARTUP_REPORT:
        try:
            cash, bp = get_account_snapshot()
            body = (
                "Bot initialized and running.\n"
                f"Time (UTC): {now_utc().isoformat()}\n"
                f"BTC_TRADE={BTC_TRADE_SYMBOL}, BTC_POS={BTC_POS_SYMBOL}, BITI={BITI_SYMBOL}\n"
                f"CHECK_EVERY={CHECK_SECONDS}s, REPORT_EVERY_OPEN={REPORT_EVERY_SECONDS_OPEN}s\n"
                f"Account snapshot: cash=${cash:.2f}, buying_power=${bp:.2f}\n"
            )
            sendReport(body=body, subject_prefix="Startup")
            log_line("EMAIL: startup report sent")
        except Exception as e:
            log_line(f"EMAIL ERROR (startup): {repr(e)}")

    while True:
        try:
            open_orders = list_open_orders_for({BTC_POS_SYMBOL, BTC_TRADE_SYMBOL, BITI_SYMBOL})
            if open_orders:
                log_line(f"SKIP: {len(open_orders)} open order(s) exist for BTC/BITI; waiting.")
            else:
                bar_time, price = get_latest_btc_close_5m()

                if last_bar_time is not None and bar_time <= last_bar_time:
                    log_line(f"SKIP: already processed BTC bar @ {bar_time.isoformat()} price={price:.2f}")
                else:
                    last_bar_time = bar_time
                    set_dt(state, "last_bar_time", last_bar_time)
                    save_state(state)

                    closes4 = get_last_n_btc_closes_5m(4)
                    sig = compute_signal_from_last3(closes4)
                    is_open = market_is_open()

                    btc_qty = safe_get_position_qty(BTC_POS_SYMBOL)
                    biti_qty = safe_get_position_qty(BITI_SYMBOL)
                    cash, bp = get_account_snapshot()

                    log_line(
                        f"BTC 5m close={price:.2f} | signal={sig.direction} move={sig.move_pct*100:.3f}% | "
                        f"market_open={is_open} | BTC_QTY={btc_qty:.8f} | BITI_QTY={biti_qty:.6f} | "
                        f"CASH=${cash:.2f} | BP=${bp:.2f}"
                    )

                    # INITIAL ALLOCATION
                    if not has_any_holdings():
                        log_line("INFO: No holdings detected (BTC=0, BITI=0). Considering initial allocation...")

                        if sig.direction in ("UP", "DOWN") and sig.move_pct >= MIN_MOVE_PCT_3BARS:
                            if sig.direction == "UP":
                                rotate_to_btc(price)
                            else:
                                if is_open:
                                    rotate_to_biti()
                                else:
                                    log_line("INFO: market closed; bearish initial -> staying in CASH (no BITI).")
                        else:
                            if not is_open:
                                if DEFAULT_ALLOCATION_WHEN_CASH_ONLY_MARKET_CLOSED == "BTC":
                                    log_line("INFO: market closed + no clean signal -> defaulting to BTC.")
                                    rotate_to_btc(price)
                                else:
                                    log_line("INFO: market closed + no clean signal -> staying in CASH.")
                            else:
                                if DEFAULT_ALLOCATION_WHEN_CASH_ONLY_MARKET_OPEN == "BTC":
                                    log_line("INFO: market open + no clean signal -> defaulting to BTC.")
                                    rotate_to_btc(price)
                                elif DEFAULT_ALLOCATION_WHEN_CASH_ONLY_MARKET_OPEN == "CASH":
                                    log_line("INFO: market open + no clean signal -> staying in CASH.")
                                else:
                                    log_line("INFO: market open + no clean signal -> WAIT (staying in CASH).")

                        last_trade_time = now_utc()
                        set_dt(state, "last_trade_time", last_trade_time)
                        save_state(state)

                    # NORMAL OPERATION
                    else:
                        if last_trade_time is None:
                            last_trade_time = now_utc()
                            set_dt(state, "last_trade_time", last_trade_time)
                            save_state(state)

                        since = (now_utc() - last_trade_time).total_seconds()
                        if since < COOLDOWN_SECONDS:
                            log_line(f"COOLDOWN: {int(since)}s since last trade (<{COOLDOWN_SECONDS}s) -> HOLD")
                        else:
                            if sig.direction in ("UP", "DOWN") and sig.move_pct >= MIN_MOVE_PCT_3BARS:
                                did_trade = False
                                if sig.direction == "UP":
                                    did_trade = rotate_to_btc(price)
                                else:
                                    if is_open:
                                        did_trade = rotate_to_biti()
                                    else:
                                        log_line("INFO: market closed; bearish -> rotate to CASH (no BITI)")
                                        did_trade = rotate_to_cash()

                                if did_trade:
                                    last_trade_time = now_utc()
                                    set_dt(state, "last_trade_time", last_trade_time)
                                    save_state(state)
                            else:
                                log_line("HOLD: signal not strong/clean enough (needs 3-in-a-row + min move).")

            # Reporting every 2 hours while market is open
            is_open_now = market_is_open()
            if is_open_now:
                if last_open_report_time is None:
                    last_open_report_time = now_utc()
                    set_dt(state, "last_open_report_time", last_open_report_time)
                    save_state(state)

                elapsed = (now_utc() - last_open_report_time).total_seconds()
                if elapsed >= REPORT_EVERY_SECONDS_OPEN:
                    body = build_report_since(last_report_time)
                    try:
                        sendReport(body=body, subject_prefix="Open-Market Report")
                        log_line("EMAIL: open-market 2-hour report sent")

                        last_report_time = now_utc()
                        set_dt(state, "last_report_time", last_report_time)

                        last_open_report_time = now_utc()
                        set_dt(state, "last_open_report_time", last_open_report_time)
                        save_state(state)
                    except Exception as e:
                        log_line(f"EMAIL ERROR (report): {repr(e)}")
            else:
                if last_open_report_time is not None:
                    log_line("INFO: market closed; pausing 2-hour report schedule until next open.")
                    last_open_report_time = None
                    set_dt(state, "last_open_report_time", None)
                    save_state(state)

        except Exception as e:
            log_line(f"ERROR: {repr(e)}")

        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    main()


