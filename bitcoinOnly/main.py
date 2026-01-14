from __future__ import annotations

import os
import time
import datetime as dt
from dotenv import load_dotenv

import alpaca_trade_api as tradeapi
from alpaca_trade_api import TimeFrame

from rainbow import get_levels
from gmail import sendReport
from state import load_state, save_state, get_dt, set_dt

load_dotenv()

# ---------------- CONFIG ----------------
STARTING_MONEY_USD = 100000.00  # <--- change this

SYMBOL = "BTCUSD"  # Alpaca crypto pair commonly used in examples :contentReference[oaicite:4]{index=4}
LOG_FILE = "trade_log.txt"

PRICE_CHECK_SECONDS = 60
REPORT_EVERY_SECONDS = 12 * 60 * 60  # 12 hours

# If you want to avoid “churn” from tiny differences:
MIN_TRADE_USD = 5.00

TRADE_COOLDOWN_SECONDS = 10 * 60  # 10 minutes cooldown to avoid spam buys


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
    print(out)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(out + "\n")


def get_latest_btc_price() -> float:
    """
    Gets the latest minute close using get_crypto_bars(TimeFrame.Minute).
    This pattern is commonly used with alpaca_trade_api. :contentReference[oaicite:5]{index=5}
    """
    end = now_utc()
    start = end - dt.timedelta(minutes=3)

    bars = alpaca.get_crypto_bars(
        SYMBOL,
        TimeFrame.Minute,
        start.isoformat(),
        end.isoformat(),
    ).df

    if bars is None or bars.empty:
        raise RuntimeError("No crypto bars returned")

    # last close
    return float(bars["close"].iloc[-1])


def safe_get_position_qty(symbol: str) -> float:
    try:
        pos = alpaca.get_position(symbol)
        return float(pos.qty)
    except Exception:
        return 0.0


def safe_get_cash() -> float:
    # For crypto, buying power/cash can vary by account type; keep it simple:
    acct = alpaca.get_account()
    # 'cash' exists for most accounts
    return float(getattr(acct, "cash", 0.0))


def submit_market_buy_notional(symbol: str, usd: float) -> None:
    alpaca.submit_order(
        symbol=symbol,
        notional=round(usd, 2),
        side="buy",
        type="market",
        time_in_force="gtc",  # crypto supports gtc/ioc :contentReference[oaicite:6]{index=6}
    )


def submit_market_sell_qty(symbol: str, qty: float) -> None:
    alpaca.submit_order(
        symbol=symbol,
        qty=str(qty),
        side="sell",
        type="market",
        time_in_force="gtc",
    )


def build_report(last_report_time: dt.datetime) -> str:
    # Tail the log file since last report time (simple approach)
    if not os.path.exists(LOG_FILE):
        return "No log file yet."

    lines = []
    cutoff = last_report_time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            # crude filter: include all lines after the cutoff timestamp substring
            # (good enough for a simple bot report)
            if line.startswith("[") and line[1:20] >= cutoff:
                lines.append(line.rstrip())

    if not lines:
        return "No activity in the last 12 hours."

    return "\n".join(lines[-400:])  # cap size so email doesn't explode


# ---------------- STRATEGY ----------------
def strategy_step(price: float, last_trade_time: dt.datetime | None) -> bool:
    levels = get_levels(now_utc())

    blue = levels.blue_level
    green = levels.green_level
    yellow = levels.yellow_level

    btc_qty = safe_get_position_qty(SYMBOL)
    cash = safe_get_cash()

    btc_value = btc_qty * price

        # Cooldown check
    if last_trade_time is not None:
        since = (now_utc() - last_trade_time).total_seconds()
        if since < TRADE_COOLDOWN_SECONDS:
            log_line(f"COOLDOWN: {int(since)}s since last trade (<{TRADE_COOLDOWN_SECONDS}s), HOLD")
            return False


    log_line(
        f"PRICE={price:.2f} | BTC_QTY={btc_qty:.8f} BTC_VALUE=${btc_value:.2f} CASH=${cash:.2f} "
        f"| LEVELS blue={blue:.2f} green={green:.2f} yellow={yellow:.2f}"
    )

    # ---- YOUR PSEUDOCODE (interpreted precisely) ----
    # if below green level:
    #   buy enough BTC so BTC value ~= starting_money * 0.2
    # if below blue level:
    #   buy BTC with all money left (target BTC value ~= starting_money)
    # if above yellow level:
    #   sell all BTC

    # 1) SELL condition
    if price > yellow and btc_qty > 0:
        log_line(f"ACTION: SELL ALL (price {price:.2f} > yellow {yellow:.2f})")
        submit_market_sell_qty(SYMBOL, btc_qty)
        log_line(f"ORDER: sell qty={btc_qty:.8f}")
        return True

    # 2) BLUE: go all-in up to STARTING_MONEY_USD
    if price < blue:
        target_btc_value = STARTING_MONEY_USD
        needed = max(0.0, target_btc_value - btc_value)
        buy_usd = min(needed, cash)

        if buy_usd >= MIN_TRADE_USD:
            log_line(f"ACTION: BUY ALL-IN (price {price:.2f} < blue {blue:.2f}), notional=${buy_usd:.2f}")
            submit_market_buy_notional(SYMBOL, buy_usd)
            log_line(f"ORDER: buy notional=${buy_usd:.2f}")
            return True
        else:
            log_line("ACTION: (blue) No buy; either already allocated or insufficient cash/min trade.")
        return

    # 3) GREEN: maintain ~20% allocation
    if price < green:
        target_btc_value = STARTING_MONEY_USD * 0.20
        needed = max(0.0, target_btc_value - btc_value)
        buy_usd = min(needed, cash)

        if buy_usd >= MIN_TRADE_USD:
            log_line(f"ACTION: BUY to 20% (price {price:.2f} < green {green:.2f}), notional=${buy_usd:.2f}")
            submit_market_buy_notional(SYMBOL, buy_usd)
            log_line(f"ORDER: buy notional=${buy_usd:.2f}")
            return True
        else:
            log_line("ACTION: (green) No buy; already at/above 20% target or insufficient cash/min trade.")
        return

    # Otherwise: no action
    log_line("ACTION: HOLD (no thresholds triggered)")
    return False


# ---------------- MAIN LOOP ----------------
def main() -> None:
    log_line("=== BOT START ===")
    log_line(f"STARTING_MONEY_USD=${STARTING_MONEY_USD:.2f} | SYMBOL={SYMBOL}")

    # Load saved state
    state = load_state()

    last_report_time = get_dt(state, "last_report_time") or now_utc()
    last_trade_time = get_dt(state, "last_trade_time")  # can be None
    last_bar_time = get_dt(state, "last_bar_time")      # can be None

    # Schedule next report relative to last_report_time
    next_report_ts = time.time() + REPORT_EVERY_SECONDS
    if last_report_time:
        # If we restarted, try to keep the 12-hour rhythm:
        elapsed = (now_utc() - last_report_time).total_seconds()
        remaining = REPORT_EVERY_SECONDS - elapsed
        next_report_ts = time.time() + max(30, remaining)  # minimum 30s

    while True:
        try:
            # ---- Get latest bar and ensure it's "new" ----
            end = now_utc()
            start = end - dt.timedelta(minutes=3)

            bars = alpaca.get_crypto_bars(
                SYMBOL,
                TimeFrame.Minute,
                start.isoformat(),
                end.isoformat(),
            ).df

            if bars is None or bars.empty:
                raise RuntimeError("No crypto bars returned")

            bar_time = bars.index[-1].to_pydatetime()
            if bar_time.tzinfo is None:
                bar_time = bar_time.replace(tzinfo=dt.timezone.utc)

            price = float(bars["close"].iloc[-1])

            # If we already processed this exact minute bar, skip
            if last_bar_time is not None and bar_time <= last_bar_time:
                log_line(f"SKIP: already processed bar @ {bar_time.isoformat()}")
            else:
                # Update bar time in state immediately
                last_bar_time = bar_time
                set_dt(state, "last_bar_time", last_bar_time)
                save_state(state)

                # ---- Run strategy with cooldown ----
                did_trade = strategy_step(price, last_trade_time)

                if did_trade:
                    last_trade_time = now_utc()
                    set_dt(state, "last_trade_time", last_trade_time)
                    save_state(state)

        except Exception as e:
            log_line(f"ERROR: {repr(e)}")

        # ---- Email report every 12 hours ----
        if time.time() >= next_report_ts:
            try:
                body = build_report(last_report_time)
                sendReport(body=body)
                log_line("EMAIL: sent 12-hour report")

                last_report_time = now_utc()
                set_dt(state, "last_report_time", last_report_time)
                save_state(state)

            except Exception as e:
                log_line(f"EMAIL ERROR: {repr(e)}")

            next_report_ts = time.time() + REPORT_EVERY_SECONDS

        time.sleep(PRICE_CHECK_SECONDS)



if __name__ == "__main__":
    main()
