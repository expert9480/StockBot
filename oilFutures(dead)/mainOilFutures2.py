import os
import time
import csv
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError
from dotenv import load_dotenv

load_dotenv()

# =============================
# LOGGING
# =============================
LOG_PATH = os.getenv("BOT_LOG_PATH", "oil_bot.log")
logger = logging.getLogger("oil_bot")
logger.setLevel(logging.INFO)

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=5)
file_handler.setFormatter(fmt)
logger.addHandler(file_handler)

# =============================
# CONFIG
# =============================
CAPITAL = float(os.getenv("CAPITAL", "100000"))
PREFERRED_SYMBOLS = [
    s.strip()
    for s in os.getenv("PREFERRED_SYMBOLS", "BNO,USO").split(",")
    if s.strip()
]

API_KEY = os.getenv("alpacaKey")
API_SECRET = os.getenv("alpacaSecret")
BASE_URL = os.getenv("alpacaBaseURL", "https://paper-api.alpaca.markets")
alpaca = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

INTERVAL = os.getenv("INTERVAL", "5m")  # yfinance interval
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "20"))
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "60"))

WINDOW = int(os.getenv("WINDOW", "96"))
Z_ENTRY = float(os.getenv("Z_ENTRY", "1.2"))
Z_EXIT = float(os.getenv("Z_EXIT", "0.4"))
MAX_KELLY = float(os.getenv("MAX_KELLY", "0.35"))

EMA_FAST = int(os.getenv("EMA_FAST", "12"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "48"))

MAX_LEVER = float(os.getenv("MAX_LEVER", "1.25"))
MAX_DOLLAR_POSITION = float(os.getenv("MAX_DOLLAR_POSITION", "60000"))
MIN_ORDER_SHARES = int(os.getenv("MIN_ORDER_SHARES", "1"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "120"))

DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.03"))
KILL_SWITCH = os.getenv("KILL_SWITCH", "0") == "1"

BOOTSTRAP_IF_FLAT = os.getenv("BOOTSTRAP_IF_FLAT", "1") == "1"
BOOTSTRAP_PCT_BP = float(os.getenv("BOOTSTRAP_PCT_BP", "0.05"))  # 5% of buying power

SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "3"))
TRADES_CSV = os.getenv("TRADES_CSV", "trades.csv")

MIN_ROWS_FOR_MODEL = max(3 * WINDOW, 300)

# =============================
# UTIL
# =============================
def pick_trade_symbol():
    for sym in PREFERRED_SYMBOLS:
        try:
            a = alpaca.get_asset(sym)
            if getattr(a, "tradable", False):
                logger.info(f"Selected trade symbol: {sym}")
                return sym
        except Exception as e:
            logger.warning(f"Could not load Alpaca asset {sym}: {e}")
    logger.warning("Falling back to USO as trade symbol.")
    return "USO"

TRADE_SYMBOL = pick_trade_symbol()

def ensure_trades_csv():
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "ts_utc",
                    "symbol",
                    "side",
                    "qty",
                    "submitted_price",
                    "effective_price",
                    "order_id",
                    "zscore",
                    "kelly",
                    "signal",
                    "target_dollars",
                ]
            )

def append_trade_csv(row: dict):
    ensure_trades_csv()
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                row.get(k)
                for k in [
                    "ts_utc",
                    "symbol",
                    "side",
                    "qty",
                    "submitted_price",
                    "effective_price",
                    "order_id",
                    "zscore",
                    "kelly",
                    "signal",
                    "target_dollars",
                ]
            ]
        )

# =============================
# DATA LOADING (YFINANCE ONLY)
# =============================
def load_yf_data(trade_symbol: str, lookback_days: int, interval: str = "5m"):
    tickers = ["BZ=F", "CL=F", trade_symbol]
    period = f"{lookback_days}d"

    raw = yf.download(
        tickers,
        interval=interval,
        period=period,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )

    def extract_close(ticker):
        if isinstance(raw.columns, pd.MultiIndex):
            return raw[ticker]["Close"]
        else:
            return raw["Close"]

    spot = extract_close("BZ=F").rename("spot")
    fut = extract_close("CL=F").rename("futures")
    etf = extract_close(trade_symbol).rename("trade_close")

    df = pd.concat([spot, fut, etf], axis=1).dropna()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    return df

# =============================
# MODEL
# =============================
def rolling_regression(df, window):
    if len(df) < window + 5:
        return pd.DataFrame(index=df.index)

    y = df["spot"]
    x = df["futures"]

    mean_x = x.rolling(window).mean()
    mean_y = y.rolling(window).mean()
    cov_xy = (x * y).rolling(window).mean() - mean_x * mean_y
    var_x = (x**2).rolling(window).mean() - mean_x**2

    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x
    fitted = alpha + beta * x
    residual = y - fitted

    zscore = (residual - residual.rolling(window).mean()) / residual.rolling(window).std()

    out = df.copy()
    out["residual"] = residual
    out["zscore"] = zscore
    return out.dropna()

def compute_kelly(df, window):
    if len(df) < window + 5:
        df["kelly"] = 0.0
        return df

    ret = df["trade_close"].pct_change()
    mu = ret.rolling(window).mean()
    sigma2 = ret.rolling(window).var()

    k = (mu / sigma2).replace([np.inf, -np.inf], np.nan)
    df["kelly"] = k.clip(0, MAX_KELLY).fillna(0.0)
    return df

def add_trend_filter(df):
    df["ema_fast"] = df["trade_close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["trade_close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["trend_up"] = df["ema_fast"] > df["ema_slow"]
    return df

def generate_spread_signals(df, z_entry):
    sig = pd.Series(0, index=df.index)
    sig[df["zscore"] > z_entry] = -1
    sig[df["zscore"] < -z_entry] = 1

    sig[(df["trend_up"] == True) & (sig < 0)] = 0
    sig[(df["trend_up"] == False) & (sig > 0)] = 0

    df["spread_signal"] = sig
    return df

def map_signal_to_etf(df, symbol):
    if "BNO" in symbol.upper():
        df["signal"] = df["spread_signal"]
    else:
        df["signal"] = -df["spread_signal"]
    return df

def size_positions(df, capital):
    vol = df["trade_close"].pct_change().rolling(96).std().replace(0, np.nan)
    raw = (0.008 * capital / vol) * df["signal"]
    raw = raw * (0.25 + 0.75 * df["kelly"])

    cap = min(MAX_DOLLAR_POSITION, capital * MAX_LEVER)
    df["target_dollars"] = raw.clip(-cap, cap)
    return df

# =============================
# EXECUTION
# =============================
def submit_order(symbol, qty_delta, meta):
    if abs(qty_delta) < MIN_ORDER_SHARES:
        return None

    side = "buy" if qty_delta > 0 else "sell"
    px = meta["submitted_price"]
    slip = px * (1 + SLIPPAGE_BPS / 10000 * (1 if side == "buy" else -1))

    try:
        order = alpaca.submit_order(
            symbol=symbol,
            qty=int(abs(qty_delta)),
            side=side,
            type="market",
            time_in_force="day",
        )
    except APIError as e:
        msg = str(e).lower()
        if "wash trade" in msg:
            logger.warning(f"Wash trade blocked by Alpaca: {e}")
            return None
        logger.exception(f"Order error: {e}")
        return None

    meta.update(
        {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "side": side,
            "qty": abs(int(qty_delta)),
            "order_id": order.id,
            "effective_price": slip,
        }
    )
    append_trade_csv(meta)
    logger.info(
        f"Submitted {side} {abs(int(qty_delta))} {symbol} at mkt "
        f"(ref={px}, eff≈{slip})"
    )
    return order

def get_current_position_qty(symbol):
    try:
        for p in alpaca.list_positions():
            if p.symbol == symbol:
                return float(p.qty)
    except Exception as e:
        logger.warning(f"Error fetching positions: {e}")
    return 0.0

# =============================
# BOOTSTRAP
# =============================
def should_bootstrap(trade_symbol: str) -> bool:
    try:
        df = load_yf_data(trade_symbol, LOOKBACK_DAYS, INTERVAL)
        if len(df) < MIN_ROWS_FOR_MODEL:
            logger.info("Bootstrap check: not enough rows for model.")
            return False
        df = rolling_regression(df, WINDOW)
        if df.empty:
            logger.info("Bootstrap check: empty after regression.")
            return False
        latest = df.iloc[-1]
        z = float(latest["zscore"])
        logger.info(f"Bootstrap check: latest z={z:.2f}")
        return abs(z) < 0.5
    except Exception as e:
        logger.warning(f"Bootstrap check failed: {e}")
        return False

def bootstrap_initial_position(symbol, buying_power) -> bool:
    if not BOOTSTRAP_IF_FLAT:
        logger.info("Bootstrap disabled by config.")
        return False

    qty = get_current_position_qty(symbol)
    if qty != 0:
        logger.info(f"Startup: existing position detected ({qty} shares). No bootstrap.")
        return False

    target_notional = buying_power * BOOTSTRAP_PCT_BP
    hist = yf.Ticker(symbol).history(period="1d", interval="1m")
    if hist.empty:
        logger.warning("Bootstrap: no recent price data from yfinance.")
        return False
    last_px = hist["Close"].iloc[-1]
    shares = int(target_notional / last_px)

    if shares <= 0:
        logger.warning("Bootstrap qty <= 0. Skipping bootstrap.")
        return False

    logger.info(f"Bootstrap: buying {shares} {symbol} (~{BOOTSTRAP_PCT_BP*100:.1f}% of BP).")
    try:
        alpaca.submit_order(
            symbol=symbol,
            qty=shares,
            side="buy",
            type="market",
            time_in_force="day",
        )
        return True
    except Exception as e:
        logger.exception(f"Bootstrap order error: {e}")
        return False

# =============================
# LIVE LOOP
# =============================
def live_trading_loop():
    if KILL_SWITCH:
        logger.error("KILL_SWITCH enabled. Exiting.")
        return

    logger.info(f"Bot Active. Trading {TRADE_SYMBOL}")

    account = alpaca.get_account()
    start_equity = float(account.equity)

    did_bootstrap = False
    if BOOTSTRAP_IF_FLAT and should_bootstrap(TRADE_SYMBOL):
        did_bootstrap = bootstrap_initial_position(TRADE_SYMBOL, float(account.buying_power))
    else:
        logger.info("Bootstrap skipped (disabled, not flat, or signal not neutral).")

    last_trade_ts = time.time() if did_bootstrap else 0

    while True:
        try:
            clock = alpaca.get_clock()
            if not clock.is_open:
                time.sleep(60)
                continue

            account = alpaca.get_account()
            curr_equity = float(account.equity)

            dd = (start_equity - curr_equity) / start_equity
            if dd >= DAILY_LOSS_LIMIT_PCT:
                logger.error(
                    f"Daily loss limit hit. Drawdown={dd:.2%} >= {DAILY_LOSS_LIMIT_PCT:.2%}. Stopping."
                )
                break

            df = load_yf_data(TRADE_SYMBOL, LOOKBACK_DAYS, INTERVAL)
            logger.info(f"Loaded YF rows={len(df)}")

            if len(df) < MIN_ROWS_FOR_MODEL:
                logger.warning(
                    f"Not enough rows for model: have {len(df)}, need {MIN_ROWS_FOR_MODEL}. Skipping."
                )
                time.sleep(SLEEP_SECONDS)
                continue

            df = rolling_regression(df, WINDOW)
            if df.empty:
                logger.warning("Empty after rolling_regression. Skipping.")
                time.sleep(SLEEP_SECONDS)
                continue

            df = compute_kelly(df, WINDOW)
            df = add_trend_filter(df)
            df = generate_spread_signals(df, Z_ENTRY)
            df = map_signal_to_etf(df, TRADE_SYMBOL)
            df = size_positions(df, curr_equity)

            if df.empty:
                logger.warning("DF empty after full pipeline. Skipping.")
                time.sleep(SLEEP_SECONDS)
                continue

            latest = df.iloc[-1]
            px = float(latest["trade_close"])
            if px <= 0:
                logger.warning("Latest trade_close <= 0. Skipping.")
                time.sleep(SLEEP_SECONDS)
                continue

            target_sh = int(latest["target_dollars"] / px)
            curr_sh = get_current_position_qty(TRADE_SYMBOL)
            delta = target_sh - curr_sh

            logger.info(
                f"z={latest['zscore']:.2f}, sig={latest['signal']}, "
                f"kelly={latest['kelly']:.3f}, tgt_sh={target_sh}, "
                f"curr_sh={curr_sh}, delta={delta}"
            )

            now = time.time()
            if abs(delta) >= MIN_ORDER_SHARES and (now - last_trade_ts) > COOLDOWN_SECONDS:
                submit_order(
                    TRADE_SYMBOL,
                    delta,
                    {
                        "submitted_price": px,
                        "zscore": float(latest["zscore"]),
                        "kelly": float(latest["kelly"]),
                        "signal": int(latest["signal"]),
                        "target_dollars": float(latest["target_dollars"]),
                    },
                )
                last_trade_ts = now
            else:
                if abs(delta) < MIN_ORDER_SHARES:
                    logger.info("Delta below MIN_ORDER_SHARES; no trade.")
                else:
                    logger.info("Cooldown active; skipping trade.")

        except Exception as e:
            logger.exception(f"Loop error: {e}")

        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    live_trading_loop()
