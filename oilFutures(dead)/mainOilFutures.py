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
from alpaca_trade_api.rest import TimeFrame
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

# console = logging.StreamHandler()
# console.setFormatter(fmt)
# logger.addHandler(console)

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

INTERVAL = os.getenv("INTERVAL", "5m")
ALPACA_TF = os.getenv("ALPACA_TF", "5Min")
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
BOOTSTRAP_DOLLARS = float(os.getenv("BOOTSTRAP_DOLLARS", "2000"))

SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "3"))
TRADES_CSV = os.getenv("TRADES_CSV", "trades.csv")

MIN_ROWS_FOR_MODEL = 300  # after alignment

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
    return "USO"

TRADE_SYMBOL = pick_trade_symbol()

def ensure_trades_csv():
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "ts_utc","symbol","side","qty","submitted_price",
                "effective_price","order_id","zscore","kelly",
                "signal","target_dollars"
            ])

def append_trade_csv(row: dict):
    ensure_trades_csv()
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([row.get(k) for k in [
            "ts_utc","symbol","side","qty","submitted_price",
            "effective_price","order_id","zscore","kelly",
            "signal","target_dollars"
        ]])

# =============================
# DATA LOADING
# =============================
def load_proxy_data(period="5d", interval="5m"):
    tickers = ["BZ=F", "CL=F"]
    raw = yf.download(
        tickers, interval=interval, period=period,
        group_by="column", auto_adjust=False, progress=False
    )
    close = raw["Close"]
    df = close.rename(columns={"BZ=F": "spot", "CL=F": "futures"}).dropna()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    return df

def _parse_timeframe(tf_str: str) -> TimeFrame:
    s = tf_str.lower()
    digits = "".join(filter(str.isdigit, s))
    val = int(digits) if digits else 5
    if "min" in s or "m" in s:
        return TimeFrame(val, TimeFrame.Minute)
    if "hour" in s or "h" in s:
        return TimeFrame(val, TimeFrame.Hour)
    return TimeFrame(val, TimeFrame.Minute)

def load_alpaca_bars(symbol: str, lookback_days: int, tf: str):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    tf_obj = _parse_timeframe(tf)

    bars = alpaca.get_bars(
        symbol, tf_obj,
        start=start.isoformat(), end=end.isoformat(),
        adjustment="raw", feed="iex"
    ).df

    df = bars[["close"]].rename(columns={"close": "trade_close"})
    df.index = pd.to_datetime(df.index, utc=True)
    return df

# =============================
# RESAMPLE + ALIGN
# =============================
def align_and_resample(proxy_df, trade_df, interval="5min"):
    start = max(proxy_df.index.min(), trade_df.index.min())
    end = min(proxy_df.index.max(), trade_df.index.max())

    idx = pd.date_range(start=start, end=end, freq=interval, tz="UTC")

    proxy_resampled = proxy_df.reindex(idx).ffill()
    trade_resampled = trade_df.reindex(idx).ffill()

    df = proxy_resampled.join(trade_resampled, how="inner")
    return df.dropna()

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
    df["ema_fast"] = df["trade_close"].ewm(span=EMA_FAST).mean()
    df["ema_slow"] = df["trade_close"].ewm(span=EMA_SLOW).mean()
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
    slip = px * (1 + SLIPPAGE_BPS/10000 * (1 if side=="buy" else -1))

    order = alpaca.submit_order(
        symbol=symbol, qty=int(abs(qty_delta)),
        side=side, type="market", time_in_force="day"
    )

    meta.update({
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "side": side,
        "qty": abs(int(qty_delta)),
        "order_id": order.id,
        "effective_price": slip
    })
    append_trade_csv(meta)
    logger.info(f"Submitted {side} {qty_delta} {symbol} at mkt (ref={px}, eff≈{slip})")
    return order

def get_current_position_qty(symbol):
    try:
        for p in alpaca.list_positions():
            if p.symbol == symbol:
                return float(p.qty)
    except:
        pass
    return 0.0

# =============================
# BOOTSTRAP
# =============================
def bootstrap_initial_position(symbol, buying_power):
    qty = get_current_position_qty(symbol)
    if qty != 0:
        logger.info(f"Startup check: existing position detected ({qty} shares). No bootstrap.")
        return

    target = buying_power * 0.05
    last_px = alpaca.get_latest_trade(symbol).price
    shares = int(target / last_px)

    if shares <= 0:
        logger.warning("Bootstrap qty <= 0. Skipping.")
        return

    logger.info(f"Bootstrap: buying {shares} {symbol} (~5% of BP).")
    alpaca.submit_order(symbol=symbol, qty=shares, side="buy", type="market", time_in_force="day")

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

    bootstrap_initial_position(TRADE_SYMBOL, float(account.buying_power))

    last_trade_ts = 0

    while True:
        try:
            clock = alpaca.get_clock()
            if not clock.is_open:
                time.sleep(60)
                continue

            account = alpaca.get_account()
            curr_equity = float(account.equity)

            if (start_equity - curr_equity) / start_equity >= DAILY_LOSS_LIMIT_PCT:
                logger.error("Daily loss limit hit.")
                break

            proxy = load_proxy_data(period=f"{LOOKBACK_DAYS}d", interval=INTERVAL)
            trade_bars = load_alpaca_bars(TRADE_SYMBOL, LOOKBACK_DAYS, ALPACA_TF)

            df = align_and_resample(proxy, trade_bars, interval="5min")

            logger.info(f"Aligned rows={len(df)}")

            if len(df) < MIN_ROWS_FOR_MODEL:
                logger.warning(f"Not enough rows ({len(df)}) < {MIN_ROWS_FOR_MODEL}. Skipping.")
                time.sleep(SLEEP_SECONDS)
                continue

            df = rolling_regression(df, WINDOW)
            df = compute_kelly(df, WINDOW)
            df = add_trend_filter(df)
            df = generate_spread_signals(df, Z_ENTRY)
            df = map_signal_to_etf(df, TRADE_SYMBOL)
            df = size_positions(df, curr_equity)

            if df.empty:
                logger.warning("DF empty after pipeline.")
                time.sleep(SLEEP_SECONDS)
                continue

            latest = df.iloc[-1]
            px = latest["trade_close"]
            target_sh = int(latest["target_dollars"] / px)
            curr_sh = get_current_position_qty(TRADE_SYMBOL)
            delta = target_sh - curr_sh

            logger.info(
                f"z={latest['zscore']:.2f}, sig={latest['signal']}, "
                f"kelly={latest['kelly']:.3f}, tgt_sh={target_sh}, curr_sh={curr_sh}, delta={delta}"
            )

            now = time.time()
            if abs(delta) >= MIN_ORDER_SHARES and (now - last_trade_ts) > COOLDOWN_SECONDS:
                submit_order(
                    TRADE_SYMBOL, delta,
                    {
                        "submitted_price": px,
                        "zscore": float(latest["zscore"]),
                        "kelly": float(latest["kelly"]),
                        "signal": int(latest["signal"]),
                        "target_dollars": float(latest["target_dollars"])
                    }
                )
                last_trade_ts = now

        except Exception as e:
            logger.exception(f"Loop error: {e}")

        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    live_trading_loop()
