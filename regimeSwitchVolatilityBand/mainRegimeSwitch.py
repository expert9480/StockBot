# volatility_band_bot.py
from __future__ import annotations

import os
import time
import pickle
import datetime as dt
from typing import Dict, Any, Optional, List

import numpy as np
import yfinance as yf
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

from gmail import sendReport

load_dotenv()

# ---------------- CONFIG ----------------
LOG_FILE = "vol_band_bot.log"
STATE_FILE = "vol_band_state.pkl"

WATCH_ETFS = ["SPY", "QQQ", "IWM"]

CHECK_SECONDS = 60
SAVE_EVERY_SECONDS = 15 * 60
REPORT_EVERY_SECONDS_OPEN = 4 * 60 * 60

# Volatility settings
VOL_LOOKBACK_RETURNS = 30     # number of bars
LOW_VOL = 0.010               # 1.0% daily vol
HIGH_VOL = 0.020              # 2.0% daily vol
HYSTERESIS = 0.002            # prevents flip-flop

MIN_MODE_SECONDS = 30 * 60    # must stay in a mode 30 min before switching

# Capital allocation per mode
MODE_TARGET_EXPOSURE = {
    "RISK_ON": 1.0,
    "NEUTRAL": 0.5,
    "RISK_OFF": 0.0,
}

MAX_TRADE_USD_PER_SYMBOL = 40_000
MIN_BUY_USD = 100.0
BUY_COOLDOWN_SECONDS = 30 * 60

# Alpaca
API_KEY = os.getenv("alpacaKey")
API_SECRET = os.getenv("alpacaSecret")
BASE_URL = os.getenv("alpacaBaseURL", "https://paper-api.alpaca.markets")
alpaca = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def log(level: str, msg: str, ctx: Optional[Dict[str, Any]] = None):
    ts = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {level:<5} {msg}"
    if ctx:
        line += " | " + " ".join(f"{k}={v}" for k, v in ctx.items())
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def default_state() -> Dict[str, Any]:
    return {
        "mode": "NEUTRAL",
        "mode_enter_time": now_utc().isoformat(),
        "last_buy_time": None,
        "last_report_time": None,
        "peak_equity": None,
    }


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        log("INFO", "No state file, starting fresh")
        return default_state()
    try:
        with open(STATE_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        log("ERROR", "Failed to load state", {"err": e})
        return default_state()


def save_state(state: Dict[str, Any]):
    with open(STATE_FILE, "wb") as f:
        pickle.dump(state, f)

def market_is_open() -> bool:
    return alpaca.get_clock().is_open


def yf_realized_vol(symbol: str) -> float:
    hist = yf.Ticker(symbol).history(period="60d", interval="1d")
    rets = np.log(hist["Close"]).diff().dropna()
    return float(rets[-VOL_LOOKBACK_RETURNS:].std())


def account_cash() -> float:
    return float(alpaca.get_account().cash)

def compute_vol_regime(vol: float, current_mode: str) -> str:
    if current_mode == "RISK_OFF":
        if vol < HIGH_VOL - HYSTERESIS:
            return "NEUTRAL"
        return "RISK_OFF"

    if current_mode == "RISK_ON":
        if vol > LOW_VOL + HYSTERESIS:
            return "NEUTRAL"
        return "RISK_ON"

    # NEUTRAL
    if vol <= LOW_VOL:
        return "RISK_ON"
    if vol >= HIGH_VOL:
        return "RISK_OFF"
    return "NEUTRAL"

def rebalance(target_exposure: float):
    cash = account_cash()
    total_to_invest = cash * target_exposure
    per_symbol = min(
        total_to_invest / len(WATCH_ETFS),
        MAX_TRADE_USD_PER_SYMBOL,
    )

    for sym in WATCH_ETFS:
        if per_symbol >= MIN_BUY_USD:
            alpaca.submit_order(
                symbol=sym,
                notional=round(per_symbol, 2),
                side="buy",
                type="market",
                time_in_force="day",
            )
            log("ACTION", "BUY", {"symbol": sym, "usd": round(per_symbol, 2)})

def maybe_send_report(state: Dict[str, Any], vol: float):
    now = now_utc()
    last = state.get("last_report_time")
    if last:
        last = dt.datetime.fromisoformat(last)
        if (now - last).total_seconds() < REPORT_EVERY_SECONDS_OPEN:
            return

    body = (
        f"Mode: {state['mode']}\n"
        f"Realized Vol: {vol:.4f}\n"
        f"Target Exposure: {MODE_TARGET_EXPOSURE[state['mode']]*100:.0f}%\n"
        f"Cash: ${account_cash():.2f}"
    )
    sendReport(body, subject_prefix="Volatility Regime")
    state["last_report_time"] = now.isoformat()
    log("EMAIL", "Report sent")

def main():
    log("INFO", "Volatility Band Bot starting")
    state = load_state()

    while True:
        try:
            if not market_is_open():
                log("THINK", "Market closed")
                time.sleep(CHECK_SECONDS)
                continue

            vols = [yf_realized_vol(sym) for sym in WATCH_ETFS]
            portfolio_vol = float(np.mean(vols))

            old_mode = state["mode"]
            new_mode = compute_vol_regime(portfolio_vol, old_mode)

            time_in_mode = (
                now_utc() - dt.datetime.fromisoformat(state["mode_enter_time"])
            ).total_seconds()

            if new_mode != old_mode and time_in_mode >= MIN_MODE_SECONDS:
                log("TRIG", "Mode change", {"from": old_mode, "to": new_mode})
                state["mode"] = new_mode
                state["mode_enter_time"] = now_utc().isoformat()
                rebalance(MODE_TARGET_EXPOSURE[new_mode])

            maybe_send_report(state, portfolio_vol)
            save_state(state)

        except Exception as e:
            log("ERROR", "Loop failure", {"err": e})

        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    main()
