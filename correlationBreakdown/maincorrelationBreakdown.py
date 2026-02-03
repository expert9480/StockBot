# correlation_breakdown_bot.py
from __future__ import annotations

import os
import time
import pickle
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import yfinance as yf
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

from gmail import sendReport

# ---------------- CONFIG ----------------
load_dotenv()

LOG_FILE = "corr_breakdown_bot.log"
STATE_FILE = "corr_breakdown_state.pkl"

# Risk ETFs to hold when conditions are healthy
WATCH_ETFS = ["SPY", "QQQ", "IWM"]

# Correlation pair to monitor (classic: SPY vs QQQ)
PAIR_A = "SPY"
PAIR_B = "QQQ"

# Optional defensive holding when diverged (set to None to hold cash)
DEFENSIVE_ETF: Optional[str] = "BIL"

# Loop timing
CHECK_SECONDS = 60
SAVE_EVERY_SECONDS = 15 * 60
REPORT_EVERY_SECONDS_OPEN = 4 * 60 * 60

# Data settings: 5-minute bars are a good compromise
YF_INTERVAL = "5m"
YF_PERIOD = "5d"          # enough bars for rolling windows intraday

# Rolling windows
CORR_WINDOW_BARS = 78      # ~1 trading day of 5m bars (78 = 390/5)
SPREAD_WINDOW_BARS = 78

# Triggers / thresholds
CORR_LOW = 0.30            # breakdown if corr falls below this (with confirm)
CORR_RECOVER = 0.60        # recovery if corr rises above this (with confirm)

Z_ENTRY = 2.0              # divergence entry threshold
Z_EXIT = 0.7               # divergence exit threshold

CONFIRM_BARS = 3           # require N consecutive checks in condition

# Trading controls
STARTING_MONEY_USD = 100_000.00
MAX_TRADE_USD_PER_SYMBOL = 50_000.00
MIN_BUY_USD = 50.0
BUY_COOLDOWN_SECONDS = 30 * 60

# Alpaca env vars
API_KEY = os.getenv("alpacaKey")
API_SECRET = os.getenv("alpacaSecret")
BASE_URL = os.getenv("alpacaBaseURL", "https://paper-api.alpaca.markets")
alpaca = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")


# ---------------- LOGGING ----------------
def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _fmt_kv(ctx: Optional[Dict[str, Any]]) -> str:
    if not ctx:
        return ""
    parts: List[str] = []
    for k, v in ctx.items():
        try:
            if isinstance(v, float):
                parts.append(f"{k}={v:.6f}")
            else:
                parts.append(f"{k}={v}")
        except Exception:
            parts.append(f"{k}={repr(v)}")
    return " | " + " ".join(parts)


def log_event(level: str, msg: str, ctx: Optional[Dict[str, Any]] = None) -> None:
    ts = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {level:<5} {msg}{_fmt_kv(ctx)}"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------- STATE ----------------
def default_state() -> Dict[str, Any]:
    return {
        "meta": {
            "mode": "NORMAL",              # NORMAL, BREAKDOWN_WATCH, DIVERGED, RECOVERY
            "mode_enter_time": now_utc().isoformat(),
            "last_report_time": None,
            "last_save_time": None,
            "last_check_time": None,
            "last_buy_time": None,

            # confirmation counters
            "breakdown_count": 0,
            "recovery_count": 0,

            # last computed stats (for reporting)
            "last_corr": None,
            "last_spread_z": None,
            "last_beta": None,

            "loop_count": 0,
        }
    }


def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, STATE_FILE)
    state["meta"]["last_save_time"] = now_utc().isoformat()


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        log_event("INFO", "STATE: no existing state file -> default_state()")
        return default_state()
    try:
        with open(STATE_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        log_event("ERROR", "STATE: failed to load -> default_state()", {"err": repr(e)})
        return default_state()


# ---------------- ALPACA HELPERS ----------------
def market_is_open() -> bool:
    return bool(alpaca.get_clock().is_open)


def account_cash_and_bp() -> Tuple[float, float]:
    acct = alpaca.get_account()
    cash = float(getattr(acct, "cash", 0.0))
    bp = float(getattr(acct, "buying_power", 0.0))
    return cash, bp


def spendable_usd() -> float:
    """
    Conservative: if buying_power > cash, use cash (avoid leverage).
    """
    cash, bp = account_cash_and_bp()
    usable = bp if bp <= cash else cash
    return max(0.0, usable - 5.00)


def safe_position_qty(symbol: str) -> float:
    try:
        pos = alpaca.get_position(symbol)
        return float(pos.qty)
    except Exception:
        return 0.0


def positions_qty(symbols: List[str]) -> Dict[str, float]:
    return {s: safe_position_qty(s) for s in symbols}


def is_flat(symbols: List[str]) -> bool:
    q = positions_qty(symbols)
    return all(qty <= 0 for qty in q.values())


def list_open_orders_for(symbols: set[str]) -> List[Any]:
    try:
        orders = alpaca.list_orders(status="open", limit=200)
        return [o for o in orders if getattr(o, "symbol", "") in symbols]
    except Exception as e:
        log_event("WARN", "ALPACA: list_orders failed", {"err": repr(e)})
        return []


def submit_sell_all(symbol: str) -> bool:
    qty = safe_position_qty(symbol)
    if qty <= 0:
        log_event("THINK", "SELL skipped (no position)", {"symbol": symbol, "qty": qty})
        return False
    try:
        o = alpaca.submit_order(
            symbol=symbol,
            qty=str(qty),
            side="sell",
            type="market",
            time_in_force="day",
        )
        log_event("ACTION", "SELL all", {"symbol": symbol, "qty": qty, "order_id": getattr(o, "id", "")})
        return True
    except Exception as e:
        log_event("ERROR", "SELL failed", {"symbol": symbol, "qty": qty, "err": repr(e)})
        return False


def submit_buy_notional(symbol: str, usd: float) -> bool:
    if usd <= 0:
        log_event("THINK", "BUY skipped (usd<=0)", {"symbol": symbol, "usd": usd})
        return False
    try:
        o = alpaca.submit_order(
            symbol=symbol,
            notional=round(usd, 2),
            side="buy",
            type="market",
            time_in_force="day",
        )
        log_event("ACTION", "BUY notional", {"symbol": symbol, "usd": float(usd), "order_id": getattr(o, "id", "")})
        return True
    except Exception as e:
        log_event("ERROR", "BUY failed", {"symbol": symbol, "usd": float(usd), "err": repr(e)})
        return False


def buy_cooldown_ok(state: Dict[str, Any]) -> bool:
    last_iso = state["meta"].get("last_buy_time")
    if not last_iso:
        return True
    try:
        last = dt.datetime.fromisoformat(last_iso)
    except Exception:
        return True
    return (now_utc() - last).total_seconds() >= BUY_COOLDOWN_SECONDS


def mark_buy_time(state: Dict[str, Any]) -> None:
    state["meta"]["last_buy_time"] = now_utc().isoformat()


def compute_per_symbol_buy_usd() -> Dict[str, float]:
    usable = min(spendable_usd(), STARTING_MONEY_USD)
    if usable <= 0:
        return {sym: 0.0 for sym in WATCH_ETFS}

    per = usable / max(1, len(WATCH_ETFS))
    per = min(per, MAX_TRADE_USD_PER_SYMBOL)
    return {sym: per for sym in WATCH_ETFS}


# ---------------- MARKET DATA (YFINANCE) ----------------
def yf_pair_history(a: str, b: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetch aligned close arrays for two symbols.
    Uses 5m bars by default; aligns on timestamps.
    """
    tickers = yf.Tickers(f"{a} {b}")
    ha = tickers.tickers[a].history(period=YF_PERIOD, interval=YF_INTERVAL)
    hb = tickers.tickers[b].history(period=YF_PERIOD, interval=YF_INTERVAL)

    if ha is None or ha.empty or hb is None or hb.empty:
        raise RuntimeError("Empty yfinance history for pair")

    # Align by index intersection
    idx = ha.index.intersection(hb.index)
    ha = ha.loc[idx]
    hb = hb.loc[idx]

    ca = ha["Close"].astype(float).to_numpy()
    cb = hb["Close"].astype(float).to_numpy()

    if len(ca) < max(CORR_WINDOW_BARS, SPREAD_WINDOW_BARS) + 5:
        raise RuntimeError(f"Not enough bars for windows (have {len(ca)})")

    return ca, cb


def rolling_corr_and_spread_z(close_a: np.ndarray, close_b: np.ndarray) -> Tuple[float, float, float]:
    """
    Returns:
      corr: rolling correlation of log returns
      spread_z: z-score of spread = logA - beta*logB
      beta: regression slope of logA ~ beta*logB over spread window
    """
    # log returns
    la = np.log(close_a)
    lb = np.log(close_b)
    ra = np.diff(la)
    rb = np.diff(lb)

    # rolling correlation window on returns
    ra_w = ra[-CORR_WINDOW_BARS:]
    rb_w = rb[-CORR_WINDOW_BARS:]
    corr = float(np.corrcoef(ra_w, rb_w)[0, 1])

    # beta via least squares on log prices over spread window
    la_w = la[-SPREAD_WINDOW_BARS:]
    lb_w = lb[-SPREAD_WINDOW_BARS:]
    # Solve la = beta*lb + c  (polyfit gives slope, intercept)
    beta, intercept = np.polyfit(lb_w, la_w, 1)
    beta = float(beta)

    spread = la_w - (beta * lb_w + intercept)
    mu = float(np.mean(spread))
    sd = float(np.std(spread, ddof=1)) if len(spread) > 2 else float(np.std(spread))
    if sd <= 1e-12:
        spread_z = 0.0
    else:
        spread_z = float((spread[-1] - mu) / sd)

    return corr, spread_z, beta


# ---------------- MODE / STRATEGY ----------------
def set_mode(state: Dict[str, Any], new_mode: str, reason: str, stats: Dict[str, Any]) -> None:
    old = state["meta"].get("mode")
    if new_mode == old:
        return
    state["meta"]["mode"] = new_mode
    state["meta"]["mode_enter_time"] = now_utc().isoformat()
    # reset counters when changing modes
    state["meta"]["breakdown_count"] = 0
    state["meta"]["recovery_count"] = 0
    log_event("TRIG", "MODE change", {"from": old, "to": new_mode, "reason": reason, **stats})

    # email on mode change
    body = (
        f"Correlation Breakdown Bot\n"
        f"Mode change: {old} -> {new_mode}\n"
        f"Reason: {reason}\n\n"
        f"PAIR: {PAIR_A} vs {PAIR_B}\n"
        f"corr={stats.get('corr'):.4f} | spread_z={stats.get('spread_z'):+.3f} | beta={stats.get('beta'):.4f}\n"
        f"CORR_LOW={CORR_LOW}  CORR_RECOVER={CORR_RECOVER}\n"
        f"Z_ENTRY={Z_ENTRY}    Z_EXIT={Z_EXIT}\n"
    )
    try:
        sendReport(body=body, subject_prefix="Corr Breakdown MODE")
        log_event("EMAIL", "mode-change email sent", {"to_mode": new_mode})
    except Exception as e:
        log_event("ERROR", "EMAIL mode-change failed", {"err": repr(e)})


def go_defensive() -> None:
    """
    Exit risk assets and optionally buy defensive ETF.
    """
    any_sold = False
    for sym in WATCH_ETFS:
        any_sold = submit_sell_all(sym) or any_sold

    if DEFENSIVE_ETF:
        # Only buy defensive if we have spendable cash
        alloc = min(spendable_usd(), STARTING_MONEY_USD, MAX_TRADE_USD_PER_SYMBOL)
        if alloc >= MIN_BUY_USD:
            submit_buy_notional(DEFENSIVE_ETF, alloc)
        else:
            log_event("THINK", "DEFENSIVE buy skipped (too small)", {"usd": alloc})

    log_event("INFO", "DEFENSIVE action complete", {"any_sold": any_sold, "defensive": DEFENSIVE_ETF})


def reinvest_if_needed(state: Dict[str, Any], stats: Dict[str, Any]) -> None:
    """
    If we're not diverged, and we're flat (in risk ETFs), reinvest.
    """
    if not buy_cooldown_ok(state):
        log_event("THINK", "No reinvest (cooldown)", {"cooldown_sec": BUY_COOLDOWN_SECONDS})
        return

    # Consider "flat" relative to WATCH_ETFS; defensive ETF doesn't block reinvest automatically
    flat_risk = is_flat(WATCH_ETFS)
    if not flat_risk:
        log_event("THINK", "No reinvest (already holding risk ETFs)")
        return

    # If holding defensive, you may want to sell it before reinvesting (clean behavior)
    if DEFENSIVE_ETF:
        submit_sell_all(DEFENSIVE_ETF)

    alloc = compute_per_symbol_buy_usd()
    log_event("TRIG", "REINVEST", {"alloc": {k: round(v, 2) for k, v in alloc.items()}, **stats})

    placed_any = False
    for sym in WATCH_ETFS:
        usd = float(alloc.get(sym, 0.0))
        if usd >= MIN_BUY_USD:
            placed_any = submit_buy_notional(sym, usd) or placed_any
        else:
            log_event("THINK", "BUY skipped (too small)", {"symbol": sym, "usd": usd, "min": MIN_BUY_USD})

    if placed_any:
        mark_buy_time(state)
    log_event("INFO", "REINVEST step complete", {"placed_any": placed_any})


# ---------------- REPORTING ----------------
def format_report(state: Dict[str, Any]) -> str:
    cash, bp = account_cash_and_bp()
    corr = state["meta"].get("last_corr")
    spread_z = state["meta"].get("last_spread_z")
    beta = state["meta"].get("last_beta")

    lines = []
    lines.append("Correlation Breakdown Bot Report")
    lines.append(f"Mode: {state['meta'].get('mode')}")
    lines.append(f"PAIR: {PAIR_A} vs {PAIR_B}")
    lines.append("")
    lines.append(f"corr={corr:.4f} spread_z={spread_z:+.3f} beta={beta:.4f}" if corr is not None else "Stats: NA")
    lines.append(f"CORR_LOW={CORR_LOW} CORR_RECOVER={CORR_RECOVER}")
    lines.append(f"Z_ENTRY={Z_ENTRY} Z_EXIT={Z_EXIT} CONFIRM_BARS={CONFIRM_BARS}")
    lines.append("")
    lines.append(f"Account: cash=${cash:.2f} buying_power=${bp:.2f} spendable=${spendable_usd():.2f}")
    lines.append("")
    lines.append("Positions (qty):")
    syms = WATCH_ETFS + ([DEFENSIVE_ETF] if DEFENSIVE_ETF else [])
    pos = positions_qty([s for s in syms if s])
    for s, q in pos.items():
        lines.append(f"  {s}: {q}")
    return "\n".join(lines)


def maybe_send_report(state: Dict[str, Any]) -> None:
    try:
        if not market_is_open():
            return
    except Exception as e:
        log_event("WARN", "ALPACA: get_clock failed (report check)", {"err": repr(e)})
        return

    now = now_utc()
    last_iso = state["meta"].get("last_report_time")
    last = dt.datetime.fromisoformat(last_iso) if last_iso else None

    if last is not None and (now - last).total_seconds() < REPORT_EVERY_SECONDS_OPEN:
        return

    body = format_report(state)
    try:
        sendReport(body=body, subject_prefix="Corr Breakdown Report")
        log_event("EMAIL", "report sent")
        state["meta"]["last_report_time"] = now.isoformat()
    except Exception as e:
        log_event("ERROR", "EMAIL report failed", {"err": repr(e)})


# ---------------- MAIN LOOP ----------------
def main() -> None:
    log_event("INFO", "BOT starting", {"watch_etfs": ",".join(WATCH_ETFS), "pair": f"{PAIR_A}/{PAIR_B}"})
    state = load_state()

    last_save: Optional[dt.datetime] = None
    if state["meta"].get("last_save_time"):
        try:
            last_save = dt.datetime.fromisoformat(state["meta"]["last_save_time"])
        except Exception:
            last_save = None

    while True:
        loop_t0 = time.time()
        try:
            state["meta"]["loop_count"] = int(state["meta"].get("loop_count", 0)) + 1
            state["meta"]["last_check_time"] = now_utc().isoformat()

            # Market status
            try:
                is_open = market_is_open()
            except Exception as e:
                log_event("ERROR", "ALPACA: get_clock failed -> skipping", {"err": repr(e)})
                is_open = False

            if not is_open:
                log_event("THINK", "No trading (market closed)")
                time.sleep(CHECK_SECONDS)
                continue

            # Open orders gate (blocks BOTH buys and sells)
            gate_symbols = set(WATCH_ETFS)
            if DEFENSIVE_ETF:
                gate_symbols.add(DEFENSIVE_ETF)
            open_orders = list_open_orders_for(gate_symbols)
            if open_orders:
                log_event("THINK", "SKIP trading (open orders exist)", {"count": len(open_orders)})
                maybe_send_report(state)
                time.sleep(CHECK_SECONDS)
                continue

            # Fetch aligned pair history and compute stats
            close_a, close_b = yf_pair_history(PAIR_A, PAIR_B)
            corr, spread_z, beta = rolling_corr_and_spread_z(close_a, close_b)

            # store for reporting
            state["meta"]["last_corr"] = corr
            state["meta"]["last_spread_z"] = spread_z
            state["meta"]["last_beta"] = beta

            mode = state["meta"].get("mode", "NORMAL")

            stats = {"corr": corr, "spread_z": spread_z, "beta": beta, "mode": mode, "loop": state["meta"]["loop_count"]}
            log_event("THINK", "Loop snapshot", stats)

            # ---- State machine transitions ----
            breakdown_condition = (corr < CORR_LOW) and (abs(spread_z) >= Z_ENTRY)
            recovery_condition = (corr > CORR_RECOVER) and (abs(spread_z) <= Z_EXIT)

            if mode in ("NORMAL", "BREAKDOWN_WATCH"):
                if breakdown_condition:
                    state["meta"]["breakdown_count"] = int(state["meta"].get("breakdown_count", 0)) + 1
                    if mode == "NORMAL":
                        set_mode(state, "BREAKDOWN_WATCH", "breakdown condition detected", stats)
                    log_event("THINK", "BREAKDOWN confirm", {"count": state["meta"]["breakdown_count"], **stats})

                    if state["meta"]["breakdown_count"] >= CONFIRM_BARS:
                        set_mode(state, "DIVERGED", "confirmed breakdown", stats)
                        go_defensive()
                else:
                    # reset if condition no longer holds
                    state["meta"]["breakdown_count"] = 0
                    if mode == "BREAKDOWN_WATCH":
                        set_mode(state, "NORMAL", "breakdown not sustained", stats)

                    # if NORMAL and we're flat, we can (re)invest
                    reinvest_if_needed(state, stats)

            elif mode == "DIVERGED":
                if recovery_condition:
                    state["meta"]["recovery_count"] = int(state["meta"].get("recovery_count", 0)) + 1
                    log_event("THINK", "RECOVERY confirm", {"count": state["meta"]["recovery_count"], **stats})
                    if state["meta"]["recovery_count"] >= CONFIRM_BARS:
                        set_mode(state, "RECOVERY", "confirmed recovery started", stats)
                        reinvest_if_needed(state, stats)
                else:
                    state["meta"]["recovery_count"] = 0
                    # stay defensive; do nothing

            elif mode == "RECOVERY":
                # If we recover enough, return to NORMAL; if breakdown again, go back to DIVERGED
                if breakdown_condition:
                    state["meta"]["breakdown_count"] = int(state["meta"].get("breakdown_count", 0)) + 1
                    if state["meta"]["breakdown_count"] >= CONFIRM_BARS:
                        set_mode(state, "DIVERGED", "re-breakdown during recovery", stats)
                        go_defensive()
                elif recovery_condition:
                    # stable recovery -> NORMAL
                    set_mode(state, "NORMAL", "recovery stable", stats)
                    reinvest_if_needed(state, stats)
                else:
                    # neither: drift, keep current holdings
                    pass

            # Reporting
            maybe_send_report(state)

            # Periodic save
            now = now_utc()
            if last_save is None or (now - last_save).total_seconds() >= SAVE_EVERY_SECONDS:
                save_state(state)
                last_save = now
                log_event("STATE", "saved", {"loop": state["meta"]["loop_count"]})

            loop_ms = (time.time() - loop_t0) * 1000.0
            log_event("INFO", "Loop complete", {"ms": loop_ms})

        except KeyboardInterrupt:
            log_event("INFO", "KeyboardInterrupt -> saving state and exiting")
            try:
                save_state(state)
                log_event("STATE", "saved on exit")
            except Exception as e:
                log_event("ERROR", "failed saving on exit", {"err": repr(e)})
            break
        except Exception as e:
            log_event("ERROR", "Unhandled exception in loop", {"err": repr(e)})

        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    main()
