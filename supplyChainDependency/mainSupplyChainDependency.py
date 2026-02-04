from __future__ import annotations

import os
import time
import pickle
import datetime as dt
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import yfinance as yf
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

from gmail import sendReport

# ===================== CONFIG =====================
load_dotenv()

LOG_FILE = "supply_chain_bot.log"
STATE_FILE = "supply_chain_state.pkl"

# Graph: leaders -> followers
# Start small; add more edges once stable.
LEADER_FOLLOWERS: Dict[str, List[str]] = {
    "NVDA": ["TSM", "ASML", "AMAT", "MU"],
    "AAPL": ["AVGO", "QRVO", "SWKS"],
    "MSFT": ["NVDA", "AMD"],
}

# Trading universe for gates
ALL_SYMBOLS = sorted({s for s in LEADER_FOLLOWERS} | {f for fs in LEADER_FOLLOWERS.values() for f in fs})

# Timing
CHECK_SECONDS = 60
SAVE_EVERY_SECONDS = 15 * 60
REPORT_EVERY_SECONDS_OPEN = 4 * 60 * 60

# Data (yfinance)
YF_INTERVAL = "5m"
YF_PERIOD = "5d"

# Trigger logic
LOOKBACK_BARS = 6                 # 6 x 5m = 30 minutes
LEADER_SHOCK_PCT = 0.02           # 2% move over lookback
FOLLOWER_MAX_MOVE_PCT = 0.008     # enter only if follower hasn't already moved > 0.8%

CONFIRM_CHECKS = 2                # require trigger for N consecutive loops
EDGE_COOLDOWN_SECONDS = 3 * 60 * 60  # per edge cooldown after any completed trade

# Trade management
HOLD_SECONDS = 2 * 60 * 60        # hold 2 hours (time stop)
TAKE_PROFIT_PCT = 0.010           # +1.0%
STOP_LOSS_PCT = 0.0075            # -0.75%

# Position sizing
STARTING_MONEY_USD = 100_000.0
PER_TRADE_USD = 2_500.0
MAX_CONCURRENT_TRADES = 5
MIN_BUY_USD = 50.0

BUY_COOLDOWN_SECONDS = 30 * 60    # global cooldown between new entries

# Alpaca
API_KEY = os.getenv("alpacaKey")
API_SECRET = os.getenv("alpacaSecret")
BASE_URL = os.getenv("alpacaBaseURL", "https://paper-api.alpaca.markets")
alpaca = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")


# ===================== LOGGING =====================
def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _fmt(ctx: Optional[Dict[str, Any]]) -> str:
    if not ctx:
        return ""
    parts = []
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
    line = f"[{ts}] {level:<5} {msg}{_fmt(ctx)}"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ===================== STATE =====================
def default_state() -> Dict[str, Any]:
    # edge_key = "LEADER>FOLLOWER"
    edge_states = {}
    for leader, followers in LEADER_FOLLOWERS.items():
        for fol in followers:
            ek = f"{leader}>{fol}"
            edge_states[ek] = {
                "state": "IDLE",              # IDLE, TRIGGERED, IN_TRADE, COOLDOWN
                "confirm": 0,
                "cooldown_until": None,
                "trigger_time": None,
                "trade": None,                # dict when in trade
            }
    return {
        "meta": {
            "loop_count": 0,
            "last_check_time": None,
            "last_save_time": None,
            "last_report_time": None,
            "last_entry_time": None,  # global cooldown
        },
        "edges": edge_states,
        "stats": {
            "trades_total": 0,
            "wins": 0,
            "losses": 0,
            "last_10": [],  # list of {edge, pnl_pct, reason, exit_time}
        }
    }


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        log_event("INFO", "STATE: none -> default_state()")
        return default_state()
    try:
        with open(STATE_FILE, "rb") as f:
            s = pickle.load(f)
        return s
    except Exception as e:
        log_event("ERROR", "STATE: failed load -> default_state()", {"err": repr(e)})
        return default_state()


def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, STATE_FILE)
    state["meta"]["last_save_time"] = now_utc().isoformat()


# ===================== ALPACA HELPERS =====================
def market_is_open() -> bool:
    return bool(alpaca.get_clock().is_open)


def list_open_orders_for(symbols: set[str]) -> List[Any]:
    try:
        orders = alpaca.list_orders(status="open", limit=200)
        return [o for o in orders if getattr(o, "symbol", "") in symbols]
    except Exception as e:
        log_event("WARN", "ALPACA: list_orders failed", {"err": repr(e)})
        return []


def safe_position_qty(symbol: str) -> float:
    try:
        pos = alpaca.get_position(symbol)
        return float(pos.qty)
    except Exception:
        return 0.0


def submit_buy_notional(symbol: str, usd: float) -> Optional[str]:
    try:
        o = alpaca.submit_order(
            symbol=symbol,
            notional=round(float(usd), 2),
            side="buy",
            type="market",
            time_in_force="day",
        )
        oid = getattr(o, "id", "")
        log_event("ACTION", "BUY", {"symbol": symbol, "usd": float(usd), "order_id": oid})
        return oid
    except Exception as e:
        log_event("ERROR", "BUY failed", {"symbol": symbol, "usd": float(usd), "err": repr(e)})
        return None


def submit_sell_all(symbol: str) -> Optional[str]:
    qty = safe_position_qty(symbol)
    if qty <= 0:
        return None
    try:
        o = alpaca.submit_order(
            symbol=symbol,
            qty=str(qty),
            side="sell",
            type="market",
            time_in_force="day",
        )
        oid = getattr(o, "id", "")
        log_event("ACTION", "SELL", {"symbol": symbol, "qty": qty, "order_id": oid})
        return oid
    except Exception as e:
        log_event("ERROR", "SELL failed", {"symbol": symbol, "qty": qty, "err": repr(e)})
        return None


def can_enter_global(state: Dict[str, Any]) -> bool:
    last = state["meta"].get("last_entry_time")
    if not last:
        return True
    try:
        last_dt = dt.datetime.fromisoformat(last)
    except Exception:
        return True
    return (now_utc() - last_dt).total_seconds() >= BUY_COOLDOWN_SECONDS


def mark_global_entry(state: Dict[str, Any]) -> None:
    state["meta"]["last_entry_time"] = now_utc().isoformat()


# ===================== DATA HELPERS =====================
def yf_last_bars(symbols: List[str]) -> Dict[str, np.ndarray]:
    """
    Returns dict symbol -> close array aligned to its own bars.
    For this bot we only need *per symbol* lookback returns, not cross-align.
    """
    out: Dict[str, np.ndarray] = {}
    tickers = yf.Tickers(" ".join(symbols))
    for s in symbols:
        h = tickers.tickers[s].history(period=YF_PERIOD, interval=YF_INTERVAL)
        if h is None or h.empty:
            raise RuntimeError(f"No yfinance history for {s}")
        closes = h["Close"].astype(float).to_numpy()
        if len(closes) < LOOKBACK_BARS + 2:
            raise RuntimeError(f"Not enough bars for {s}: have {len(closes)}")
        out[s] = closes
    return out


def pct_change(closes: np.ndarray, lookback: int) -> float:
    a = float(closes[-1])
    b = float(closes[-1 - lookback])
    if b == 0:
        return 0.0
    return (a / b) - 1.0


# ===================== REPORTING =====================
def send_mode_email(subject_prefix: str, body: str) -> None:
    try:
        sendReport(body=body, subject_prefix=subject_prefix)
        log_event("EMAIL", "sent", {"subject_prefix": subject_prefix})
    except Exception as e:
        log_event("ERROR", "EMAIL failed", {"err": repr(e), "subject_prefix": subject_prefix})


def format_summary(state: Dict[str, Any]) -> str:
    st = state["stats"]
    lines = []
    lines.append("Supply Chain Bot Summary")
    lines.append(f"Trades: {st['trades_total']} | Wins: {st['wins']} | Losses: {st['losses']}")
    lines.append("")
    lines.append("Active trades:")
    any_active = False
    for ek, ed in state["edges"].items():
        if ed["state"] == "IN_TRADE" and ed["trade"]:
            t = ed["trade"]
            any_active = True
            lines.append(f"  {ek} | sym={t['symbol']} entry={t['entry_price']:.2f} qty={t['qty']} tp={t['tp']:.2f} sl={t['sl']:.2f}")
    if not any_active:
        lines.append("  (none)")
    lines.append("")
    lines.append("Last 10 closed:")
    for row in st["last_10"][-10:]:
        lines.append(f"  {row['edge']} pnl={row['pnl_pct']:+.2f}% reason={row['reason']} at={row['exit_time']}")
    return "\n".join(lines)


def maybe_send_periodic_report(state: Dict[str, Any]) -> None:
    if not market_is_open():
        return
    now = now_utc()
    last_iso = state["meta"].get("last_report_time")
    last = dt.datetime.fromisoformat(last_iso) if last_iso else None
    if last is not None and (now - last).total_seconds() < REPORT_EVERY_SECONDS_OPEN:
        return
    send_mode_email("SupplyChain Report", format_summary(state))
    state["meta"]["last_report_time"] = now.isoformat()


# ===================== TRADE MANAGEMENT =====================
def enter_trade(state: Dict[str, Any], edge_key: str, follower: str, follower_price: float) -> None:
    # cap concurrent
    active = sum(1 for ed in state["edges"].values() if ed["state"] == "IN_TRADE")
    if active >= MAX_CONCURRENT_TRADES:
        log_event("THINK", "Entry blocked (max concurrent)", {"active": active, "max": MAX_CONCURRENT_TRADES, "edge": edge_key})
        return

    usd = PER_TRADE_USD
    if usd < MIN_BUY_USD:
        return

    oid = submit_buy_notional(follower, usd)
    if not oid:
        return

    # We can't guarantee immediate fill price; store observed price as approximation
    entry_price = follower_price
    tp = entry_price * (1.0 + TAKE_PROFIT_PCT)
    sl = entry_price * (1.0 - STOP_LOSS_PCT)
    expiry = (now_utc() + dt.timedelta(seconds=HOLD_SECONDS)).isoformat()

    qty = safe_position_qty(follower)  # may be 0 if not filled yet; ok (we'll re-read later)
    state["edges"][edge_key]["state"] = "IN_TRADE"
    state["edges"][edge_key]["trade"] = {
        "symbol": follower,
        "entry_price": float(entry_price),
        "entry_time": now_utc().isoformat(),
        "tp": float(tp),
        "sl": float(sl),
        "expiry_time": expiry,
        "qty": float(qty),
        "entry_order_id": oid,
    }

    mark_global_entry(state)

    send_mode_email(
        "SupplyChain ENTRY",
        f"ENTRY: {edge_key}\n"
        f"Follower: {follower}\n"
        f"Entry(ref): {entry_price:.2f}\n"
        f"TP: {tp:.2f} (+{TAKE_PROFIT_PCT*100:.2f}%)\n"
        f"SL: {sl:.2f} (-{STOP_LOSS_PCT*100:.2f}%)\n"
        f"Expiry: {expiry}\n"
    )


def exit_trade(state: Dict[str, Any], edge_key: str, reason: str, last_price: float) -> None:
    t = state["edges"][edge_key].get("trade")
    if not t:
        # reset edge safely
        state["edges"][edge_key]["state"] = "COOLDOWN"
        state["edges"][edge_key]["cooldown_until"] = (now_utc() + dt.timedelta(seconds=EDGE_COOLDOWN_SECONDS)).isoformat()
        return

    sym = t["symbol"]
    submit_sell_all(sym)

    pnl_pct = (float(last_price) / float(t["entry_price"]) - 1.0) * 100.0

    st = state["stats"]
    st["trades_total"] += 1
    if pnl_pct >= 0:
        st["wins"] += 1
    else:
        st["losses"] += 1

    st["last_10"].append({
        "edge": edge_key,
        "pnl_pct": float(pnl_pct),
        "reason": reason,
        "exit_time": now_utc().isoformat(),
    })
    st["last_10"] = st["last_10"][-10:]

    # move to cooldown
    state["edges"][edge_key]["state"] = "COOLDOWN"
    state["edges"][edge_key]["trade"] = None
    state["edges"][edge_key]["confirm"] = 0
    state["edges"][edge_key]["cooldown_until"] = (now_utc() + dt.timedelta(seconds=EDGE_COOLDOWN_SECONDS)).isoformat()

    send_mode_email(
        "SupplyChain EXIT",
        f"EXIT: {edge_key}\n"
        f"Symbol: {sym}\n"
        f"Reason: {reason}\n"
        f"Last(ref): {last_price:.2f}\n"
        f"PNL(ref): {pnl_pct:+.2f}%\n"
    )


def edge_in_cooldown(edge: Dict[str, Any]) -> bool:
    iso = edge.get("cooldown_until")
    if not iso:
        return False
    try:
        until = dt.datetime.fromisoformat(iso)
    except Exception:
        return False
    return now_utc() < until


# ===================== MAIN LOOP =====================
def main() -> None:
    log_event("INFO", "BOT starting", {"edges": sum(len(v) for v in LEADER_FOLLOWERS.values())})
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

            # Market + open orders gate
            if not market_is_open():
                log_event("THINK", "Market closed")
                time.sleep(CHECK_SECONDS)
                continue

            open_orders = list_open_orders_for(set(ALL_SYMBOLS))
            if open_orders:
                log_event("THINK", "SKIP trading (open orders exist)", {"count": len(open_orders)})
                maybe_send_periodic_report(state)
                time.sleep(CHECK_SECONDS)
                continue

            # Fetch last bars for all symbols in graph
            bars = yf_last_bars(ALL_SYMBOLS)

            # First: manage exits for any IN_TRADE edges
            for ek, ed in state["edges"].items():
                if ed["state"] != "IN_TRADE" or not ed.get("trade"):
                    continue
                t = ed["trade"]
                sym = t["symbol"]
                last_p = float(bars[sym][-1])

                # refresh qty in case fill occurred after entry
                if float(t.get("qty", 0.0)) <= 0:
                    t["qty"] = safe_position_qty(sym)

                tp = float(t["tp"])
                sl = float(t["sl"])
                expiry = dt.datetime.fromisoformat(t["expiry_time"])

                if last_p >= tp:
                    log_event("TRIG", "Exit TP", {"edge": ek, "price": last_p, "tp": tp})
                    exit_trade(state, ek, "TAKE_PROFIT", last_p)
                elif last_p <= sl:
                    log_event("TRIG", "Exit SL", {"edge": ek, "price": last_p, "sl": sl})
                    exit_trade(state, ek, "STOP_LOSS", last_p)
                elif now_utc() >= expiry:
                    log_event("TRIG", "Exit TIME", {"edge": ek, "price": last_p})
                    exit_trade(state, ek, "TIME_STOP", last_p)

            # Second: evaluate entries (IDLE/TRIGGERED/COOLDOWN)
            for leader, followers in LEADER_FOLLOWERS.items():
                leader_ret = pct_change(bars[leader], LOOKBACK_BARS)

                for fol in followers:
                    ek = f"{leader}>{fol}"
                    ed = state["edges"][ek]

                    # cooldown handling
                    if ed["state"] == "COOLDOWN":
                        if edge_in_cooldown(ed):
                            continue
                        ed["state"] = "IDLE"
                        ed["cooldown_until"] = None
                        ed["confirm"] = 0

                    if ed["state"] == "IN_TRADE":
                        continue  # already holding

                    # leader shock condition (directional or absolute)
                    shock = abs(leader_ret) >= LEADER_SHOCK_PCT

                    # follower lag condition
                    fol_ret = pct_change(bars[fol], LOOKBACK_BARS)
                    lag_ok = abs(fol_ret) <= FOLLOWER_MAX_MOVE_PCT

                    # Only enter when global cooldown ok
                    if not can_enter_global(state):
                        continue

                    if shock and lag_ok:
                        ed["confirm"] = int(ed.get("confirm", 0)) + 1
                        ed["state"] = "TRIGGERED"
                        log_event("THINK", "Edge trigger", {"edge": ek, "leader_ret": leader_ret, "fol_ret": fol_ret, "confirm": ed["confirm"]})

                        if ed["confirm"] >= CONFIRM_CHECKS:
                            # Enter trade on follower
                            fol_price = float(bars[fol][-1])
                            log_event("TRIG", "ENTER", {"edge": ek, "follower": fol, "price": fol_price, "leader_ret": leader_ret, "fol_ret": fol_ret})
                            enter_trade(state, ek, fol, fol_price)
                            # reset confirm (edge becomes IN_TRADE inside enter_trade)
                            ed["confirm"] = 0
                    else:
                        # reset if not sustained
                        if ed["state"] == "TRIGGERED" and ed.get("confirm", 0) > 0:
                            log_event("THINK", "Trigger reset", {"edge": ek})
                        ed["state"] = "IDLE"
                        ed["confirm"] = 0

            # periodic report + save
            maybe_send_periodic_report(state)

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
            log_event("ERROR", "Unhandled exception", {"err": repr(e)})

        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    main()
