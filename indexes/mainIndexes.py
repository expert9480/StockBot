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
REPORT_EVERY_SECONDS_OPEN = 4 * 60 * 60  # every 4 hours while market open

# Strategy thresholds
# Risk-off: sell when VIX spikes
VIX_SELL_LEVEL = 28.0

# Normal regime: allowed to buy / (re)invest when VIX is calm
VIX_BUY_LEVEL = 20.0

# Optional confirmation before buying (3 consecutive upticks for EACH ETF)
UPTREND_WINDOW = 3
REQUIRE_UPTICK_FOR_BUY = False  # set True if you want the 3-check confirmation

# Buy safety
STARTING_MONEY_USD = 100_000.00
MAX_TRADE_USD_PER_SYMBOL = 50_000.00
MIN_BUY_USD = 50.0                 # ignore tiny buys
BUY_COOLDOWN_SECONDS = 30 * 60     # don't buy more than once per 30 minutes

# Watched ETFs
WATCH_ETFS = ["SPY", "VOO", "VT"]
VIX_SYMBOL = "^VIX"

# Alpaca env vars
API_KEY = os.getenv("alpacaKey")
API_SECRET = os.getenv("alpacaSecret")
BASE_URL = os.getenv("alpacaBaseURL", "https://paper-api.alpaca.markets")


# ----------------- LOGGING (FILE ONLY) -----------------
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


def log_line(msg: str) -> None:
    log_event("INFO", msg)


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


# ----------------- STATE -----------------
def default_state() -> Dict[str, Any]:
    return {
        "day": now_utc().date().isoformat(),
        "trees": {sym: PriceBST() for sym in (WATCH_ETFS + [VIX_SYMBOL])},
        "meta": {
            "mode": "INVESTED",        # INVESTED or SOLD_OUT (risk-off)
            "sell_snapshot": {},       # {sym: sell_price}
            "sell_time": None,         # iso timestamp
            "last_report_prices": {},  # {sym: price_at_last_report}
            "day_open_prices": {},     # {sym: open_price_for_day}
            "last_report_time": None,  # iso timestamp
            "last_save_time": None,    # iso timestamp
            "last_check_time": None,   # iso timestamp
            "uptick_buffer": {sym: [] for sym in WATCH_ETFS},
            "loop_count": 0,
            "last_buy_time": None,     # iso timestamp
        },
    }


def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, STATE_FILE)
    state["meta"]["last_save_time"] = now_utc().isoformat()


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        log_event("INFO", "STATE: no existing state file -> using default_state()")
        return default_state()

    try:
        with open(STATE_FILE, "rb") as f:
            s = pickle.load(f)

        today = now_utc().date().isoformat()
        if s.get("day") != today:
            log_event("INFO", "STATE: day mismatch -> resetting state", {"loaded_day": s.get("day"), "today": today})
            s = default_state()
        return s
    except Exception as e:
        log_event("ERROR", "STATE: failed to load -> starting fresh", {"err": repr(e)})
        return default_state()


# ----------------- MARKET DATA (YFINANCE) -----------------
def yf_last_price(symbol: str) -> float:
    t0 = time.time()
    t = yf.Ticker(symbol)

    price: Optional[float] = None
    try:
        price = t.fast_info.get("last_price", None)
    except Exception:
        price = None

    if price is None:
        hist = t.history(period="1d", interval="1m")
        if hist is None or hist.empty:
            raise RuntimeError(f"No yfinance data for {symbol}")
        price = float(hist["Close"].iloc[-1])

    dt_ms = (time.time() - t0) * 1000.0
    log_event("DATA", "yfinance last price", {"symbol": symbol, "price": float(price), "ms": dt_ms})
    return float(price)


def yf_day_open_price(symbol: str) -> float:
    t0 = time.time()
    t = yf.Ticker(symbol)

    hist = t.history(period="1d", interval="1d")
    if hist is None or hist.empty:
        hist = t.history(period="5d", interval="1d")
        if hist is None or hist.empty:
            raise RuntimeError(f"No open price data for {symbol}")

    op = float(hist["Open"].iloc[-1])
    dt_ms = (time.time() - t0) * 1000.0
    log_event("DATA", "yfinance open price", {"symbol": symbol, "open": op, "ms": dt_ms})
    return op


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
    return max(0.0, usable - 5.00)


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


# ----------------- STRATEGY HELPERS -----------------
def update_uptick_buffer(state: Dict[str, Any], sym: str, price: float) -> None:
    buf: List[float] = state["meta"]["uptick_buffer"].setdefault(sym, [])
    buf.append(price)
    while len(buf) > UPTREND_WINDOW:
        buf.pop(0)


def is_upticking_3(state: Dict[str, Any]) -> bool:
    for sym in WATCH_ETFS:
        buf: List[float] = state["meta"]["uptick_buffer"].get(sym, [])
        if len(buf) < UPTREND_WINDOW:
            return False
        if not (buf[0] < buf[1] < buf[2]):
            return False
    return True


def uptick_debug(state: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for sym in WATCH_ETFS:
        buf = state["meta"]["uptick_buffer"].get(sym, [])
        seq_ok = False
        if len(buf) >= 3:
            seq_ok = bool(buf[0] < buf[1] < buf[2])
        out[sym] = {"buf": [round(x, 4) for x in buf], "3_up": seq_ok}
    return out


def positions_qty(symbols: List[str]) -> Dict[str, float]:
    return {s: safe_position_qty(s) for s in symbols}


def is_flat(symbols: List[str]) -> bool:
    q = positions_qty(symbols)
    return all(qty <= 0 for qty in q.values())


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

    try:
        is_open = market_is_open()
    except Exception as e:
        log_event("WARN", "ALPACA: get_clock failed (report check)", {"err": repr(e)})
        return

    if not is_open:
        return

    now = now_utc()
    last_iso = state["meta"].get("last_report_time")
    last = dt.datetime.fromisoformat(last_iso) if last_iso else None

    if last is not None and (now - last).total_seconds() < REPORT_EVERY_SECONDS_OPEN:
        return

    body = format_report(state, prices)
    try:
        sendReport(body=body, subject_prefix="Index/VIX Report")
        log_event("EMAIL", "report sent")
    except Exception as e:
        log_event("ERROR", "EMAIL report failed", {"err": repr(e)})

    for sym, p in prices.items():
        state["meta"]["last_report_prices"][sym] = p
    state["meta"]["last_report_time"] = now.isoformat()


# ----------------- MAIN LOOP -----------------
def main() -> None:
    log_event("INFO", "BOT starting", {"watch_etfs": ",".join(WATCH_ETFS), "vix": VIX_SYMBOL})
    state = load_state()

    # Ensure trees exist
    for sym in (WATCH_ETFS + [VIX_SYMBOL]):
        state["trees"].setdefault(sym, PriceBST())

    # Startup: record opens once per day
    if not state["meta"].get("day_open_prices"):
        log_event("INFO", "INIT: fetching day open prices")
        try:
            for sym in (WATCH_ETFS + [VIX_SYMBOL]):
                state["meta"]["day_open_prices"][sym] = yf_day_open_price(sym)
        except Exception as e:
            log_event("ERROR", "INIT: failed to fetch open prices", {"err": repr(e)})

    # Startup report
    if ENABLE_REPORTS and ENABLE_STARTUP_REPORT:
        try:
            prices = {sym: yf_last_price(sym) for sym in (WATCH_ETFS + [VIX_SYMBOL])}
            body = "Bot initialized and running.\n\n" + format_report(state, prices)
            sendReport(body=body, subject_prefix="Startup")
            log_event("EMAIL", "startup report sent")
        except Exception as e:
            log_event("ERROR", "EMAIL startup failed", {"err": repr(e)})

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

            # day rollover safeguard
            today = now_utc().date().isoformat()
            if state.get("day") != today:
                log_event("INFO", "DAY rollover -> resetting state", {"prev_day": state.get("day"), "today": today})
                state = default_state()

            # Fetch prices
            prices: Dict[str, float] = {}
            data_errors: Dict[str, str] = {}
            for sym in (WATCH_ETFS + [VIX_SYMBOL]):
                try:
                    prices[sym] = yf_last_price(sym)
                except Exception as e:
                    data_errors[sym] = repr(e)

            if data_errors:
                log_event("WARN", "DATA: pricing incomplete -> skipping trading", {"errors": data_errors})
                maybe_send_report(state, prices)
                now = now_utc()
                if last_save is None or (now - last_save).total_seconds() >= SAVE_EVERY_SECONDS:
                    save_state(state)
                    last_save = now
                    log_event("STATE", "saved (after data error)")
                time.sleep(CHECK_SECONDS)
                continue

            # Insert into BSTs
            ts = time.time()
            for sym, p in prices.items():
                state["trees"][sym].insert(ts, p)

            # Update uptick buffers
            for sym in WATCH_ETFS:
                update_uptick_buffer(state, sym, prices[sym])

            # Backfill open prices if missing
            for sym in (WATCH_ETFS + [VIX_SYMBOL]):
                if sym not in state["meta"]["day_open_prices"]:
                    try:
                        state["meta"]["day_open_prices"][sym] = yf_day_open_price(sym)
                    except Exception as e:
                        log_event("WARN", "DATA: failed to backfill open price", {"symbol": sym, "err": repr(e)})

            # Market status
            try:
                is_open = market_is_open()
            except Exception as e:
                log_event("ERROR", "ALPACA: get_clock failed -> skipping trading", {"err": repr(e)})
                is_open = False

            mode = state["meta"].get("mode", "INVESTED")
            vix = float(prices[VIX_SYMBOL])
            flat = is_flat(WATCH_ETFS)

            log_event(
                "THINK",
                "Loop snapshot",
                {
                    "loop": state["meta"]["loop_count"],
                    "mode": mode,
                    "market_open": is_open,
                    "vix": vix,
                    "buy_lvl": VIX_BUY_LEVEL,
                    "sell_lvl": VIX_SELL_LEVEL,
                    "flat": flat,
                },
            )

            # Open orders gate (blocks BOTH buys and sells)
            open_orders = list_open_orders_for(set(WATCH_ETFS))
            if open_orders:
                summary = []
                for o in open_orders[:10]:
                    summary.append(
                        {
                            "sym": getattr(o, "symbol", ""),
                            "side": getattr(o, "side", ""),
                            "type": getattr(o, "type", ""),
                            "qty": getattr(o, "qty", ""),
                            "notional": getattr(o, "notional", ""),
                            "status": getattr(o, "status", ""),
                            "id": getattr(o, "id", ""),
                        }
                    )
                log_event("THINK", "SKIP trading (open orders exist)", {"count": len(open_orders), "orders": summary})
            else:
                # --- SELL RULE (risk-off) ---
                if mode == "INVESTED" and vix >= VIX_SELL_LEVEL:
                    log_event("THINK", "Evaluate SELL", {"vix": vix, "threshold": VIX_SELL_LEVEL, "market_open": is_open})
                    if is_open:
                        log_event("TRIG", "SELL: VIX spike -> SELL ALL", {"vix": vix})
                        any_sold = False
                        for sym in WATCH_ETFS:
                            any_sold = submit_sell_all(sym) or any_sold

                        state["meta"]["sell_snapshot"] = {sym: prices[sym] for sym in WATCH_ETFS}
                        state["meta"]["sell_time"] = now_utc().isoformat()
                        state["meta"]["mode"] = "SOLD_OUT"

                        log_event("INFO", "SELL complete -> SOLD_OUT", {"any_sold": any_sold})
                    else:
                        log_event("THINK", "SELL blocked (market closed)", {"vix": vix})

                # --- BUY RULE (normal regime) ---
                else:
                    # buys are allowed when VIX is calm
                    if not is_open:
                        log_event("THINK", "No trading (market closed)")
                    elif vix > VIX_BUY_LEVEL:
                        log_event("THINK", "No buying (VIX above calm threshold)", {"vix": vix, "buy_lvl": VIX_BUY_LEVEL})
                    elif not buy_cooldown_ok(state):
                        log_event("THINK", "No buying (cooldown)", {"cooldown_sec": BUY_COOLDOWN_SECONDS})
                    else:
                        # Only (re)invest if either:
                        # - we are SOLD_OUT (risk-off), OR
                        # - we are INVESTED but actually flat (no positions)
                        should_invest = (mode == "SOLD_OUT") or (mode == "INVESTED" and flat)

                        if not should_invest:
                            log_event("THINK", "No action (already invested with positions)")
                        else:
                            uptick_ok = is_upticking_3(state)
                            if REQUIRE_UPTICK_FOR_BUY and not uptick_ok:
                                log_event("THINK", "WAIT: calm regime but uptick not confirmed", {"uptick": uptick_debug(state)})
                            else:
                                alloc = compute_per_symbol_buy_usd()
                                cash, bp = account_cash_and_bp()
                                log_event(
                                    "TRIG",
                                    "BUY: (re)investing in calm regime",
                                    {
                                        "mode": mode,
                                        "flat": flat,
                                        "vix": vix,
                                        "cash": cash,
                                        "buying_power": bp,
                                        "spendable": round(spendable_usd(), 2),
                                        "alloc": {k: round(v, 2) for k, v in alloc.items()},
                                        "require_uptick": REQUIRE_UPTICK_FOR_BUY,
                                        "uptick_ok": uptick_ok,
                                    },
                                )

                                placed_any = False
                                for sym in WATCH_ETFS:
                                    usd = float(alloc.get(sym, 0.0))
                                    if usd >= MIN_BUY_USD:
                                        placed_any = submit_buy_notional(sym, usd) or placed_any
                                    else:
                                        log_event("THINK", "BUY skipped (too small)", {"symbol": sym, "usd": usd, "min": MIN_BUY_USD})

                                mark_buy_time(state)
                                state["meta"]["mode"] = "INVESTED"
                                state["meta"]["sell_snapshot"] = {}
                                state["meta"]["sell_time"] = None
                                log_event("INFO", "BUY step complete -> INVESTED", {"placed_any": placed_any})

            # Reporting
            maybe_send_report(state, prices)

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
