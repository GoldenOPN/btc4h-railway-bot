import json
import os
import re
import traceback
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

import btc_4h_v1_cloud_runner as signal_core


AQUA_URL = "https://webtrading.aquafunded.com/terminal"
BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "aqua-4h-github-runner.log"
STATE_FILE = BASE_DIR / "aqua-4h-agent-state.json"

INITIAL_BALANCE = float(os.environ.get("AQUA_INITIAL_BALANCE", "100000"))
TRAILING_RATE = 0.05
DAILY_DD = 0.03 * INITIAL_BALANCE
VALID_DAY_PROFIT = 0.005 * INITIAL_BALANCE
CONSISTENCY_LIMIT = 0.15
DEFAULT_PAYOUT_DAYS = int(os.environ.get("AQUA_PAYOUT_DAYS", "14"))


def log(message: str):
    line = f"{pd.Timestamp.now(tz='UTC').isoformat()} | {message}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def telegram(text: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=data,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20):
        pass


def current_target_local() -> pd.Timestamp:
    override = os.environ.get("TARGET_DATETIME")
    if override:
        return pd.to_datetime(override)
    return signal_core.current_target_local()


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def day_key(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%d")


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def to_float(text: str) -> float:
    cleaned = text.replace(" ", "").replace(",", "")
    return float(cleaned)


def account_snapshot(page) -> dict:
    body = page.locator("body").inner_text()
    balance_match = re.search(r"Balance:\s*([0-9][0-9\s.,]*)", body)
    equity_match = re.search(r"Equity:\s*([0-9][0-9\s.,]*)", body)
    if not balance_match or not equity_match:
        raise RuntimeError("Could not parse balance/equity from AquaFunded web terminal.")
    balance = to_float(balance_match.group(1))
    equity = to_float(equity_match.group(1))
    return {"balance": balance, "equity": equity}


def ensure_state(state: dict, snap: dict, now: pd.Timestamp) -> dict:
    if os.environ.get("AQUA_RESET_CYCLE") == "1":
        state = {}

    if not state:
        state = {
            "initial_balance": INITIAL_BALANCE,
            "cycle_start_balance": snap["balance"],
            "cycle_start_time": now.isoformat(),
            "highest_balance": max(INITIAL_BALANCE, snap["balance"]),
            "daily_anchor_day": day_key(now),
            "daily_anchor_balance": snap["balance"],
            "daily_realized": {},
            "closed_trades": [],
            "first_trade_time": None,
        }

    state["highest_balance"] = max(float(state.get("highest_balance", INITIAL_BALANCE)), snap["balance"])

    today = day_key(now)
    if state.get("daily_anchor_day") != today:
        state["daily_anchor_day"] = today
        state["daily_anchor_balance"] = snap["balance"]
        state.setdefault("daily_realized", {})

    return state


def record_realized_pnl(state: dict, now: pd.Timestamp, pnl: float, note: str):
    if abs(pnl) < 1e-9:
        return
    today = day_key(now)
    daily = state.setdefault("daily_realized", {})
    daily[today] = round(float(daily.get(today, 0.0)) + pnl, 2)

    trades = state.setdefault("closed_trades", [])
    trades.append({"ts": now.isoformat(), "pnl": round(float(pnl), 2), "note": note})
    state["closed_trades"] = trades[-200:]


def payout_metrics(state: dict, snap: dict, now: pd.Timestamp) -> dict:
    daily_realized = state.get("daily_realized", {})
    positive_days = [float(v) for v in daily_realized.values() if float(v) > 0.0]
    biggest_day_profit = max(positive_days) if positive_days else 0.0
    cycle_profit = float(snap["balance"] - float(state["cycle_start_balance"]))
    required_profit_for_consistency = (
        (biggest_day_profit / CONSISTENCY_LIMIT) + 0.01 if biggest_day_profit > 0.0 else 0.0
    )
    consistency_ok = cycle_profit > 0.0 and (biggest_day_profit < CONSISTENCY_LIMIT * cycle_profit if biggest_day_profit > 0.0 else True)
    valid_days = sum(1 for v in daily_realized.values() if float(v) >= VALID_DAY_PROFIT)
    first_trade_time = pd.to_datetime(state["first_trade_time"]) if state.get("first_trade_time") else None
    payout_wait_met = False
    if first_trade_time is not None:
        payout_wait_met = (now - first_trade_time).days >= DEFAULT_PAYOUT_DAYS

    highest_balance = float(state["highest_balance"])
    trailing_floor = highest_balance * (1.0 - TRAILING_RATE)
    trailing_cushion = float(snap["equity"] - trailing_floor)

    daily_anchor = float(state["daily_anchor_balance"])
    daily_floor = daily_anchor - DAILY_DD
    daily_cushion = float(snap["equity"] - daily_floor)
    current_day_profit = float(daily_realized.get(day_key(now), 0.0))

    payout_ready = bool(
        cycle_profit > 0.0
        and valid_days >= 5
        and consistency_ok
        and payout_wait_met
    )

    return {
        "cycle_profit": round(cycle_profit, 2),
        "biggest_day_profit": round(biggest_day_profit, 2),
        "required_profit_for_consistency": round(required_profit_for_consistency, 2),
        "valid_days": int(valid_days),
        "consistency_ok": bool(consistency_ok),
        "payout_wait_met": bool(payout_wait_met),
        "payout_ready": bool(payout_ready),
        "trailing_floor": round(trailing_floor, 2),
        "trailing_cushion": round(trailing_cushion, 2),
        "daily_floor": round(daily_floor, 2),
        "daily_cushion": round(daily_cushion, 2),
        "current_day_profit": round(current_day_profit, 2),
    }


def recent_win_rate(state: dict, limit: int = 12) -> float:
    trades = [t for t in state.get("closed_trades", []) if abs(float(t["pnl"])) > 1e-9]
    if not trades:
        return 0.0
    recent = trades[-limit:]
    wins = sum(1 for t in recent if float(t["pnl"]) > 0.0)
    return wins / len(recent)


def choose_lot_size(metrics: dict, state: dict) -> tuple[float, str]:
    if metrics["payout_ready"]:
        return 0.0, "Payout conditions met. Stay flat until you request the withdrawal."

    if metrics["daily_cushion"] < 1200:
        return 0.0, "Daily drawdown cushion too small."

    if metrics["trailing_cushion"] < 2200:
        return 0.0, "Trailing drawdown cushion too small."

    if metrics["current_day_profit"] <= -1200:
        return 0.0, "Daily soft stop hit after losses."

    cycle_profit = metrics["cycle_profit"]
    current_day_profit = metrics["current_day_profit"]
    recent_wr = recent_win_rate(state)
    consistency_gap = metrics["cycle_profit"] - metrics["required_profit_for_consistency"]

    daily_profit_cap = 600.0
    if cycle_profit >= 4000:
        daily_profit_cap = 800.0
    if cycle_profit >= 7000:
        daily_profit_cap = 1200.0

    if current_day_profit >= daily_profit_cap:
        return 0.0, "Consistency guard: daily profit cap reached, stop for the day."

    # Start smaller to protect drawdown and keep single-day profits payout-friendly.
    lot = 0.4
    reason = "Base size 0.4"

    if (
        cycle_profit >= 2500
        and metrics["valid_days"] >= 2
        and recent_wr >= 0.60
        and metrics["trailing_cushion"] >= 3200
        and metrics["daily_cushion"] >= 1800
        and consistency_gap >= 800
    ):
        lot = 0.6
        reason = "Raised to 0.6 after good run and enough drawdown/consistency buffer"

    if (
        cycle_profit >= 5000
        and metrics["valid_days"] >= 4
        and recent_wr >= 0.68
        and metrics["trailing_cushion"] >= 4500
        and metrics["daily_cushion"] >= 2200
        and consistency_gap >= 1800
    ):
        lot = 0.8
        reason = "Raised to 0.8 after stronger cycle progress"

    if (
        cycle_profit >= 8000
        and metrics["valid_days"] >= 5
        and recent_wr >= 0.75
        and metrics["trailing_cushion"] >= 6500
        and metrics["daily_cushion"] >= 2500
        and consistency_gap >= 3000
    ):
        lot = 1.0
        reason = "Raised to 1.0 only after payout-friendly buffer is strong"

    return lot, reason


def login_if_needed(page):
    page.goto(AQUA_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(5000)

    login_box = page.get_by_placeholder("Enter Login")
    if login_box.count() == 0:
        page.get_by_text("New Order").first.wait_for(timeout=30000)
        return

    login = os.environ["AQUA_MT5_LOGIN"]
    password = os.environ["AQUA_MT5_PASSWORD"]
    page.get_by_placeholder("Enter Login").fill(login)
    page.get_by_placeholder("Enter Password").fill(password)
    page.get_by_role("button", name="Connect to account").click()
    page.get_by_text("New Order").first.wait_for(timeout=30000)
    page.wait_for_timeout(3000)


def no_positions_visible(page) -> bool:
    empty = page.get_by_text("You don’t have any positions")
    return empty.count() > 0 and empty.first.is_visible()


def close_all_positions(page):
    closed = []
    for _ in range(5):
        if no_positions_visible(page):
            break

        tickets = page.get_by_text(re.compile(r"\b\d{8}\b"))
        if tickets.count() == 0:
            break

        tickets.last.dblclick()
        close_btn = page.get_by_role("button", name=re.compile(r"^Close #"))
        close_btn.wait_for(timeout=10000)
        close_text = close_btn.inner_text()
        close_btn.click()
        page.get_by_role("button", name="OK").wait_for(timeout=10000)
        page.get_by_role("button", name="OK").click()
        page.wait_for_timeout(2000)
        closed.append(close_text)

    return closed


def open_order_ticket(page):
    create_btn = page.get_by_role("button", name="Create New Order")
    if create_btn.count() > 0 and create_btn.first.is_visible():
        create_btn.first.click()
        page.wait_for_timeout(1500)
        return

    page.get_by_text("New Order").first.click()
    page.wait_for_timeout(1500)


def fill_order(page, sig: dict, volume: float):
    inputs = page.locator("input:visible")
    if inputs.count() < 3:
        raise RuntimeError("Could not locate enough visible input fields in the order ticket.")

    inputs.nth(0).fill(f"{volume:.2f}")
    inputs.nth(1).fill(f"{sig['sl']:.2f}")
    inputs.nth(2).fill(f"{sig['tp']:.2f}")

    if inputs.count() >= 4:
        try:
            inputs.nth(3).fill(f"4h {sig['target_local']} {sig['bias']}")
        except Exception:
            pass


def place_signal_order(page, sig: dict, volume: float):
    open_order_ticket(page)
    fill_order(page, sig, volume)
    button_name = "Buy by Market" if sig["bias"] == "Buy" else "Sell by Market"
    page.get_by_role("button", name=button_name).click()
    page.get_by_role("button", name="OK").wait_for(timeout=10000)
    page.get_by_role("button", name="OK").click()
    page.wait_for_timeout(2000)


def run():
    now = now_utc()
    state = load_state()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        login_if_needed(page)

        snap_before = account_snapshot(page)
        state = ensure_state(state, snap_before, now)

        closed = close_all_positions(page)
        page.wait_for_timeout(1500)
        snap_after_close = account_snapshot(page)
        realized_pnl = round(snap_after_close["balance"] - snap_before["balance"], 2)
        if closed:
            record_realized_pnl(state, now, realized_pnl, "; ".join(closed))
            log(f"Closed prior position(s) | realized_pnl={realized_pnl:+.2f} | {closed}")

        state = ensure_state(state, snap_after_close, now)
        metrics = payout_metrics(state, snap_after_close, now)
        lot_size, lot_reason = choose_lot_size(metrics, state)

        summary = [
            f"balance={snap_after_close['balance']:.2f}",
            f"equity={snap_after_close['equity']:.2f}",
            f"cycle_profit={metrics['cycle_profit']:+.2f}",
            f"valid_days={metrics['valid_days']}",
            f"biggest_day={metrics['biggest_day_profit']:.2f}",
            f"consistency_need={metrics['required_profit_for_consistency']:.2f}",
            f"trailing_floor={metrics['trailing_floor']:.2f}",
            f"daily_floor={metrics['daily_floor']:.2f}",
            f"lot={lot_size:.2f}",
        ]
        log("Account agent | " + " | ".join(summary) + f" | reason={lot_reason}")

        if lot_size <= 0.0:
            save_state(state)
            browser.close()
            message = "Aqua 4H: no new trade\n" + "\n".join(summary) + "\nReason: " + lot_reason
            telegram(message)
            return

        target_local = current_target_local()
        sig = signal_core.build_signal(target_local)
        if not state.get("first_trade_time"):
            state["first_trade_time"] = now.isoformat()

        place_signal_order(page, sig, lot_size)
        page.wait_for_timeout(1500)
        snap_after_open = account_snapshot(page)
        state = ensure_state(state, snap_after_open, now)
        save_state(state)
        browser.close()

    message = "\n".join(
        [
            "Aqua 4H trade executed",
            f"Target candle: {sig['target_local']} WAT",
            f"Bias: {sig['bias']}",
            f"Entry: ${sig['entry']:.2f}",
            f"TP: ${sig['tp']:.2f}",
            f"SL: ${sig['sl']:.2f}",
            f"Prob buy: {sig['prob_buy']:.3f}",
            f"Model: {sig['picked_model']}",
            f"Lot size: {lot_size:.2f}",
            f"Lot reason: {lot_reason}",
            f"Cycle profit: {metrics['cycle_profit']:+.2f}",
            f"Valid payout days: {metrics['valid_days']}",
            f"Need total profit >= {metrics['required_profit_for_consistency']:.2f} for consistency",
        ]
    )
    telegram(message)
    log("Execution complete")


if __name__ == "__main__":
    try:
        run()
    except PlaywrightTimeoutError:
        error = traceback.format_exc()
        log("Playwright timeout\n" + error)
        telegram("Aqua 4H runner timeout\n\n" + error[-3000:])
        raise
    except Exception:
        error = traceback.format_exc()
        log("Runner failed\n" + error)
        telegram("Aqua 4H runner failed\n\n" + error[-3000:])
        raise
