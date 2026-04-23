import json
import logging
import os
import traceback
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

import btc_4h_v1_improved_manual as signal


BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "btc-4h-v1-cloud-state.json"
RUNNER_LOG = BASE_DIR / "btc-4h-v1-cloud-runner.log"
LOCAL_TIMEZONE = "Africa/Lagos"


def setup_logger():
    logger = logging.getLogger("btc_4h_v1_cloud_runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (
        logging.FileHandler(RUNNER_LOG, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def load_state():
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def telegram_enabled():
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN")) and bool(os.environ.get("TELEGRAM_CHAT_ID"))


def send_telegram(text: str):
    if not telegram_enabled():
        return
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=20):
        pass


def current_target_local():
    now_utc = pd.Timestamp.now(tz="UTC").floor("4h")
    return now_utc.tz_convert(LOCAL_TIMEZONE).tz_localize(None)


def ts_to_wire(ts: pd.Timestamp):
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def wire_to_ts(text: str):
    return pd.to_datetime(text)


def build_signal(target_local: pd.Timestamp):
    signal.TARGET_DATETIME = pd.Timestamp(target_local)
    target_ts = signal.target_utc_from_local(signal.TARGET_DATETIME)
    target_local_start = target_ts.tz_localize("UTC").tz_convert(LOCAL_TIMEZONE)
    cutoff = target_ts - pd.Timedelta(hours=4)
    fetch_end = min(signal.now_utc_naive(), target_ts + pd.Timedelta(hours=4))

    raw_4h = signal.core.fetch_ohlcv("4h", signal.core.START_4H, fetch_end)
    raw_daily = signal.core.fetch_ohlcv("1d", signal.core.START_DAILY, fetch_end)
    model_4h = raw_4h[raw_4h.index <= cutoff].copy()

    if cutoff not in model_4h.index:
        raise ValueError(f"Missing prior 4H close for UTC {cutoff}.")

    features_4h = signal.core.add_4h_features(model_4h)
    features_daily = signal.core.add_daily_features(raw_daily[raw_daily.index <= cutoff.normalize()].copy())
    target_payload = signal.core.feature_row(target_ts, model_4h, features_4h, features_daily)
    if target_payload is None:
        raise ValueError(f"Could not build target features for UTC {target_ts}.")

    train_df = signal.make_training_rows(model_4h, features_4h, features_daily, target_ts)
    target_row = pd.Series(target_payload)
    tree_years = set(train_df["ts"].dt.year.tolist() + [target_ts.year])
    tree_models = signal.fit_tree_models(train_df, tree_years)
    picked_model, recent_scores = signal.score_recent_same_hour(train_df, model_4h, target_ts, tree_models)
    probs = signal.candidate_probs(train_df, target_row, tree_models)
    prob = probs[picked_model]
    bias = "Buy" if prob >= 0.5 else "Sell"
    trade = signal.build_trade(target_row["entry"], bias)

    return {
        "target_utc": ts_to_wire(target_ts),
        "target_local": ts_to_wire(target_local_start.tz_localize(None)),
        "cutoff_utc": ts_to_wire(cutoff),
        "bias": bias,
        "entry": float(trade["entry"]),
        "tp": float(trade["tp"]),
        "sl": float(trade["sl"]),
        "prob_buy": float(prob),
        "picked_model": picked_model,
        "recent_scores": {k: float(v) for k, v in recent_scores.items()},
        "all_probs": {k: float(v) for k, v in probs.items()},
    }


def evaluate_open_trade(open_trade: dict):
    target_ts = wire_to_ts(open_trade["target_utc"])
    fetch_end = signal.now_utc_naive()
    raw_4h = signal.core.fetch_ohlcv("4h", signal.core.START_4H, fetch_end)
    candles = signal.target_candle(target_ts, raw_4h)
    trade = {
        "bias": open_trade["bias"],
        "entry": float(open_trade["entry"]),
        "tp": float(open_trade["tp"]),
        "sl": float(open_trade["sl"]),
    }
    outcome, pnl, hit_ts, note = signal.evaluate_trade(trade, candles)
    return {
        "outcome": outcome,
        "pnl": float(pnl),
        "hit_utc": ts_to_wire(hit_ts) if hit_ts is not None else "",
        "note": note,
    }


def format_signal_message(sig: dict):
    lines = [
        "BTC/USD 4H signal",
        f"Target candle: {sig['target_local']} WAT",
        f"Bias: {sig['bias']}",
        f"Entry: ${sig['entry']:.2f}",
        f"TP: ${sig['tp']:.2f}",
        f"SL: ${sig['sl']:.2f}",
        f"Prob buy: {sig['prob_buy']:.3f}",
        f"Picked model: {sig['picked_model']}",
    ]
    return "\n".join(lines)


def format_close_message(open_trade: dict, result: dict):
    lines = [
        "BTC/USD 4H close update",
        f"Target candle: {open_trade['target_local']} WAT",
        f"Bias: {open_trade['bias']}",
        f"Entry: ${float(open_trade['entry']):.2f}",
        f"TP: ${float(open_trade['tp']):.2f}",
        f"SL: ${float(open_trade['sl']):.2f}",
        f"Outcome: {result['outcome']}",
        f"PnL: {result['pnl']:+.2f}",
        f"Hit UTC: {result['hit_utc'] or '-'}",
        f"Note: {result['note']}",
    ]
    return "\n".join(lines)


def main():
    logger = setup_logger()
    state = load_state()
    target_local = current_target_local()
    current_target_utc = ts_to_wire(signal.target_utc_from_local(target_local))

    open_trade = state.get("open_trade")
    if open_trade and open_trade.get("target_utc") != current_target_utc:
        result = evaluate_open_trade(open_trade)
        logger.info("Closed prior candle %s with %s %+.2f", open_trade["target_utc"], result["outcome"], result["pnl"])
        try:
            send_telegram(format_close_message(open_trade, result))
        except Exception as exc:
            logger.warning("Telegram close notification failed: %s", exc)
        state["last_closed"] = {
            "target_utc": open_trade["target_utc"],
            "outcome": result["outcome"],
            "pnl": result["pnl"],
        }
        state.pop("open_trade", None)
        save_state(state)

    if state.get("last_signal_target_utc") == current_target_utc:
        logger.info("Signal for %s already sent. Nothing to do.", current_target_utc)
        return

    sig = build_signal(target_local)
    signal.LOG_FILE = BASE_DIR / f"btc-4h-v1-signal-{pd.to_datetime(sig['target_utc']).strftime('%Y%m%d-%H%M')}.txt"
    logger.info("New signal %s %s entry %.2f tp %.2f sl %.2f", sig["target_local"], sig["bias"], sig["entry"], sig["tp"], sig["sl"])

    try:
        send_telegram(format_signal_message(sig))
    except Exception as exc:
        logger.warning("Telegram signal notification failed: %s", exc)

    state["last_signal_target_utc"] = sig["target_utc"]
    state["open_trade"] = sig
    save_state(state)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error_text = traceback.format_exc()
        logger = setup_logger()
        logger.error("Cloud runner failed\n%s", error_text)
        try:
            send_telegram("BTC/USD 4H cloud runner error\n\n" + error_text[-3000:])
        except Exception:
            pass
        raise
