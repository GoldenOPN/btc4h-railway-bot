import logging
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# ============================================================
# ONLY CHANGE THIS DATETIME
# ============================================================
# Put any Nigeria/WAT time inside the 4H candle you want to test or trade.
# The script automatically maps it to the correct exchange 4H candle in UTC.
#
# Bitstamp 4H UTC candles start at:
#   00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
#
# Nigeria/WAT candle start times are:
#   01:00, 05:00, 09:00, 13:00, 17:00, 21:00 WAT
#
# Backtest example:
#   TARGET_DATETIME = pd.to_datetime("2026-04-22 01:00:00")
#
# Live example:
#   To trade the 09:00 WAT candle, run after the 05:00 WAT candle has closed
#   and set TARGET_DATETIME to that 09:00 WAT candle time.
TARGET_DATETIME = pd.to_datetime("2026-04-23 01:00:00")


EXCHANGE_ID = "bitstamp"
SYMBOL = "BTC/USD"
LOCAL_TIMEZONE = "Africa/Lagos"

# Best 4H setting selected by 2025 calibration, then forward-tested on 2026.
TP = 500.0
SL = 1000.0

START_4H = "2023-01-01"
START_DAILY = "2022-01-01"
FETCH_LIMIT = 1000

LOG_FILE = Path("/Users/goldenopuiyo/Desktop/codex_trading_bundle/btc-4h-knn-v12-lean-output.txt")

FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_6", "ret_12", "ret_18", "ret_42",
    "vol_6", "vol_12", "vol_24", "vol_42", "atr_14", "rsi_14",
    "body_pct", "range_pct", "close_loc_20", "dist_hi_42", "dist_lo_42", "vol_z_42",
    "dret_1", "dret_3", "dret_7", "dvol_7", "dclose_loc",
    "prev_day_body", "prev_day_range", "prev_day_close_loc",
]


def setup_logger():
    logger = logging.getLogger("btc_4h_knn_v12_lean")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def target_utc_from_local(target_dt: pd.Timestamp) -> pd.Timestamp:
    target = pd.Timestamp(target_dt)
    if target.tzinfo is None:
        target = target.tz_localize(LOCAL_TIMEZONE)
    else:
        target = target.tz_convert(LOCAL_TIMEZONE)
    return target.tz_convert("UTC").floor("4h").tz_localize(None)


def now_utc_naive() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def fetch_ohlcv(timeframe: str, start_date: str, end_ts: pd.Timestamp) -> pd.DataFrame:
    exchange = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    step_ms = {"4h": 4 * 60 * 60 * 1000, "1d": 24 * 60 * 60 * 1000}[timeframe]

    since = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ms = int((end_ts + pd.Timedelta(days=1)).tz_localize("UTC").timestamp() * 1000)
    rows = []

    while since < end_ms:
        batch = exchange.fetch_ohlcv(SYMBOL, timeframe=timeframe, since=since, limit=FETCH_LIMIT)
        if not batch:
            break
        rows.extend(batch)
        next_since = int(batch[-1][0]) + step_ms
        if next_since <= since:
            break
        since = next_since

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[df.index <= end_ts].copy()


def add_4h_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close, high, low, open_, volume = out["close"], out["high"], out["low"], out["open"], out["volume"]
    ret = np.log(close / close.shift(1))
    true_range = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)

    for n in (1, 2, 3, 6, 12, 18, 42):
        out[f"ret_{n}"] = np.log(close / close.shift(n))
    for n in (6, 12, 24, 42):
        out[f"vol_{n}"] = ret.rolling(n, n).std()

    out["atr_14"] = true_range.rolling(14, 14).mean() / close
    gain = ret.clip(lower=0).rolling(14, 14).mean()
    loss = (-ret.clip(upper=0)).rolling(14, 14).mean()
    rs = gain / (loss + 1e-9)
    out["rsi_14"] = ((100.0 - 100.0 / (1.0 + rs)) - 50.0) / 50.0
    out["body_pct"] = (close - open_) / (high - low + 1e-9)
    out["range_pct"] = (high - low) / close
    out["close_loc_20"] = (
        (close - low.rolling(20, 20).min())
        / (high.rolling(20, 20).max() - low.rolling(20, 20).min() + 1e-9)
        - 0.5
    )
    out["dist_hi_42"] = close / high.rolling(42, 42).max() - 1.0
    out["dist_lo_42"] = close / low.rolling(42, 42).min() - 1.0
    log_volume = np.log1p(volume)
    out["vol_z_42"] = (
        (log_volume - log_volume.rolling(42, 42).mean())
        / (log_volume.rolling(42, 42).std() + 1e-9)
    )

    return out.replace([np.inf, -np.inf], np.nan).dropna().copy()


def add_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close, high, low, open_ = out["close"], out["high"], out["low"], out["open"]
    ret = np.log(close / close.shift(1))

    out["dret_1"] = ret
    out["dret_3"] = np.log(close / close.shift(3))
    out["dret_7"] = np.log(close / close.shift(7))
    out["dvol_7"] = ret.rolling(7, 7).std()
    out["dclose_loc"] = (
        (close - low.rolling(7, 7).min())
        / (high.rolling(7, 7).max() - low.rolling(7, 7).min() + 1e-9)
        - 0.5
    )
    out["prev_day_body"] = (close - open_) / (high - low + 1e-9)
    out["prev_day_range"] = (high - low) / close
    out["prev_day_close_loc"] = (close - low) / (high - low + 1e-9) - 0.5

    return out.replace([np.inf, -np.inf], np.nan).dropna().copy()


def feature_row(ts: pd.Timestamp, raw_4h: pd.DataFrame, features_4h: pd.DataFrame, features_daily: pd.DataFrame):
    anchor = ts - pd.Timedelta(hours=4)
    if anchor not in raw_4h.index or anchor not in features_4h.index:
        return None

    previous_daily_close = ts.normalize() - pd.Timedelta(days=1)
    usable_daily = features_daily[features_daily.index <= previous_daily_close]
    if len(usable_daily) == 0:
        return None

    row_4h = features_4h.loc[anchor]
    row_d = usable_daily.iloc[-1]
    payload = {"ts": ts, "entry": float(raw_4h.loc[anchor, "close"])}

    for feature in FEATURES:
        payload[feature] = row_d[feature] if feature in features_daily.columns else row_4h[feature]

    return payload


def target_candle(ts: pd.Timestamp, raw_4h: pd.DataFrame) -> pd.DataFrame:
    return raw_4h[(raw_4h.index >= ts) & (raw_4h.index < ts + pd.Timedelta(hours=4))]


def build_trade(entry: float, bias: str):
    return {
        "bias": bias,
        "entry": float(entry),
        "tp": float(entry + TP if bias == "Buy" else entry - TP),
        "sl": float(entry - SL if bias == "Buy" else entry + SL),
    }


def evaluate_trade(trade: dict, candles: pd.DataFrame):
    for t, row in candles.iterrows():
        op, high, low, close = map(float, (row["open"], row["high"], row["low"], row["close"]))

        if trade["bias"] == "Buy":
            if op >= trade["tp"]:
                return "win", TP, t, "gap tp"
            if op <= trade["sl"]:
                return "loss", -SL, t, "gap sl"
            order = [("low", low), ("high", high)] if close >= op else [("high", high), ("low", low)]
            for side, price in order:
                if side == "high" and price >= trade["tp"]:
                    return "win", TP, t, f"tp via {order[0][0]}->{order[1][0]}"
                if side == "low" and price <= trade["sl"]:
                    return "loss", -SL, t, f"sl via {order[0][0]}->{order[1][0]}"
        else:
            if op <= trade["tp"]:
                return "win", TP, t, "gap tp"
            if op >= trade["sl"]:
                return "loss", -SL, t, "gap sl"
            order = [("low", low), ("high", high)] if close >= op else [("high", high), ("low", low)]
            for side, price in order:
                if side == "low" and price <= trade["tp"]:
                    return "win", TP, t, f"tp via {order[0][0]}->{order[1][0]}"
                if side == "high" and price >= trade["sl"]:
                    return "loss", -SL, t, f"sl via {order[0][0]}->{order[1][0]}"

    return "no_hit", 0.0, None, "no barrier"


def make_training_rows(raw_4h: pd.DataFrame, features_4h: pd.DataFrame, features_daily: pd.DataFrame, target_ts: pd.Timestamp):
    rows = []
    last_train_ts = target_ts - pd.Timedelta(hours=4)

    for ts in raw_4h.index:
        if ts > last_train_ts:
            break

        row = feature_row(ts, raw_4h, features_4h, features_daily)
        candles = target_candle(ts, raw_4h)
        if row is None or len(candles) == 0:
            continue

        entry = row["entry"]
        buy_out, buy_pnl, _buy_ts, _buy_note = evaluate_trade(build_trade(entry, "Buy"), candles)
        sell_out, sell_pnl, _sell_ts, _sell_note = evaluate_trade(build_trade(entry, "Sell"), candles)

        if buy_pnl > sell_pnl:
            label, weight = 1, 1.0
        elif sell_pnl > buy_pnl:
            label, weight = 0, 1.0
        else:
            max_up = max(0.0, float(candles["high"].max() - entry))
            max_down = max(0.0, float(entry - candles["low"].min()))
            label, weight = int(max_up >= max_down), 0.25

        row.update({"label": label, "weight": weight})
        rows.append(row)

    return pd.DataFrame(rows).dropna().sort_values("ts").reset_index(drop=True)


def predict_prob(train_df: pd.DataFrame, target_row: pd.Series) -> float:
    train = train_df.tail(720)
    if len(train) < 240:
        return 0.5

    scaler = RobustScaler()
    x = scaler.fit_transform(train[FEATURES].values.astype(float))
    z = scaler.transform(target_row[FEATURES].values.astype(float).reshape(1, -1))[0]
    y = train["label"].values.astype(float)
    sample_weight = train["weight"].values.astype(float)

    distance = np.sqrt(((x - z) ** 2).sum(axis=1))
    idx = np.argsort(distance)[: min(25, len(distance))]
    weights = sample_weight[idx] / (distance[idx] + 1e-9)
    return float(np.average(y[idx], weights=weights))


def trend_score(row: pd.Series) -> float:
    return float(
        0.25 * np.tanh(row["ret_1"] * 80.0)
        + 0.25 * np.tanh(row["ret_3"] * 45.0)
        + 0.20 * np.tanh(row["ret_6"] * 30.0)
        + 0.15 * np.tanh(row["ret_12"] * 20.0)
        + 0.15 * np.tanh(row["dret_1"] * 12.0)
        + 0.10 * row["prev_day_close_loc"]
    )


def final_prob(base: float, row: pd.Series):
    trend = trend_score(row)
    prob = base
    rescue = False

    if abs(base - 0.5) <= 0.035 and trend >= 0.15:
        prob, rescue = 0.62, True
    elif abs(base - 0.5) <= 0.035 and trend <= -0.15:
        prob, rescue = 0.38, True

    corrected = (not rescue) and -0.16 <= trend < -0.10
    if corrected:
        prob = 1.0 - prob

    return prob, trend, rescue, corrected


def main():
    logger = setup_logger()
    target_ts = target_utc_from_local(TARGET_DATETIME)
    target_local_start = target_ts.tz_localize("UTC").tz_convert(LOCAL_TIMEZONE)
    cutoff = target_ts - pd.Timedelta(hours=4)
    fetch_end = min(now_utc_naive(), target_ts + pd.Timedelta(hours=4))

    if fetch_end < cutoff:
        raise ValueError(
            f"Too early for target candle {target_local_start}. "
            f"Run after the previous 4H candle ending at {target_local_start} has closed."
        )

    raw_4h = fetch_ohlcv("4h", START_4H, fetch_end)
    raw_daily = fetch_ohlcv("1d", START_DAILY, fetch_end)
    model_4h = raw_4h[raw_4h.index <= cutoff].copy()

    if cutoff not in model_4h.index:
        raise ValueError(f"Missing prior 4H close for UTC {cutoff}.")

    features_4h = add_4h_features(model_4h)
    features_daily = add_daily_features(raw_daily[raw_daily.index <= cutoff.normalize()].copy())
    target_payload = feature_row(target_ts, model_4h, features_4h, features_daily)

    if target_payload is None:
        raise ValueError(f"Could not build target features for UTC {target_ts}.")

    train_df = make_training_rows(model_4h, features_4h, features_daily, target_ts)
    target_row = pd.Series(target_payload)
    base = predict_prob(train_df, target_row)
    prob, trend, rescue, corrected = final_prob(base, target_row)

    bias = "Buy" if prob >= 0.5 else "Sell"
    trade = build_trade(float(target_row["entry"]), bias)
    candles = target_candle(target_ts, raw_4h)

    if now_utc_naive() < target_ts + pd.Timedelta(hours=4):
        outcome, pnl, hit_ts, note = "pending", 0.0, None, "target 4H candle has not fully closed yet"
    elif len(candles) == 0:
        outcome, pnl, hit_ts, note = "pending", 0.0, None, "target 4H candle not available yet"
    else:
        outcome, pnl, hit_ts, note = evaluate_trade(trade, candles)

    logger.info("BTC/USD 4H KNN v12-style lean runner")
    logger.info("Exchange:               %s", EXCHANGE_ID)
    logger.info("Only edited datetime:   %s WAT", TARGET_DATETIME)
    logger.info("Target candle WAT:      %s", target_local_start)
    logger.info("Target candle UTC:      %s", target_ts)
    logger.info("Inputs stop at UTC:     %s", cutoff)
    logger.info("Fetched data through:   %s", fetch_end)
    logger.info("Fixed TP/SL:            %.0f / %.0f", TP, SL)
    logger.info("Training rows used:     %d", len(train_df))
    logger.info(
        "Signal stats:           base %.3f | final %.3f | trend %.3f | rescue %s | corrected %s",
        base,
        prob,
        trend,
        "yes" if rescue else "no",
        "yes" if corrected else "no",
    )
    logger.info("")
    logger.info("FINAL 4H TRADE DECISION")
    logger.info("Bias:                   %s", bias)
    logger.info("Entry:                  $%.2f", trade["entry"])
    logger.info("TP:                     $%.2f", trade["tp"])
    logger.info("SL:                     $%.2f", trade["sl"])
    logger.info("Outcome:                %s", outcome)
    logger.info("PnL:                    %+.2f", pnl)
    logger.info("Hit time UTC:           %s", hit_ts if hit_ts is not None else "-")
    logger.info("Note:                   %s", note)
    logger.info("Log saved to %s", LOG_FILE)


if __name__ == "__main__":
    main()
