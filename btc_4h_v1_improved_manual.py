import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

import btc_4h_knn_v12_lean as core


# ============================================================
# MANUAL TEST FILE
# ============================================================
# Leave the original improved lean file alone.
# Use this duplicate for manual testing.
#
# ONLY CHANGE THIS DATETIME:
# Put any Nigeria/WAT time inside the 4H candle you want to test or trade.
# Nigeria/WAT 4H candle starts are usually:
#   01:00, 05:00, 09:00, 13:00, 17:00, 21:00
#
# Example:
#   TARGET_DATETIME = pd.to_datetime("2026-04-27 09:00:00")
TARGET_DATETIME = pd.to_datetime("2026-04-23 09:00:00")


TP = 500.0
SL = 1000.0
LOCAL_TIMEZONE = "Africa/Lagos"
MIN_TRAIN_ROWS = 240
ROLLING_HOUR_LOOKBACK = 360

LOG_FILE = Path("/Users/goldenopuiyo/Desktop/codex_trading_bundle/btc-4h-v1-improved-manual-output.txt")

MOM_FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_6", "ret_12", "ret_18", "ret_42",
    "rsi_14", "body_pct", "close_loc_20",
    "dret_1", "dret_3", "dret_7", "prev_day_close_loc",
]

STRUCTURE_FEATURES = [
    "ret_1", "ret_3", "ret_6", "vol_6", "vol_12", "atr_14", "rsi_14",
    "body_pct", "range_pct", "close_loc_20", "dist_hi_42", "dist_lo_42",
    "vol_z_42", "dret_1", "dclose_loc",
    "prev_day_body", "prev_day_range", "prev_day_close_loc",
]

CANDIDATES = [
    "knn_mom_1440",
    "knn_struct_360",
    "knn_mom_720",
    "rf_struct",
    "et_struct",
]


def setup_logger():
    logger = logging.getLogger("btc_4h_v1_improved_manual")
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
    last_t = None
    last_close = None

    for t, row in candles.iterrows():
        op, high, low, close = map(float, (row["open"], row["high"], row["low"], row["close"]))
        last_t = t
        last_close = close
        order = [("low", low), ("high", high)] if close >= op else [("high", high), ("low", low)]

        if trade["bias"] == "Buy":
            if op >= trade["tp"]:
                return "tp_win", TP, t, "gap tp"
            if op <= trade["sl"]:
                return "sl_loss", -SL, t, "gap sl"
            for side, price in order:
                if side == "high" and price >= trade["tp"]:
                    return "tp_win", TP, t, "tp"
                if side == "low" and price <= trade["sl"]:
                    return "sl_loss", -SL, t, "sl"
        else:
            if op <= trade["tp"]:
                return "tp_win", TP, t, "gap tp"
            if op >= trade["sl"]:
                return "sl_loss", -SL, t, "gap sl"
            for side, price in order:
                if side == "low" and price <= trade["tp"]:
                    return "tp_win", TP, t, "tp"
                if side == "high" and price >= trade["sl"]:
                    return "sl_loss", -SL, t, "sl"

    if last_close is None:
        return "pending", 0.0, None, "target candle not available"

    pnl = float(last_close - trade["entry"]) if trade["bias"] == "Buy" else float(trade["entry"] - last_close)
    if pnl > 0.0:
        return "close_win", pnl, last_t, "close at candle end in profit"
    if pnl < 0.0:
        return "close_loss", pnl, last_t, "close at candle end in loss"
    return "flat", 0.0, last_t, "close flat"


def make_training_rows(raw_4h, features_4h, features_daily, target_ts):
    rows = []
    last_train_ts = target_ts - pd.Timedelta(hours=4)

    for ts in raw_4h.index:
        if ts > last_train_ts:
            break

        row = core.feature_row(ts, raw_4h, features_4h, features_daily)
        candles = target_candle(ts, raw_4h)
        if row is None or len(candles) == 0:
            continue

        buy_out, buy_pnl, _buy_ts, _buy_note = evaluate_trade(build_trade(row["entry"], "Buy"), candles)
        sell_out, sell_pnl, _sell_ts, _sell_note = evaluate_trade(build_trade(row["entry"], "Sell"), candles)

        label = 1 if buy_pnl >= sell_pnl else 0
        edge = abs(buy_pnl - sell_pnl)
        row.update({
            "label": label,
            "weight": max(0.25, min(5.0, edge / TP)),
            "label_end": ts + pd.Timedelta(hours=4),
            "buy_pnl": buy_pnl,
            "sell_pnl": sell_pnl,
        })
        rows.append(row)

    return pd.DataFrame(rows).dropna().sort_values("ts").reset_index(drop=True)


def trend_score(row: pd.Series) -> float:
    return float(
        0.25 * np.tanh(row["ret_1"] * 80.0)
        + 0.25 * np.tanh(row["ret_3"] * 45.0)
        + 0.20 * np.tanh(row["ret_6"] * 30.0)
        + 0.15 * np.tanh(row["ret_12"] * 20.0)
        + 0.15 * np.tanh(row["dret_1"] * 12.0)
        + 0.10 * row["prev_day_close_loc"]
    )


def adjust_prob(prob: float, row: pd.Series, mode: str) -> float:
    trend = trend_score(row)

    if mode == "rescue":
        if abs(prob - 0.5) <= 0.035 and trend >= 0.15:
            return 0.62
        if abs(prob - 0.5) <= 0.035 and trend <= -0.15:
            return 0.38
        if -0.16 <= trend < -0.10:
            return 1.0 - prob

    if mode == "fade":
        if abs(prob - 0.5) < 0.08 and row["close_loc_20"] > 0.42 and row["ret_3"] > 0:
            return min(prob, 0.44)
        if abs(prob - 0.5) < 0.08 and row["close_loc_20"] < -0.42 and row["ret_3"] < 0:
            return max(prob, 0.56)

    return float(prob)


def knn_prob(train: pd.DataFrame, row: pd.Series, features, lookback: int, k: int, mode: str) -> float:
    train = train.tail(lookback)
    if len(train) < MIN_TRAIN_ROWS:
        return 0.5

    scaler = RobustScaler()
    x = scaler.fit_transform(train[features].values.astype(float))
    z = scaler.transform(row[features].values.astype(float).reshape(1, -1))[0]
    distance = np.sqrt(((x - z) ** 2).sum(axis=1))
    idx = np.argsort(distance)[: min(k, len(distance))]
    weights = train["weight"].values.astype(float)[idx] / (distance[idx] + 1e-9)
    prob = float(np.average(train["label"].values.astype(float)[idx], weights=weights))
    return adjust_prob(prob, row, mode)


def _fit_tree_pipe(tree_train: pd.DataFrame, features, kind: str):
    if kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=250,
            max_depth=7,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        pipe = make_pipeline(RobustScaler(), clf)
        pipe.fit(
            tree_train[features].values.astype(float),
            tree_train["label"].values.astype(int),
            randomforestclassifier__sample_weight=tree_train["weight"].values.astype(float),
        )
    else:
        clf = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=12,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        pipe = make_pipeline(RobustScaler(), clf)
        pipe.fit(
            tree_train[features].values.astype(float),
            tree_train["label"].values.astype(int),
            extratreesclassifier__sample_weight=tree_train["weight"].values.astype(float),
        )

    return pipe


def fit_tree_models(train_df: pd.DataFrame, years):
    models = {}
    for year in sorted(set(int(year) for year in years)):
        year_start = pd.Timestamp(year, 1, 1)
        tree_train = train_df[train_df["ts"] < year_start]
        if len(tree_train) < MIN_TRAIN_ROWS:
            tree_train = train_df[train_df["ts"] < train_df["ts"].max()]
        if len(tree_train) < MIN_TRAIN_ROWS:
            continue
        models[("rf", year)] = _fit_tree_pipe(tree_train, STRUCTURE_FEATURES, "rf")
        models[("et", year)] = _fit_tree_pipe(tree_train, STRUCTURE_FEATURES, "et")
    return models


def tree_prob(row: pd.Series, kind: str, tree_models) -> float:
    model = tree_models.get((kind, int(row["ts"].year)))
    if model is None:
        return 0.5
    return float(model.predict_proba(row[STRUCTURE_FEATURES].values.astype(float).reshape(1, -1))[0, 1])


def candidate_probs(train: pd.DataFrame, row: pd.Series, tree_models):
    return {
        "knn_mom_1440": knn_prob(train, row, MOM_FEATURES, 1440, 35, "rescue"),
        "knn_struct_360": knn_prob(train, row, STRUCTURE_FEATURES, 360, 15, "fade"),
        "knn_mom_720": knn_prob(train, row, MOM_FEATURES, 720, 25, "rescue"),
        "rf_struct": tree_prob(row, "rf", tree_models),
        "et_struct": tree_prob(row, "et", tree_models),
    }


def score_recent_same_hour(train_df: pd.DataFrame, raw_4h: pd.DataFrame, target_ts: pd.Timestamp, tree_models):
    same_hour = train_df[train_df["ts"].dt.hour == target_ts.hour].tail(ROLLING_HOUR_LOOKBACK)
    if len(same_hour) < 30:
        return "rf_struct", {}

    scores = {name: 0.0 for name in CANDIDATES}
    for _, hist_row in same_hour.iterrows():
        hist_train = train_df[(train_df["label_end"] <= hist_row["ts"]) & (train_df["ts"] < hist_row["ts"])]
        if len(hist_train) < MIN_TRAIN_ROWS:
            continue
        probs = candidate_probs(hist_train, hist_row, tree_models)
        candles = target_candle(hist_row["ts"], raw_4h)
        for name, prob in probs.items():
            bias = "Buy" if prob >= 0.5 else "Sell"
            _outcome, pnl, _hit, _note = evaluate_trade(build_trade(hist_row["entry"], bias), candles)
            scores[name] += pnl

    return max(scores, key=scores.get), scores


def main():
    logger = setup_logger()
    target_ts = target_utc_from_local(TARGET_DATETIME)
    target_local_start = target_ts.tz_localize("UTC").tz_convert(LOCAL_TIMEZONE)
    cutoff = target_ts - pd.Timedelta(hours=4)
    fetch_end = min(now_utc_naive(), target_ts + pd.Timedelta(hours=4))

    if fetch_end < cutoff:
        raise ValueError(
            f"Too early for target candle {target_local_start}. "
            "Run after the previous 4H candle has closed."
        )

    raw_4h = core.fetch_ohlcv("4h", core.START_4H, fetch_end)
    raw_daily = core.fetch_ohlcv("1d", core.START_DAILY, fetch_end)
    model_4h = raw_4h[raw_4h.index <= cutoff].copy()

    if cutoff not in model_4h.index:
        raise ValueError(f"Missing prior 4H close for UTC {cutoff}.")

    features_4h = core.add_4h_features(model_4h)
    features_daily = core.add_daily_features(raw_daily[raw_daily.index <= cutoff.normalize()].copy())
    target_payload = core.feature_row(target_ts, model_4h, features_4h, features_daily)
    if target_payload is None:
        raise ValueError(f"Could not build target features for UTC {target_ts}.")

    train_df = make_training_rows(model_4h, features_4h, features_daily, target_ts)
    target_row = pd.Series(target_payload)
    tree_years = set(train_df["ts"].dt.year.tolist() + [target_ts.year])
    tree_models = fit_tree_models(train_df, tree_years)
    picked_model, recent_scores = score_recent_same_hour(train_df, model_4h, target_ts, tree_models)
    probs = candidate_probs(train_df, target_row, tree_models)

    prob = probs[picked_model]
    bias = "Buy" if prob >= 0.5 else "Sell"
    trade = build_trade(target_row["entry"], bias)

    candles = target_candle(target_ts, raw_4h)
    if now_utc_naive() < target_ts + pd.Timedelta(hours=4):
        outcome, pnl, hit_ts, note = "pending", 0.0, None, "target candle has not closed yet"
    else:
        outcome, pnl, hit_ts, note = evaluate_trade(trade, candles)

    logger.info("BTC/USD 4H V1 improved manual runner")
    logger.info("Exchange:               %s", core.EXCHANGE_ID)
    logger.info("Only edited datetime:   %s WAT", TARGET_DATETIME)
    logger.info("Target candle WAT:      %s", target_local_start)
    logger.info("Target candle UTC:      %s", target_ts)
    logger.info("Inputs stop at UTC:     %s", cutoff)
    logger.info("Fixed TP/SL:            %.0f / %.0f", TP, SL)
    logger.info("Training rows used:     %d", len(train_df))
    logger.info("Tree models fitted:     %d", len(tree_models))
    logger.info("Rolling same-hour pick: %s", picked_model)
    if recent_scores:
        logger.info(
            "Recent same-hour scores: %s",
            " | ".join(f"{name} {score:+.0f}" for name, score in sorted(recent_scores.items())),
        )
    logger.info(
        "Candidate probabilities: %s",
        " | ".join(f"{name} {prob:.3f}" for name, prob in sorted(probs.items())),
    )
    logger.info("")
    logger.info("FINAL 4H TRADE DECISION")
    logger.info("Bias:                   %s", bias)
    logger.info("Entry:                  $%.2f", trade["entry"])
    logger.info("TP:                     $%.2f", trade["tp"])
    logger.info("SL:                     $%.2f", trade["sl"])
    logger.info("Picked prob buy:        %.3f", prob)
    logger.info("Outcome:                %s", outcome)
    logger.info("PnL:                    %+.2f", pnl)
    logger.info("Hit time UTC:           %s", hit_ts if hit_ts is not None else "-")
    logger.info("Note:                   %s", note)
    logger.info("Log saved to %s", LOG_FILE)


if __name__ == "__main__":
    main()
