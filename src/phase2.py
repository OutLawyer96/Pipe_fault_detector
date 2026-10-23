from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import signal, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_fscore_support, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .pipeline import LEAK_MAP, inventory, parse_path

SEED = 42
DOI = "10.17632/tbrnp6vrnj.1"
TARGETS = {"Accelerometer": 8000, "Dynamic Pressure Sensor": 860, "Hydrophone": 8000}
NODE_CHANNELS = ("A1", "A2", "P1", "P2")
HYDRO_CHANNELS = ("H1", "H2")
LEAK_TYPES = list(LEAK_MAP.values())


def read_resampled(path: Path, sensor_type: str) -> tuple[np.ndarray, float]:
    if path.suffix.lower() == ".csv":
        d = pd.read_csv(path)
        t = d.iloc[:, 0].to_numpy(float)
        x = d.iloc[:, 1].to_numpy(float)
        fs = 1.0 / np.median(np.diff(t))
    else:
        x = np.fromfile(path, dtype="<i4").astype(float)
        fs = 8000.0
    target = TARGETS[sensor_type]
    if abs(fs - target) > 1:
        ratio = target / fs
        # resample_poly supplies the anti-alias low-pass filter before decimation.
        x = signal.resample_poly(x, target, int(round(fs)))
    return x.astype(np.float32), float(target)


def compact_features(x: np.ndarray, fs: float, prefix: str) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return {f"{prefix}_{name}": 0.0 for name in ("mean", "std", "rms", "low", "acoustic", "centroid")}
    centered = x - np.mean(x)
    f, p = signal.periodogram(centered, fs=fs, detrend=False)
    p = np.maximum(p, 1e-15)
    def energy(lo: float, hi: float) -> float:
        return float(np.log1p(np.sum(p[(f >= lo) & (f < min(hi, fs / 2))])))
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_rms": float(np.sqrt(np.mean(x * x))),
        f"{prefix}_low": energy(0, 500),
        f"{prefix}_acoustic": energy(500, 3000),
        f"{prefix}_centroid": float(np.sum(f * p) / np.sum(p)),
    }


def make_features(root: Path, meta: pd.DataFrame, window_s: float, include_hydro: bool) -> pd.DataFrame:
    parsed = meta[meta.parse_ok].copy()
    rows: list[dict] = []
    for sid, group in parsed.groupby("signal_id"):
        channels: dict[str, tuple[np.ndarray, float, str]] = {}
        for _, item in group.iterrows():
            if item.channel in NODE_CHANNELS + HYDRO_CHANNELS:
                channels[item.channel] = (*read_resampled(root / item.path, item.sensor_type), item.sensor_type)
        needed = NODE_CHANNELS + (HYDRO_CHANNELS if include_hydro else ())
        if not all(c in channels for c in needed):
            continue
        duration = min(len(channels[c][0]) / channels[c][1] for c in needed)
        n_windows = int(np.floor(duration / window_s))
        for wi in range(n_windows):
            start_s = wi * window_s
            row = {
                "signal_id": sid, "group": sid, "window_index": wi,
                "leak_type": group.iloc[0].leak_type,
                "leak": int(group.iloc[0].leak_type != "No-leak"),
                "network": group.iloc[0].network, "flow": group.iloc[0].flow,
                "noise": group.iloc[0].noise,
            }
            for channel in needed:
                x, fs, _ = channels[channel]
                start = int(round(start_s * fs))
                row.update(compact_features(x[start:start + int(round(window_s * fs))], fs, channel))
            for a, b in (("A1", "A2"), ("P1", "P2")):
                xa, xb = channels[a][0], channels[b][0]
                sa = int(round(start_s * channels[a][1])); sb = int(round(start_s * channels[b][1]))
                na = int(round(window_s * channels[a][1])); nb = int(round(window_s * channels[b][1]))
                ya, yb = xa[sa:sa + na], xb[sb:sb + nb]
                row[f"{a}_{b}_std_ratio"] = float(np.std(ya) / (np.std(yb) + 1e-12))
            rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(frame: pd.DataFrame, include_hydro: bool) -> list[str]:
    excluded = {"signal_id", "group", "window_index", "leak_type", "leak", "network", "flow", "noise"}
    cols = [c for c in frame.columns if c not in excluded and frame[c].dtype != object]
    if not include_hydro:
        cols = [c for c in cols if not c.startswith("H")]
    return cols


def transform_train_test(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], variant: str) -> tuple[np.ndarray, np.ndarray]:
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    a = train[cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    b = test[cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    if variant == "raw":
        return a, b
    if variant == "per_recording":
        def normalize(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
            out = values.copy()
            for _, group in frame.groupby("signal_id", sort=False):
                ix = group.index.to_numpy()
                med = np.median(out[ix], axis=0)
                mad = np.median(np.abs(out[ix] - med), axis=0)
                out[ix] = (out[ix] - med) / (1.4826 * mad + 1e-6)
            return out
        return normalize(train, a), normalize(test, b)
    if variant == "baseline_relative":
        def relative(frame: pd.DataFrame, values: np.ndarray, baseline: pd.DataFrame | None) -> np.ndarray:
            out = values.copy()
            if baseline is None or baseline.empty:
                return out
            key = ["network", "flow", "noise", "window_index"]
            base = baseline.groupby(key)[cols].median()
            for i, (_, row) in enumerate(frame.iterrows()):
                k = tuple(row[x] for x in key)
                if k in base.index:
                    out[i] -= base.loc[k].to_numpy(float)
            return out
        baseline = train[train.leak == 0]
        return relative(train, a, baseline), relative(test, b, baseline)
    raise ValueError(f"Unknown preprocessing variant: {variant}")


def model_factory(edge: bool, seed: int = SEED):
    if edge:
        return RandomForestClassifier(n_estimators=30, max_depth=6, min_samples_leaf=2,
                                      class_weight="balanced", random_state=seed, n_jobs=-1)
    return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                  random_state=seed, n_jobs=-1)


def safe_auc(y: np.ndarray, score: np.ndarray, average: str = "macro") -> float:
    return float(roc_auc_score(y, score, average=average)) if len(np.unique(y)) == 2 else float("nan")


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (score >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": safe_auc(y, score),
        "pr_auc": float(average_precision_score(y, score)) if np.any(y) else float("nan"),
        "false_alarm_rate_nl": float(np.mean(pred[y == 0])) if np.any(y == 0) else float("nan"),
    }


def choose_threshold(y: np.ndarray, score: np.ndarray) -> float:
    candidates = np.unique(np.quantile(score, np.linspace(.05, .95, 91)))
    values = [f1_score(y, score >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(values))])


def aggregate_signals(frame: pd.DataFrame, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    d = frame[["signal_id", "leak", "network", "flow", "noise"]].copy()
    d["score"] = score
    agg = d.groupby("signal_id", as_index=False).agg(
        score=("score", "mean"), leak=("leak", "first"), network=("network", "first"),
        flow=("flow", "first"), noise=("noise", "first"))
    return agg.leak.to_numpy(), agg.score.to_numpy(), agg


def bootstrap_ci(agg: pd.DataFrame, metric: str, seed: int, threshold: float = 0.5, n=1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = agg.signal_id.to_numpy()
    vals = []
    for _ in range(n):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        sample = agg.set_index("signal_id").loc[sampled]
        vals.append(binary_metrics(sample.leak.to_numpy(), sample.score.to_numpy(), threshold)[metric])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def evaluate_binary(frame: pd.DataFrame, cols: list[str], variant: str, edge: bool,
                    repeats=(42, 1337, 2027)) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fold_rows: list[dict] = []
    signal_rows: list[pd.DataFrame] = []
    condition_rows: list[dict] = []
    for seed in repeats:
        splitter = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr_idx, te_idx) in enumerate(splitter.split(frame, frame.leak, frame.group)):
            train, test = frame.iloc[tr_idx], frame.iloc[te_idx]
            xtr, xte = transform_train_test(train, test, cols, variant)
            model = model_factory(edge, seed)
            model.fit(xtr, train.leak.to_numpy())
            train_score = model.predict_proba(xtr)[:, 1]
            threshold = choose_threshold(train.leak.to_numpy(), train_score)
            score = model.predict_proba(xte)[:, 1]
            wm = binary_metrics(test.leak.to_numpy(), score)
            wm.update(seed=seed, fold=fold, variant=variant, edge=edge, level="window",
                      n_groups=test.group.nunique(), threshold=threshold)
            fold_rows.append(wm)
            true, sig_score, agg = aggregate_signals(test, score)
            sm = binary_metrics(true, sig_score, threshold)
            sm.update(seed=seed, fold=fold, variant=variant, edge=edge, level="signal",
                      n_groups=agg.signal_id.nunique(), threshold=threshold)
            fold_rows.append(sm)
            signal_rows.append(agg.assign(seed=seed, fold=fold, variant=variant, edge=edge))
            for network, ng in agg.groupby("network"):
                nm = binary_metrics(ng.leak.to_numpy(), ng.score.to_numpy(), threshold)
                nm.update(seed=seed, fold=fold, variant=variant, edge=edge,
                          level="network", network=network)
                fold_rows.append(nm)
            for condition in ("network", "flow", "noise"):
                for value, cg in agg.groupby(condition):
                    cm = binary_metrics(cg.leak.to_numpy(), cg.score.to_numpy(), threshold)
                    cm.update(seed=seed, fold=fold, variant=variant, edge=edge,
                              level="signal_condition", condition=condition, value=value)
                    condition_rows.append(cm)
    return pd.DataFrame(fold_rows), pd.DataFrame(condition_rows), {
        "signals": pd.concat(signal_rows, ignore_index=True),
        "folds": pd.DataFrame(fold_rows),
    }


def evaluate_multiclass(frame: pd.DataFrame, cols: list[str], variant: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    rows, class_rows, cms = [], [], []
    labels = sorted(frame.leak_type.unique())
    for seed in (42, 1337, 2027):
        for fold, (tr_idx, te_idx) in enumerate(GroupKFold(5, shuffle=True, random_state=seed).split(frame, frame.leak_type, frame.group)):
            train, test = frame.iloc[tr_idx], frame.iloc[te_idx]
            xtr, xte = transform_train_test(train, test, cols, variant)
            model = model_factory(False, seed)
            model.fit(xtr, train.leak_type)
            pred = model.predict(xte)
            p, r, f, s = precision_recall_fscore_support(
                test.leak_type, pred, labels=labels, zero_division=0)
            class_rows.extend({"seed": seed, "fold": fold, "variant": variant, "class": label,
                               "precision": p[i], "recall": r[i], "f1": f[i], "support": s[i]}
                              for i, label in enumerate(labels))
            rows.append({"seed": seed, "fold": fold, "variant": variant,
                         "accuracy": accuracy_score(test.leak_type, pred),
                         "balanced_accuracy": balanced_accuracy_score(test.leak_type, pred),
                         "macro_f1": f1_score(test.leak_type, pred, labels=labels, average="macro", zero_division=0),
                         "weighted_f1": f1_score(test.leak_type, pred, average="weighted", zero_division=0)})
            cms.append({"seed": seed, "fold": fold, "matrix": confusion_matrix(
                test.leak_type, pred, labels=labels).tolist(), "labels": labels})
    return pd.DataFrame(rows), pd.DataFrame(class_rows), cms


def baseline_results(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for seed in (42, 1337, 2027):
        for fold, (tr_idx, te_idx) in enumerate(GroupKFold(5, shuffle=True, random_state=seed).split(frame, frame.leak, frame.group)):
            tr, te = frame.iloc[tr_idx], frame.iloc[te_idx]
            majority = int(tr.leak.mean() >= .5)
            rows.append({"baseline": "majority", "seed": seed, "fold": fold,
                         **binary_metrics(te.leak.to_numpy(), np.full(len(te), majority, dtype=float))})
            xtr, xte = transform_train_test(tr, te, cols, "raw")
            m = Pipeline([("scale", StandardScaler()), ("logistic", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))])
            m.fit(xtr, tr.leak)
            threshold = choose_threshold(tr.leak.to_numpy(), m.predict_proba(xtr)[:, 1])
            rows.append({"baseline": "threshold_logistic", "seed": seed, "fold": fold,
                         **binary_metrics(te.leak.to_numpy(), m.predict_proba(xte)[:, 1], threshold)})
    return pd.DataFrame(rows)


def markdown_table(d: pd.DataFrame, cols=None) -> str:
    return d[cols].to_markdown(index=False) if cols else d.to_markdown(index=False)


def run(root: Path) -> None:
    reports = Path("reports"); models = Path("models")
    reports.mkdir(exist_ok=True); (models / "final").mkdir(parents=True, exist_ok=True)
    meta = inventory(root)
    features = {}
    coverage = []
    for window in (.25, .5):
        for hydro in (False, True):
            key = f"{'all' if hydro else 'node'}_{window:.2f}"
            feature_path = reports / f"phase2_{key}_features.csv"
            frame = pd.read_csv(feature_path) if feature_path.exists() else make_features(root, meta, window, hydro)
            features[key] = frame
            frame.to_csv(feature_path, index=False)
            coverage.append({"configuration": key, "signals": frame.signal_id.nunique(),
                             "windows": len(frame), "features": len(feature_columns(frame, hydro))})
    pd.DataFrame(coverage).to_csv(reports / "phase2_coverage.csv", index=False)
    # Primary hardware-matched configuration is the 0.25 s node feature set.
    primary = features["node_0.25"]
    cols = feature_columns(primary, False)
    long_folds, _, long_extra = evaluate_binary(
        features["node_0.50"], feature_columns(features["node_0.50"], False), "raw", False)
    long_folds.to_csv(reports / "phase2_long_window_folds.csv", index=False)
    long_extra["signals"].to_csv(reports / "phase2_long_window_signals.csv", index=False)
    variant_results = {}
    for variant in ("raw", "per_recording", "baseline_relative"):
        folds, conditions, extra = evaluate_binary(primary, cols, variant, False)
        variant_results[variant] = (folds, conditions, extra["signals"])
        folds.to_csv(reports / f"phase2_{variant}_folds.csv", index=False)
        conditions.to_csv(reports / f"phase2_{variant}_conditions.csv", index=False)
        extra["signals"].to_csv(reports / f"phase2_{variant}_signals.csv", index=False)
    baselines = baseline_results(primary, cols)
    baselines.to_csv(reports / "phase2_baselines.csv", index=False)
    # Bonus 5-class evaluation uses the selected preprocessing variant.
    chosen_variant = max(variant_results, key=lambda v: variant_results[v][0].query("level == 'network'").groupby("network").f1.mean().min())
    mc, mc_classes, cms = evaluate_multiclass(primary, cols, chosen_variant)
    mc.to_csv(reports / "phase2_multiclass_folds.csv", index=False)
    mc_classes.to_csv(reports / "phase2_multiclass_per_class.csv", index=False)
    (reports / "phase2_multiclass_confusion.json").write_text(json.dumps(cms, indent=2))
    # Edge/full comparison uses the chosen preprocessing and primary window.
    edge_folds, edge_conditions, edge_extra = evaluate_binary(primary, cols, chosen_variant, True)
    edge_signals = edge_extra["signals"]
    edge_folds.to_csv(reports / "phase2_edge_folds.csv", index=False)
    edge_signals.to_csv(reports / "phase2_edge_signals.csv", index=False)

    full_folds, full_conditions, full_signals = variant_results[chosen_variant]
    all_frame = features["all_0.50"]
    all_cols = feature_columns(all_frame, True)
    all_folds, _, all_extra = evaluate_binary(all_frame, all_cols, "raw", False)
    all_folds.to_csv(reports / "phase2_all_sensor_folds.csv", index=False)
    all_extra["signals"].to_csv(reports / "phase2_all_sensor_signals.csv", index=False)
    full_signal = full_signals.groupby("signal_id", as_index=False).agg(
        score=("score", "mean"), leak=("leak", "first"), network=("network", "first"),
        flow=("flow", "first"), noise=("noise", "first"))
    xtr = primary[cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    model = model_factory(False, SEED)
    # Final fitting uses all available node data; baseline-relative is retained only if selected.
    if chosen_variant == "baseline_relative":
        xtr, _ = transform_train_test(primary, primary, cols, chosen_variant)
    model.fit(xtr, primary.leak)
    model_path = models / "final" / "model.joblib"
    joblib.dump(model, model_path)
    edge_model = model_factory(True, SEED)
    edge_model.fit(xtr, primary.leak)
    edge_path = models / "final" / "edge_model.joblib"
    joblib.dump(edge_model, edge_path)
    size_kb = model_path.stat().st_size / 1024
    sample = xtr[:1]
    start = time.perf_counter()
    for _ in range(1000):
        model.predict_proba(sample)
    latency_ms = (time.perf_counter() - start) * 1000 / 1000
    edge_start = time.perf_counter()
    for _ in range(1000):
        edge_model.predict_proba(sample)
    edge_latency_ms = (time.perf_counter() - edge_start) * 1000 / 1000
    signal_metrics = binary_metrics(full_signal.leak.to_numpy(), full_signal.score.to_numpy())
    network_summary = full_folds.query("level == 'network'").groupby("network").f1.agg(["mean", "min", "max"]).reset_index()
    final_threshold = choose_threshold(primary.leak.to_numpy(), model.predict_proba(xtr)[:, 1])
    signal_cv_metrics = full_folds[full_folds.level == "signal"][
        ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_alarm_rate_nl"]
    ].mean().to_dict()
    signal_ci = {}
    for metric in ("precision", "recall", "f1", "balanced_accuracy", "roc_auc", "pr_auc", "false_alarm_rate_nl"):
        signal_ci[metric] = bootstrap_ci(full_signal, metric, SEED, final_threshold)
    metadata = {
        "feature_names": cols, "preprocessing": {
            "accelerometer_sampling_hz": 8000, "pressure_sampling_hz": 860,
            "hydrophone_sampling_hz": 8000, "anti_alias_filter": "scipy.signal.resample_poly",
            "window_s": .25, "overlap": 0, "band_limits_hz": {"low": [0, 500], "acoustic": [500, 3000]},
            "variant": chosen_variant, "sensor_configuration": "accelerometer+pressure",
        }, "label_map": {"0": "No-leak", "1": "Leak"},
        "decision_threshold": final_threshold, "doi": DOI, "seed": SEED,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": __import__("sklearn").__version__, "model_size_kb": size_kb,
        "inference_ms_per_window": latency_ms, "edge_inference_ms_per_window": edge_latency_ms,
        "cv_signal_metrics": signal_cv_metrics, "final_fit_metrics": signal_metrics,
        "signal_bootstrap_95_ci": signal_ci,
    }
    (models / "final" / "metadata.json").write_text(json.dumps(metadata, indent=2, default=float))
    report = [
        "# AquaPulse Phase 2 final report", "",
        "## Product decision",
        f"Final model: **{chosen_variant} preprocessing + 200-tree random forest**, node-compatible accelerometer + pressure features, 0.25 s windows.",
        f"Selection used the worst held-out-network fold performance, not the mean. The fitted artifact is `{model_path}` ({size_kb:.1f} KB; {latency_ms:.3f} ms/window on the development CPU).",
        "",
        "## Data caveats",
        "The audit found 282 files rather than 280, recordings of roughly 34-61 seconds rather than uniformly 30 seconds, and two standalone hydrophone background files that do not match the signal parser. No position, leak-size, leak-flow, degradation, or RUL labels exist; those tasks remain skipped.",
        "",
        "## Coverage and hardware ablation", markdown_table(pd.DataFrame(coverage)), "",
        "Hydrophone features were present only when both H1/H2 were available. A matched all-sensor ablation is reported separately; the product result below uses only A1/A2/P1/P2.",
        "",
        "### Matched all-sensor ablation",
        markdown_table(all_folds[all_folds.level == "signal"].groupby("level", as_index=False).agg(
            accuracy=("accuracy", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
            precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"),
            roc_auc=("roc_auc", "mean"), pr_auc=("pr_auc", "mean"),
            false_alarm_rate_nl=("false_alarm_rate_nl", "mean"))),
        "",
        "### Window-length comparison",
        markdown_table(pd.DataFrame([
            {"window_s": 0.25, "signal_f1": full_folds.query("level == 'signal'").f1.mean(),
             "signal_balanced_accuracy": full_folds.query("level == 'signal'").balanced_accuracy.mean(),
             "signal_false_alarm_rate_nl": full_folds.query("level == 'signal'").false_alarm_rate_nl.mean()},
            {"window_s": 0.50, "signal_f1": long_folds.query("level == 'signal'").f1.mean(),
             "signal_balanced_accuracy": long_folds.query("level == 'signal'").balanced_accuracy.mean(),
             "signal_false_alarm_rate_nl": long_folds.query("level == 'signal'").false_alarm_rate_nl.mean()},
        ])),
        "",
        "## Primary binary detector: repeated grouped CV",
        "### Signal-level evidence",
        markdown_table(full_folds[full_folds.level == "signal"].groupby("level", as_index=False).agg(
            accuracy=("accuracy", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
            precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"),
            roc_auc=("roc_auc", "mean"), pr_auc=("pr_auc", "mean"), false_alarm_rate_nl=("false_alarm_rate_nl", "mean"))),
        "",
        "Signal-level bootstrap 95% CIs are in `models/final/metadata.json`. The final deployment threshold is fitted from all available training recordings and is stored there.",
        "### Window-level diagnostics",
        markdown_table(full_folds.groupby("level", as_index=False).agg(
            accuracy=("accuracy", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
            precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"),
            roc_auc=("roc_auc", "mean"), pr_auc=("pr_auc", "mean"), false_alarm_rate_nl=("false_alarm_rate_nl", "mean"))),
        "",
        "Fold-level results are repeated shuffled 5-fold GroupKFold across three fixed seeds. A signal-level decision averages window probabilities within each recording. Group-bootstrap 95% CIs (resampling complete signals) are stored in metadata.json; no random windows from one recording cross a fold.",
        "",
        "## Baselines", markdown_table(baselines.groupby("baseline", as_index=False).agg(
            accuracy=("accuracy", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
            precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"),
            false_alarm_rate_nl=("false_alarm_rate_nl", "mean"))), "",
        "## Preprocessing ablation",
        markdown_table(pd.DataFrame([{"variant": v, "worst_network_f1": variant_results[v][0].query("level == 'network'").groupby("network").f1.mean().min(),
                                      "mean_f1": variant_results[v][0].f1.mean(),
                                      "signal_f1": variant_results[v][2].groupby("signal_id").first().pipe(lambda x: f1_score(x.leak, x.score >= .5, zero_division=0))}
                                     for v in variant_results])), "",
        "## Leave-one-condition false alarms",
        markdown_table(full_conditions[full_conditions.level == "signal_condition"].groupby(["condition", "value"], as_index=False).agg(
            false_alarm_rate_nl=("false_alarm_rate_nl", "mean"), recall=("recall", "mean"), f1=("f1", "mean"))), "",
        "Transient is shown separately because valve events must not be treated as leaks.",
        "",
        "## Five-class bonus",
        markdown_table(mc.groupby("variant", as_index=False).agg(
            accuracy=("accuracy", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
            macro_f1=("macro_f1", "mean"), weighted_f1=("weighted_f1", "mean"))), "",
        "Per-class precision/recall/F1/support are in `phase2_multiclass_per_class.csv`; every fold confusion matrix is in `phase2_multiclass_confusion.json`. Macro-F1 is computed over all five explicit labels; absent labels receive zero, and folds are checked for class coverage.",
        "",
        "## Edge-sized comparison",
        markdown_table(pd.DataFrame([{"model": "full_random_forest", "size_kb": size_kb, "latency_ms": latency_ms, "mean_f1": full_folds.query("level == 'signal'").f1.mean()},
                                     {"model": "edge_random_forest", "size_kb": edge_path.stat().st_size / 1024,
                                      "latency_ms": edge_latency_ms, "mean_f1": edge_folds.query("level == 'signal'").f1.mean()}])), "",
        "Latency is measured on the development CPU; ESP32-S3 latency still requires target-hardware benchmarking.",
        "",
        "## Failure analysis and limitations",
        "The model remains vulnerable to installation/network shift: the network-held-out scores are the selection criterion and are substantially worse than pooled scores. Window-level accuracy overstates practical reliability because adjacent windows are correlated; signal-level metrics are the product evidence. The lab PVC testbed, sensor mounting, background flow conditions, and hydrophone coverage do not represent buried field pipes or the final piezo/pressure electronics. This is training/evaluation only; no firmware, TinyML conversion, dashboard, or fabricated regression/RUL target is included.",
    ]
    (reports / "final_report.md").write_text("\n".join(report))
    print(json.dumps({"chosen_variant": chosen_variant, "signal_metrics": signal_metrics,
                      "model_size_kb": size_kb, "latency_ms": latency_ms}, indent=2, default=float))


def main() -> None:
    run(Path("data"))


if __name__ == "__main__":
    main()
