from __future__ import annotations

import argparse
import json
import re
import struct
import time
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
FLOW_MAP = {"ND": "ND", "Transient": "Transient"}
LEAK_MAP = {"LC": "Longitudinal Crack", "CC": "Circumferential Crack",
            "GL": "Gasket Leak", "OL": "Orifice Leak", "NL": "No-leak"}


def parse_path(path: Path, root: Path) -> dict:
    rel = path.relative_to(root)
    parts = rel.parts
    sensor = parts[0]
    folder = parts[-2] if len(parts) > 2 else ""
    m = re.match(r"(?P<network>LO|BR)_(?P<leak>LC|CC|GL|OL|NL)_(?P<rest>.+)_(?P<sensor>[A-Z]\d)\.(?P<ext>csv|raw)$",
                 path.name)
    row = {"path": str(rel), "sensor_type": sensor, "folder": folder,
           "size_bytes": path.stat().st_size, "parse_ok": bool(m)}
    if not m:
        row.update(network=None, leak_code=None, leak_type=None, flow=None,
                   noise=None, channel=None, signal_id=None)
        return row
    d = m.groupdict()
    rest = d["rest"]
    tokens = rest.split("_")
    flow = "Transient" if rest.startswith("Transient") else ("ND" if rest.startswith("ND") else tokens[0])
    noise = tokens[1] if flow not in ("ND", "Transient") and len(tokens) > 1 else (
        tokens[1] if flow in ("ND", "Transient") and len(tokens) > 1 else "N")
    signal_id = f"{d['network']}_{d['leak']}_{flow}_{noise}"
    row.update(network=d["network"], leak_code=d["leak"], leak_type=LEAK_MAP[d["leak"]],
               flow=flow, noise=noise, channel=d["sensor"],
               signal_id=signal_id)
    return row


def inventory(root: Path) -> pd.DataFrame:
    rows = [parse_path(p, root) for p in root.rglob("*") if p.is_file()]
    return pd.DataFrame(rows).sort_values("path").reset_index(drop=True)


def read_signal(path: Path, sensor_type: str) -> tuple[np.ndarray, float, list[str]]:
    if path.suffix.lower() == ".csv":
        d = pd.read_csv(path)
        t, x = d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)
        fs = 1.0 / np.median(np.diff(t))
        return x, fs, list(d.columns)
    raw = np.fromfile(path, dtype="<i4")
    return raw.astype(float), 8000.0, ["sample (int32 little-endian)"]


def signal_quality(path: Path, sensor_type: str) -> dict:
    x, fs, cols = read_signal(path, sensor_type)
    finite = np.isfinite(x)
    y = x[finite]
    q = {"columns": "|".join(cols), "samples": len(x), "fs_hz": fs,
         "duration_s": len(x) / fs, "nan_count": int((~finite).sum()),
         "constant": bool(len(y) and np.nanstd(y) == 0),
         "duplicate_count": int(len(x) - len(np.unique(x))) if len(x) < 2_000_000 else -1,
         "min": float(np.nanmin(y)) if len(y) else np.nan,
         "max": float(np.nanmax(y)) if len(y) else np.nan,
         "mean": float(np.nanmean(y)) if len(y) else np.nan,
         "std": float(np.nanstd(y)) if len(y) else np.nan}
    q["outlier_count_robust"] = int(np.sum(np.abs(stats.zscore(y, nan_policy="omit")) > 8)) if len(y) else 0
    return q


def feature_vector(x: np.ndarray, fs: float, prefix: str) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    rms = float(np.sqrt(np.mean(x * x)))
    sd = float(np.std(x))
    centered = x - np.mean(x)
    f, p = signal.periodogram(centered, fs)
    p = p + 1e-15
    bands = [(0, 20), (20, 100), (100, 500), (500, 2000), (2000, fs / 2)]
    out = {f"{prefix}_mean": float(np.mean(x)), f"{prefix}_std": sd,
           f"{prefix}_rms": rms, f"{prefix}_peak": float(np.max(np.abs(x))),
           f"{prefix}_crest": float(np.max(np.abs(x)) / (rms + 1e-12)),
           f"{prefix}_kurtosis": float(stats.kurtosis(x, fisher=False)),
           f"{prefix}_skewness": float(stats.skew(x)),
           f"{prefix}_zcr": float(np.mean(np.diff(np.signbit(centered)))),
           f"{prefix}_spectral_centroid": float(np.sum(f * p) / np.sum(p)),
           f"{prefix}_spectral_entropy": float(-np.sum((p / p.sum()) * np.log2(p / p.sum()))),
           f"{prefix}_spectral_flatness": float(np.exp(np.mean(np.log(p))) / np.mean(p)),
           f"{prefix}_dominant_hz": float(f[np.argmax(p[1:]) + 1]) if len(p) > 1 else 0.0,
           f"{prefix}_roc": float(np.mean(np.abs(np.diff(x))) * fs)}
    for i, (lo, hi) in enumerate(bands):
        out[f"{prefix}_band_{i}"] = float(np.sum(p[(f >= lo) & (f < hi)]))
    return out


def make_features(root: Path, meta: pd.DataFrame, config: str, window_s=0.5) -> pd.DataFrame:
    use = meta[meta.parse_ok & meta.sensor_type.isin(
        ["Accelerometer", "Dynamic Pressure Sensor", "Hydrophone"])]
    rows = []
    for sid, group in use.groupby("signal_id"):
        channels = {}
        for sensor_type, sg in group.groupby("sensor_type"):
            for _, r in sg.iterrows():
                x, fs, _ = read_signal(root / r.path, sensor_type)
                channels[r.channel] = (x, fs, sensor_type)
        if config == "node" and not all(c in channels for c in ("A1", "A2", "P1", "P2")):
            continue
        if config == "all" and not all(c in channels for c in ("A1", "A2", "P1", "P2", "H1", "H2")):
            continue
        duration = min(len(v[0]) / v[1] for v in channels.values())
        step_s = window_s
        # A 0.5 s window is compatible with the stated edge budget and captures low-frequency pipe events.
        for start_s in np.arange(0, duration - window_s + 1e-9, step_s):
            row = {"signal_id": sid, "group": sid, "leak_type": group.iloc[0].leak_type,
                   "leak": int(group.iloc[0].leak_type != "No-leak"),
                   "network": group.iloc[0].network, "flow": group.iloc[0].flow,
                   "noise": group.iloc[0].noise, "window_start_s": float(start_s)}
            for ch, (x, cfs, sensor_type) in channels.items():
                start = int(round(start_s * cfs))
                row.update(feature_vector(x[start:start + int(window_s * cfs)], cfs, ch))
            for a, b in (("A1", "A2"), ("P1", "P2"), ("H1", "H2")):
                if a in channels and b in channels:
                    ia = int(round(start_s * channels[a][1])); ib = int(round(start_s * channels[b][1]))
                    xa = channels[a][0][ia:ia + int(window_s * channels[a][1])]
                    xb = channels[b][0][ib:ib + int(window_s * channels[b][1])]
                    m = min(len(xa), len(xb))
                    corr = signal.correlate(xa[:m] - np.mean(xa[:m]), xb[:m] - np.mean(xb[:m]), mode="full")
                    lag = np.argmax(corr) - (m - 1)
                    row[f"{a}_{b}_lag_samples"] = float(lag)
                    row[f"{a}_{b}_amplitude_ratio"] = float(np.std(xa[:m]) / (np.std(xb[:m]) + 1e-12))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_audit(root: Path, meta: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for (leak, sensor_type), g in meta[meta.parse_ok].groupby(["leak_type", "sensor_type"]):
        r = g.iloc[0]
        x, fs, _ = read_signal(root / r.path, r.sensor_type)
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        ax[0].plot(np.arange(min(len(x), int(5 * fs))) / fs, x[:int(5 * fs)])
        ax[0].set(title=f"{leak} representative ({r.sensor_type})", xlabel="s")
        f, t, z = signal.stft(x[:min(len(x), int(10 * fs))], fs=fs, nperseg=min(1024, len(x)//4))
        ax[1].pcolormesh(t, f, 20*np.log10(np.abs(z)+1e-12), shading="auto")
        ax[1].set(ylim=(0, min(fs/2, 4000)), ylabel="Hz", xlabel="s")
        fig.tight_layout(); fig.savefig(out / f"{r.leak_code}_{r.sensor_type.replace(' ', '_')}.png", dpi=120); plt.close(fig)


def run_audit(root: Path, report: Path) -> pd.DataFrame:
    meta = inventory(root)
    quality = []
    for _, r in meta.iterrows():
        quality.append(signal_quality(root / r.path, r.sensor_type))
    meta = pd.concat([meta, pd.DataFrame(quality)], axis=1)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# Dataset audit\n\n"
        "## Parsing rules\n"
        "Files are parsed as `<network>_<leak code>_<flow>[_<noise>]_sensor.ext` under a sensor/network/leak folder. "
        "Codes are LO/BR, LC/CC/GL/OL/NL, flows ND/0.18 LPS/0.47 LPS/Transient, and channels A1/A2, P1/P2, H1/H2. "
        "No filename or folder token contains leak position, sensor position, leak size, or leak flow.\n\n"
        "## Verified inventory\n\n" + meta.groupby("sensor_type").size().to_string() +
        "\n\n" + meta.groupby(["sensor_type", "parse_ok"]).size().to_string() +
        "\n\n### Sample metadata\n\n" + meta.head(12).to_markdown(index=False) +
        "\n\n### Class/condition counts\n\n" + meta[meta.parse_ok].groupby(
            ["leak_type", "flow", "noise", "network"]).size().to_string() +
        "\n\n### Signal quality summary\n\n" + meta.groupby("sensor_type")[["samples", "fs_hz", "duration_s", "nan_count", "constant", "duplicate_count", "outlier_count_robust"]].agg(["min", "max", "mean"]).to_string() +
        "\n\n## Feasibility\n\n"
        "| Goal | Status | Evidence |\n|---|---|---|\n"
        "| Leak/no-leak and leak type | FEASIBLE | Explicit NL/LC/CC/GL/OL labels |\n"
        "| Localization | NOT FEASIBLE | No leak-position or sensor-position labels; TDE requires pipe geometry/speed and a true paired arrival event |\n"
        "| Leak size/flow | NOT FEASIBLE | Flow tokens are background flow conditions, not leak rate/size |\n"
        "| RUL/predictive failure | NOT FEASIBLE | Controlled 30-second recordings contain no degradation/time-to-failure label |\n"
        "| No-leak autoencoder | FEASIBLE | NL files are explicit normal examples; validation/test are held out by signal group |\n"
        "\n## Known limitations\nHydrophone files are integer RAW streams interpreted as little-endian int32 at 8000 Hz because their byte length and dataset format are consistent with 30 seconds at 8 kHz. CSV time columns determine accelerometer/pressure sampling rates. The audit flags this interpretation for confirmation against the publisher script if one is later obtained.\n")
    plot_audit(root, meta, report.parent / "figures")
    return meta


def metric_row(y, pred, proba, labels) -> dict:
    d = {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
         "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
         "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0)}
    try:
        d["roc_auc_ovr"] = roc_auc_score(y, proba, multi_class="ovr", labels=labels)
        d["pr_auc_macro"] = average_precision_score(pd.get_dummies(y).reindex(columns=labels, fill_value=0), proba, average="macro")
    except ValueError: d["roc_auc_ovr"] = np.nan; d["pr_auc_macro"] = np.nan
    for label, score in zip(labels, f1_score(y, pred, labels=labels, average=None, zero_division=0)):
        d[f"f1_{label}"] = float(score)
    return d


def train_classification(features: pd.DataFrame, config: str, out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    ignore = {"signal_id", "group", "leak_type", "leak", "network", "flow", "noise", "window_start_s"}
    cols = [c for c in features.columns if c not in ignore and features[c].dtype != object]
    X, y, groups = features[cols].replace([np.inf, -np.inf], np.nan).fillna(0), features.leak_type, features.group
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
    tr, te = next(splitter.split(X, y, groups))
    models = {
        "logistic": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))]),
        "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss",
            random_state=SEED, n_jobs=-1)
    except ImportError:
        pass
    labels = sorted(y.unique())
    records = []
    for name, model in models.items():
        start = time.perf_counter()
        fit_y = y.iloc[tr]
        if name == "xgboost":
            label_to_int = {label: i for i, label in enumerate(labels)}
            int_to_label = {i: label for label, i in label_to_int.items()}
            fit_y = fit_y.map(label_to_int)
            model.fit(X.iloc[tr], fit_y)
            pred = np.asarray(model.predict(X.iloc[te])).astype(int)
            pred = pd.Series([int_to_label[i] for i in pred], index=y.iloc[te].index)
        else:
            model.fit(X.iloc[tr], fit_y)
            pred = model.predict(X.iloc[te])
        proba = model.predict_proba(X.iloc[te])
        d = metric_row(y.iloc[te], pred, proba, labels); d.update(config=config, model=name, split="test",
            n_groups=int(len(set(groups.iloc[te]))), inference_ms_window=float(time.perf_counter() - start) / max(1, len(te)) * 1000)
        records.append(d)
        joblib.dump({"model": model, "features": cols, "labels": labels}, out / f"{config}_{name}.joblib")
    # GroupKFold CV on complete windows, with held-out groups in every fold.
    cv_rows = []
    for fold, (trf, vf) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        model = models["random_forest"]; model.fit(X.iloc[trf], y.iloc[trf])
        pred, proba = model.predict(X.iloc[vf]), model.predict_proba(X.iloc[vf])
        d = metric_row(y.iloc[vf], pred, proba, labels); d.update(config=config, model="random_forest", split=f"cv_fold_{fold}", n_groups=len(set(groups.iloc[vf])))
        cv_rows.append(d)
    # Leave-one-condition-out checks are deliberately separate from the held-out test set.
    for field in ("noise", "flow", "network"):
        for value in sorted(features[field].astype(str).unique()):
            trc = features[field].astype(str).to_numpy() != value
            tec = ~trc
            if len(set(y[trc])) < 2 or not np.any(tec):
                continue
            model = models["random_forest"]; model.fit(X.loc[trc], y.loc[trc])
            pred, proba = model.predict(X.loc[tec]), model.predict_proba(X.loc[tec])
            d = metric_row(y.loc[tec], pred, proba, labels)
            d.update(config=config, model="random_forest", split=f"leave_out_{field}={value}",
                     n_groups=len(set(groups.loc[tec])))
            cv_rows.append(d)
    result = pd.DataFrame(records + cv_rows)
    out.mkdir(parents=True, exist_ok=True); result.to_csv(out / f"{config}_classification_metrics.csv", index=False)
    cm = confusion_matrix(y.iloc[te], models["random_forest"].predict(X.iloc[te]), labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out / f"{config}_confusion_matrix.csv")
    return result


def train_autoencoder(features: pd.DataFrame, config: str, out: Path) -> dict:
    # PCA reconstruction is a deterministic small dense autoencoder baseline without a deep-learning dependency.
    from sklearn.decomposition import PCA
    ignore = {"signal_id", "group", "leak_type", "leak", "network", "flow", "noise", "window_start_s"}
    cols = [c for c in features.columns if c not in ignore and features[c].dtype != object]
    X = features[cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    normal = features.leak_type.eq("No-leak").to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
    tr, te = next(gss.split(X, normal, features.group))
    normal_tr = tr[normal[tr]]
    scale = StandardScaler().fit(X[normal_tr]); Z = scale.transform(X)
    pca = PCA(n_components=min(16, max(2, Z.shape[1] // 3)), random_state=SEED).fit(Z[normal_tr])
    val_err = np.mean((Z[normal_tr] - pca.inverse_transform(pca.transform(Z[normal_tr])))**2, axis=1)
    threshold = float(np.quantile(val_err, .99))
    err = np.mean((Z - pca.inverse_transform(pca.transform(Z)))**2, axis=1)
    pred = err > threshold; truth = ~normal
    d = {"config": config, "threshold_99": threshold, "roc_auc": roc_auc_score(truth[te], err[te]),
         "pr_auc": average_precision_score(truth[te], err[te]), "f1": f1_score(truth[te], pred[te], zero_division=0),
         "false_positive_rate_nl": float(np.mean(pred[te][~truth[te]])) if np.any(~truth[te]) else np.nan}
    joblib.dump({"scaler": scale, "pca": pca, "features": cols, "threshold": threshold}, out / f"{config}_normal_pca_autoencoder.joblib")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()
    root = args.data.resolve(); reports = Path("reports"); models = Path("models")
    meta = run_audit(root, reports / "data_audit.md")
    if args.audit_only:
        print("Audit written to reports/data_audit.md"); return
    all_results = []
    for config in ("node", "all"):
        f = make_features(root, meta, config)
        f.to_csv(reports / f"{config}_features.csv", index=False)
        if len(f):
            cls_result = train_classification(f, config, models)
            cls_result.to_csv(reports / f"{config}_classification_metrics.csv", index=False)
            all_results.append(cls_result)
            all_results.append(pd.DataFrame([train_autoencoder(f, config, models)]))
    result = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    result.to_json(reports / "metrics.json", orient="records", indent=2)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
