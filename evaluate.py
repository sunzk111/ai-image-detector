from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# These are predeclared diagnostic comparisons, NOT thresholds fitted on test labels.
DEFAULT_THRESHOLDS = (0.5, 0.3, 0.1, 0.01, 0.001,
                      1e-4, 1e-5, 5e-6, 1e-6, 1e-8, 1e-10,)
METRIC_FIELDS = ["accuracy", "auroc", "f1", "count", "threshold", "auroc_logits",
                 "tp", "tn", "fp", "fn", "ai_recall", "real_recall", "precision",
                 "false_positive_rate", "balanced_accuracy"]
REPORT_FIELDS = ["condition", "operation", "value", *METRIC_FIELDS]
PREDICTION_FIELDS = ["condition", "operation", "value", "sample_id", "path", "generator",
                     "label", "logit", "probability", "threshold", "prediction", "error_type"]
QUANTILES = {"min": 0, "p01": .01, "p05": .05, "p25": .25, "median": .5,
             "p75": .75, "p95": .95, "p99": .99, "max": 1}


def checked_threshold(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Threshold must be finite and in [0, 1]")
    return value


def comparison_thresholds(active, requested=None):
    values = DEFAULT_THRESHOLDS if requested is None else requested
    return sorted({checked_threshold(active), *(checked_threshold(v) for v in values)}, reverse=True)


def divide(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def summarize_scores(labels, logits, probabilities, threshold, thresholds):
    y = np.asarray(labels)
    z = np.asarray(logits, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected a nonempty vector of binary labels")
    if z.shape != y.shape or p.shape != y.shape:
        raise ValueError("Labels, logits and probabilities must have matching shapes")
    if not np.isfinite(z).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Nonfinite logits/probabilities or probabilities outside [0, 1]")
    y = y.astype(np.int64)
    both_classes = np.unique(y).size == 2
    auc = float(roc_auc_score(y, p)) if both_classes else None
    auc_logits = float(roc_auc_score(y, z)) if both_classes else None
    rows = []
    for cutoff in comparison_thresholds(threshold, thresholds):
        pred = p >= cutoff  # Same >= decision rule as the existing metrics.py.
        positive = y == 1
        tp, fp = int((pred & positive).sum()), int((pred & ~positive).sum())
        fn, tn = int((~pred & positive).sum()), int((~pred & ~positive).sum())
        recall, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
        rows.append({
            "accuracy": (tp + tn) / len(y), "auroc": auc,
            "f1": divide(2 * tp, 2 * tp + fp + fn) or 0.0, "count": len(y),
            "threshold": cutoff, "auroc_logits": auc_logits,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "ai_recall": recall, "real_recall": specificity, "precision": divide(tp, tp + fp),
            "false_positive_rate": divide(fp, fp + tn),
            "balanced_accuracy": (recall + specificity) / 2 if both_classes else None,
        })
    distributions = []
    for label, name in ((0, "real"), (1, "ai")):
        mask = y == label
        row = {"label": label, "class_name": name, "count": int(mask.sum())}
        for prefix, values in (("logit", z[mask]), ("probability", p[mask])):
            for key, quantile in QUANTILES.items():
                row[f"{prefix}_{key}"] = float(np.quantile(values, quantile)) if len(values) else None
        row["probability_zero_count"] = int((p[mask] == 0).sum())
        row["probability_one_count"] = int((p[mask] == 1).sum())
        distributions.append(row)
    active = next(row for row in rows if row["threshold"] == float(threshold))
    return active, rows, distributions


def condition_grid(config: dict[str, Any]) -> list[tuple[str, str, float | int | None]]:
    a = config["augmentation"]
    conditions = [("clean", "clean", None)]
    conditions.extend((f"jpeg_q{q}", "jpeg", q) for q in a["jpeg_qualities"])
    conditions.extend((f"blur_sigma{s:g}", "blur", s) for s in a["blur_sigmas"])
    conditions.extend((f"resize_{s:g}x", "resize", s) for s in a["resize_scales"])
    conditions.extend((f"noise_sigma{s:g}", "noise", s) for s in a["noise_sigmas"])
    jitter = float(a["color_jitter"])
    conditions.extend([(f"color_jitter_minus{jitter:g}", "color_jitter_minus", jitter),
                       (f"color_jitter_plus{jitter:g}", "color_jitter_plus", jitter)])
    crop = float(a["center_crop_fraction"])
    conditions.append((f"center_crop_{crop:g}", "crop", crop))
    return conditions


def make_loader(dataset, config, seed):
    from utils import seed_worker
    return DataLoader(dataset, batch_size=int(config["evaluation"]["batch_size"]),
                      shuffle=False, num_workers=int(config["data"].get("num_workers", 4)),
                      pin_memory=bool(config["data"].get("pin_memory", True)) and torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed), drop_last=False)


def sample_path(dataset, index):
    while isinstance(dataset, Subset):
        index, dataset = int(dataset.indices[index]), dataset.dataset
    samples = getattr(dataset, "samples", None)
    # HF images may not have a local file; keep sample_id and leave path empty.
    return str(samples[index].path) if samples is not None else ""


@torch.inference_mode()
def score_condition(model, loader, device, precision, threshold, description,
                    *, prediction_writer=None, operation="clean", value=None, thresholds=None):
    labels, logits_all, probabilities = [], [], []
    offset = 0
    for batch in tqdm(loader, desc=description, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=precision.dtype,
                            enabled=precision.autocast_enabled):
            logits, _ = model(images)
        # Cast BEFORE sigmoid. This avoids extra BF16 sigmoid rounding, but does
        # not pretend that a BF16 model forward has full FP32 inference precision.
        scores = logits.detach().float().reshape(-1)
        probs = torch.sigmoid(scores).cpu().tolist()
        scores = scores.cpu().tolist()
        targets = batch["label"].reshape(-1).cpu().tolist()
        if len(scores) != len(targets):
            raise ValueError("Expected one model logit per image")
        if not all(math.isfinite(z) for z in scores) or not all(y in (0, 1) for y in targets):
            raise ValueError("Nonfinite model logits or invalid labels")
        ids = batch.get("sample_id", [str(offset + i) for i in range(len(targets))])
        generators = batch.get("generator", ["unknown"] * len(targets))
        for i, (y, z, p) in enumerate(zip(targets, scores, probs)):
            predicted = int(p >= threshold)
            if prediction_writer is not None:
                prediction_writer.writerow({
                    "condition": description, "operation": operation, "value": value,
                    "sample_id": str(ids[i]), "path": sample_path(loader.dataset, offset + i),
                    "generator": str(generators[i]), "label": int(y), "logit": z,
                    "probability": p, "threshold": threshold, "prediction": predicted,
                    "error_type": "correct" if predicted == int(y) else ("false_negative" if y == 1 else "false_positive"),
                })
        labels.extend(targets)
        logits_all.extend(scores)
        probabilities.extend(probs)
        offset += len(targets)
    active, sweep, distributions = summarize_scores(labels, logits_all, probabilities, threshold, thresholds)
    return {**active, "_thresholds": sweep, "_distributions": distributions}


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def new_output_dir(parent, requested=None):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(requested).expanduser().resolve() if requested else parent / stamp
    # Fail closed instead of overwriting an earlier diagnostic run.
    path.mkdir(parents=True, exist_ok=False)
    return path


def add_condition(results, sweep_rows, distributions, condition, metrics):
    display_name, operation, value = condition
    context = {"condition": display_name, "operation": operation, "value": value}
    sweep_rows.extend({**context, **row} for row in metrics.pop("_thresholds"))
    distributions.extend({**context, **row} for row in metrics.pop("_distributions"))
    results.append({**context, **metrics})
    print(f"{display_name}: accuracy={metrics['accuracy']:.4f}, "
          f"AI recall={metrics['ai_recall']}, FP={metrics['fp']}, FN={metrics['fn']}, "
          f"AUROC={metrics['auroc']}, logit AUROC={metrics['auroc_logits']}", flush=True)


def save_reports(output_dir, results, sweep, distributions, metadata):
    transformed = [r for r in results if r["condition"] != "clean"]
    clean = next((r for r in results if r["condition"] == "clean"), None)
    worst = min(transformed, key=lambda r: r["accuracy"]) if transformed else None
    summary = {**metadata, "status": "complete", "label_mapping": {"real": 0, "ai": 1},
               "decision_rule": "probability >= threshold", "thresholds_are_diagnostic_only": True,
               "automatically_selected_threshold": None,
               "auroc_definition": "auroc uses FP32-sigmoid probabilities; auroc_logits uses raw model logits to avoid sigmoid saturation ties",
               "clean_accuracy": clean["accuracy"] if clean else None,
               "mean_transformed_accuracy": sum(r["accuracy"] for r in transformed) / len(transformed) if transformed else None,
               "worst_condition": worst["condition"] if worst else None,
               "worst_condition_accuracy": worst["accuracy"] if worst else None,
               "conditions": results}
    write_csv(output_dir / "robustness.csv", results, REPORT_FIELDS)
    write_csv(output_dir / "threshold_sweep.csv", sweep, REPORT_FIELDS)
    write_csv(output_dir / "score_distributions.csv", distributions, list(distributions[0]))
    (output_dir / "robustness.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Reports saved: {output_dir}", flush=True)
    print("Threshold comparisons do NOT change best.pt or select a threshold from test labels.", flush=True)


def main(config_path, checkpoint_path, source_name, *, threshold_override=None,
         thresholds=None, selected_conditions=None, max_samples=None, output_dir=None):
    # Lazy project imports also allow --from-predictions without loading a model.
    from augmentations import ImagePreprocessor, deterministic_condition
    from dataset import build_dataset
    from model import DinoBinaryClassifier
    from threshold_calibration import resolve_evaluation_threshold
    from utils import configure_runtime, get_device, load_config, resolve_precision, runtime_summary, seed_everything

    config = load_config(config_path)
    if source_name not in config["data"]:
        raise KeyError(f"data.{source_name} is not defined in {config_path}")
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    all_conditions = condition_grid(config)
    unknown = set(selected_conditions or ()) - {c[0] for c in all_conditions}
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    runtime_config = config.get("runtime", {})
    device = get_device(runtime_config.get("device", "auto"))
    configure_runtime(runtime_config, device)
    precision = resolve_precision(runtime_config, device)
    runtime = runtime_summary(device, precision)
    print(f"Runtime: {runtime}", flush=True)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("config", config)["model"]
    threshold = (checked_threshold(threshold_override) if threshold_override is not None
                 else resolve_evaluation_threshold(config["evaluation"], checkpoint))
    cutoffs = comparison_thresholds(threshold, thresholds)
    threshold_source = "command_line" if threshold_override is not None else config["evaluation"].get("threshold_source", "config")
    print(f"Decision threshold: {threshold:.6f} (source={threshold_source})", flush=True)
    print(f"Diagnostic thresholds: {cutoffs}", flush=True)
    model = DinoBinaryClassifier(model_config)
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    model.to(device).eval()
    preprocessor = ImagePreprocessor(model_config["image_size"], model_config["image_mean"], model_config["image_std"])
    run_dir = new_output_dir(checkpoint_path.parent / "evaluation_runs", output_dir)
    metadata = {"checkpoint": str(checkpoint_path), "source": source_name,
                "threshold": threshold, "threshold_source": threshold_source,
                "thresholds": cutoffs, "runtime": runtime, "max_samples": max_samples,
                "prediction_file": str(run_dir / "predictions.csv"),
                "checkpoint_size": checkpoint_path.stat().st_size,
                "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns}
    (run_dir / "run_status.json").write_text(json.dumps({**metadata, "status": "running"}, indent=2), encoding="utf-8")
    results, sweep_rows, distributions = [], [], []
    try:
        with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            for index, condition in enumerate(all_conditions):
                name, operation, value = condition
                if selected_conditions and name not in selected_conditions:
                    continue
                # Preserve original condition indices and seeds, even for a subset.
                dataset = build_dataset(config["data"][source_name], config["data"], preprocessor,
                                        evaluation_transform=deterministic_condition(operation, value, seed + index))
                if max_samples is not None and max_samples < len(dataset):
                    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:max_samples].sort().values.tolist()
                    dataset = Subset(dataset, indices)
                metrics = score_condition(model, make_loader(dataset, config, seed), device, precision,
                                          threshold, name, prediction_writer=writer, operation=operation,
                                          value=value, thresholds=cutoffs)
                handle.flush()
                add_condition(results, sweep_rows, distributions, condition, metrics)
        save_reports(run_dir, results, sweep_rows, distributions, metadata)
        # Keep existing full-run consumers working; smoke/subset runs never replace them.
        if output_dir is None and selected_conditions is None and max_samples is None:
            for key, default in (("output_csv", "robustness.csv"), ("output_json", "robustness.json")):
                destination = checkpoint_path.parent / config["evaluation"].get(key, default)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(run_dir / default, destination)
        status = {**metadata, "status": "complete", "condition_count": len(results)}
    except Exception as exc:
        (run_dir / "run_status.json").write_text(json.dumps({**metadata, "status": "failed", "error": str(exc)}, indent=2), encoding="utf-8")
        raise
    (run_dir / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def rescore_predictions(path, threshold_override=None, thresholds=None, output_dir=None):
    path = Path(path).expanduser().resolve()
    status_path = path.parent / "run_status.json"
    if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") != "complete":
        raise ValueError("Prediction run is incomplete; do not rescore a partial CSV")
    results, sweep_rows, distributions, seen = [], [], [], set()
    active_threshold = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"condition", "operation", "value", "label", "logit", "probability", "threshold"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Input must be predictions.csv, not an aggregated robustness.csv")
        for name, group in itertools.groupby(reader, key=lambda row: row["condition"]):
            if name in seen:
                raise ValueError("Prediction rows must be grouped by condition")
            seen.add(name)
            rows = list(group)
            saved_thresholds = {checked_threshold(r["threshold"]) for r in rows}
            if len(saved_thresholds) != 1:
                raise ValueError("Mixed source thresholds in prediction CSV")
            cutoff = checked_threshold(threshold_override) if threshold_override is not None else saved_thresholds.pop()
            if active_threshold is not None and cutoff != active_threshold:
                raise ValueError("Conditions must use the same active threshold")
            active_threshold = cutoff
            active, sweep, dist = summarize_scores([float(r["label"]) for r in rows],
                                                   [float(r["logit"]) for r in rows],
                                                   [float(r["probability"]) for r in rows], cutoff, thresholds)
            value = json.loads(rows[0]["value"]) if rows[0]["value"] else None
            add_condition(results, sweep_rows, distributions, (name, rows[0]["operation"], value),
                          {**active, "_thresholds": sweep, "_distributions": dist})
    if not results:
        raise ValueError("Prediction CSV is empty")
    run_dir = new_output_dir(path.parent / "rescored", output_dir)
    save_reports(run_dir, results, sweep_rows, distributions,
                 {"prediction_file": str(path), "threshold": active_threshold,
                  "threshold_source": "command_line" if threshold_override is not None else "saved_predictions",
                  "thresholds": comparison_thresholds(active_threshold, thresholds),
                  "inference_performed": False})
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate robustness, export scores, and compare fixed thresholds")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--source", default="val", help="Dataset key under config data")
    parser.add_argument("--threshold", type=float, help="Report cutoff override; never modifies the checkpoint")
    parser.add_argument("--thresholds", type=float, nargs="+", help="Predeclared diagnostic cutoffs; active cutoff is always included")
    parser.add_argument("--conditions", nargs="+", help="Condition names, e.g. clean jpeg_q30 blur_sigma2")
    parser.add_argument("--max-samples", type=int, help="Seeded subset for a smoke check; not a final evaluation")
    parser.add_argument("--output-dir", help="New directory; existing directories are never overwritten")
    parser.add_argument("--from-predictions", help="Recompute reports from predictions.csv without model inference")
    args = parser.parse_args()
    if args.from_predictions:
        if args.conditions or args.max_samples or args.checkpoint:
            parser.error("--from-predictions cannot be combined with --conditions, --max-samples or --checkpoint")
        rescore_predictions(args.from_predictions, args.threshold, args.thresholds, args.output_dir)
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required unless --from-predictions is used")
        main(args.config, args.checkpoint, args.source, threshold_override=args.threshold,
             thresholds=args.thresholds, selected_conditions=args.conditions,
             max_samples=args.max_samples, output_dir=args.output_dir)
