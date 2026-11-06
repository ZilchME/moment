"""Train/evaluate MOMENT soft sensing, or predict from a self-contained checkpoint."""

import argparse
import csv
import hashlib
import json
import random
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from momentfm.data.debutanizer import (
    DebutanizerWindows, Standardizer, chronological_rows, load_debutanizer, make_features,
)
from momentfm.models.soft_sensor import MOMENTSoftSensor
from momentfm.utils.device import resolve_device


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def regression_metrics(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if truth.shape != prediction.shape or not len(truth):
        raise ValueError("Metrics require equally sized, nonempty arrays.")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require finite values.")
    residual = truth - prediction
    sse = float(np.sum(residual ** 2))
    sst = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "n": len(truth),
        "mse": sse / len(truth),
        "rmse": float(np.sqrt(sse / len(truth))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": 1 - sse / sst if sst > 0 else None,
        "rmse_paper_n_minus_1": float(np.sqrt(sse / (len(truth) - 1))) if len(truth) > 1 else None,
    }


def loader_for(dataset, args, *, shuffle=False):
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, generator=generator, drop_last=False,
                      pin_memory=args.device == "cuda")


def inputs_to_device(batch, device):
    return {key: batch[key].to(device) for key in ("x", "input_mask", "observed_mask")}


@torch.no_grad()
def cache_features(model, loader, device):
    """Cache frozen features in CPU RAM once; do not fit on validation/test labels."""
    model.eval()
    features, targets = [], []
    for batch in loader:
        features.append(model.encode(**inputs_to_device(batch, device)).cpu())
        targets.append(batch["target"])
    # no_grad (rather than inference_mode) permits head backprop through these inputs.
    return TensorDataset(torch.cat(features), torch.cat(targets))


def batch_prediction(model, batch, device, cached):
    if cached:
        features, target = batch
        return model.head(features.to(device)).squeeze(-1), target.to(device)
    return model(**inputs_to_device(batch, device)), batch["target"].to(device)


def epoch_mse(model, loader, device, *, optimizer=None, cached=False):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction, target = batch_prediction(model, batch, device, cached)
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite loss; check the data and optimization settings.")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=1.0, error_if_nonfinite=True,
                )
                optimizer.step()
            total += float(loss.detach().cpu()) * len(target)
            count += len(target)
    return total / count


@torch.no_grad()
def predict_rows(model, loader, device, y_scaler):
    model.eval()
    rows, predictions = [], []
    for batch in loader:
        values = model(**inputs_to_device(batch, device))
        predictions.append(values.cpu().numpy())
        rows.append(batch["row_index"].numpy())
    prediction = y_scaler.inverse(np.concatenate(predictions)[:, None]).ravel()
    if not np.isfinite(prediction).all():
        raise RuntimeError("Model produced a nonfinite prediction.")
    return np.concatenate(rows), prediction


def window_dataset(features, targets, rows, args, backbone):
    return DebutanizerWindows(features, targets, rows, args.context_length,
                              backbone.seq_len, backbone.patch_len)


def save_checkpoint(path, model, args, x_scaler, y_scaler, feature_names,
                    best_epoch, val_mse, data_hash):
    checkpoint = {
        "format_version": 1,
        "model_name": "moment_soft_sensor",
        "backbone_config": deepcopy(vars(model.backbone.config)),
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "n_channels": model.n_channels,
        "feature_names": list(feature_names),
        "x_scaler": x_scaler.state(), "y_scaler": y_scaler.state(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "best_epoch": best_epoch,
        "best_validation_mse_standardized": val_mse,
        "training_data_sha256": data_hash,
        "versions": {name: version(name) for name in ("torch", "transformers", "numpy", "momentfm")},
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def restore_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("model_name") != "moment_soft_sensor":
        raise ValueError("Unsupported soft-sensor checkpoint.")
    config = checkpoint["config"]
    backbone_config = deepcopy(checkpoint["backbone_config"])
    backbone_config["enable_gradient_checkpointing"] = False
    model = MOMENTSoftSensor.from_config(
        backbone_config, checkpoint["n_channels"], mode=config["mode"],
        window_stats=not config["no_window_stats"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return (model.eval().to(device=device, dtype=torch.float32), checkpoint,
            Standardizer.from_state(checkpoint["x_scaler"]),
            Standardizer.from_state(checkpoint["y_scaler"]))


def write_predictions(path, raw, rows, prediction, train_mean):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", "y_true", "y_pred", "train_mean", "persistence"])
        for row, value in zip(rows, prediction):
            truth, previous = raw[row, 7], raw[row - 1, 7]
            writer.writerow([int(row), float(truth) if np.isfinite(truth) else "", float(value),
                             train_mean, float(previous) if np.isfinite(previous) else ""])


def evaluate_predictions(raw, rows, prediction, train_mean):
    truth = raw[rows, 7]
    return {
        "moment": regression_metrics(truth, prediction),
        "train_mean": regression_metrics(truth, np.full(len(rows), train_mean)),
        "persistence_requires_measured_past_y": regression_metrics(truth, raw[rows - 1, 7]),
    }


def plotting():
    # Use a noninteractive backend on training servers without a display.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_learning_curve(directory, history, best_epoch):
    plt = plotting()
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    epochs = [item["epoch"] for item in history]
    ax.plot(epochs, [item["train_mse_standardized"] for item in history], label="Train")
    ax.plot(epochs, [item["validation_mse_standardized"] for item in history], label="Validation")
    ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch: {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="Standardized MSE", title="Training and validation loss")
    ax.legend()
    fig.savefig(Path(directory) / "learning_curve.png", dpi=180)
    plt.close(fig)


def save_evaluation_plots(directory, raw, rows, prediction, train_mean):
    plt = plotting()
    truth = raw[rows, 7]
    scores = regression_metrics(truth, prediction)
    residual = prediction - truth
    directory = Path(directory)

    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    ax.plot(rows, truth, label="Measured", linewidth=1.2)
    ax.plot(rows, prediction, label="MOMENT", linewidth=1.0, alpha=0.85)
    ax.axhline(train_mean, color="gray", linestyle="--", label="Training mean")
    ax.set(xlabel="Sample index (0-based)", ylabel="y (CSV scale)",
           title=f"Held-out test predictions | RMSE={scores['rmse']:.4g}, MAE={scores['mae']:.4g}")
    ax.legend()
    fig.savefig(directory / "prediction_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
    ax.scatter(truth, prediction, s=10, alpha=0.5)
    low, high = float(min(truth.min(), prediction.min())), float(max(truth.max(), prediction.max()))
    padding = max((high - low) * 0.05, 1e-6)
    limits = (low - padding, high + padding)
    ax.plot(limits, limits, "--", color="gray", label="Ideal")
    r2 = f"{scores['r2']:.4g}" if scores["r2"] is not None else "undefined (constant target)"
    ax.set(xlabel="Measured y (CSV scale)", ylabel="Predicted y (CSV scale)",
           xlim=limits, ylim=limits, title=f"Measured vs predicted | R²={r2}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.savefig(directory / "prediction_scatter.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    axes[0].plot(rows, residual, linewidth=1)
    axes[0].axhline(0, color="gray", linestyle="--")
    axes[0].set(xlabel="Sample index (0-based)", ylabel="Prediction - measured", title="Test residuals")
    axes[1].hist(residual, bins=min(40, max(1, int(np.sqrt(len(rows))))), color="steelblue")
    axes[1].set(xlabel="Prediction - measured (CSV scale)", ylabel="Count", title="Residual distribution")
    fig.savefig(directory / "residuals.png", dpi=180)
    plt.close(fig)


def prepare_output(directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {directory}; choose a new run directory.")
    directory.mkdir(parents=True, exist_ok=True)


def train(args):
    plotting()  # Fail before training if plotting dependencies are missing.
    raw = load_debutanizer(args.data)
    splits = chronological_rows(len(raw), args.first_row, args.train_size, args.val_size)
    features, names = make_features(raw, args.features)
    x_scaler = Standardizer.fit(features[splits["train"]])
    y_scaler = Standardizer.fit(raw[splits["train"], 7:8])
    x = x_scaler.transform(features)
    y = y_scaler.transform(raw[:, 7:8]).ravel()
    # Detect output collisions before creating/downloading a model.
    prepare_output(args.output)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.device = str(device)
    model = MOMENTSoftSensor.from_pretrained(
        args.model, len(names), mode=args.mode, window_stats=not args.no_window_stats,
        revision=args.revision, local_files_only=args.local_files_only,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device=device, dtype=torch.float32)
    datasets = {name: window_dataset(x, y, rows, args, model.backbone) for name, rows in splits.items()}
    cached = args.mode == "probe"
    if cached:
        training = cache_features(model, loader_for(datasets["train"], args), device)
        validation = cache_features(model, loader_for(datasets["validation"], args), device)
    else:
        training, validation = datasets["train"], datasets["validation"]
    train_loader = loader_for(training, args, shuffle=True)
    val_loader = loader_for(validation, args)
    groups = [{"params": model.head.parameters(), "lr": args.lr}]
    if not cached:
        groups.append({"params": model.backbone.parameters(), "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    best_mse, best_epoch, stale = float("inf"), 0, 0
    history = []
    data_hash = file_sha256(args.data)
    checkpoint_path = args.output / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train_mse = epoch_mse(model, train_loader, device, optimizer=optimizer, cached=cached)
        val_mse = epoch_mse(model, val_loader, device, cached=cached)
        history.append({"epoch": epoch, "train_mse_standardized": train_mse,
                        "validation_mse_standardized": val_mse})
        print(json.dumps(history[-1]), flush=True)
        if val_mse < best_mse:
            best_mse, best_epoch, stale = val_mse, epoch, 0
            save_checkpoint(checkpoint_path, model, args, x_scaler, y_scaler, names,
                            epoch, val_mse, data_hash)
        else:
            stale += 1
        json_write(args.output / "history.json", history)
        if stale >= args.patience:
            break
    # Test targets are not used for fitting, early stopping or feature selection.
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state"], strict=True)
    del best
    rows, prediction = predict_rows(model, loader_for(datasets["test"], args), device, y_scaler)
    train_mean = float(raw[splits["train"], 7].mean())
    scores = evaluate_predictions(raw, rows, prediction, train_mean)
    report = {
        "data_sha256": data_hash, "raw_samples": len(raw), "target": "y",
        "feature_mode": args.features, "feature_names": list(names),
        "model": args.model, "revision": args.revision, "seed": args.seed,
        "context_length": args.context_length, "window_stats": not args.no_window_stats,
        "mode": args.mode, "best_epoch": best_epoch,
        "best_validation_mse_standardized": best_mse,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "evaluation": "same-time soft sensing: inputs through t predict y(t); no recursive target feedback",
        "target_scale": "CSV y values; physical units/upstream normalization parameters are unavailable",
        "historical_y_inputs": args.features == "causal13",
        "persistence_comparison": "persistence uses measured y(t-1), including when MOMENT uses process inputs only",
        "split": {name: {"n": len(indices), "raw_index_range_inclusive": [int(indices[0]), int(indices[-1])]}
                  for name, indices in splits.items()},
        "metrics": scores,
    }
    json_write(args.output / "metrics.json", report)
    write_predictions(args.output / "predictions.csv", raw, rows, prediction, train_mean)
    save_learning_curve(args.output, history, best_epoch)
    # Reload a standalone checkpoint and repeat inference on exactly the held-out rows.
    # Release the training model/optimizer before allocating another backbone on the GPU.
    del optimizer, groups, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    validation_output = args.output / "inference_validation"
    print("Reloading best.pt for held-out inference validation...", flush=True)
    reloaded_rows, reloaded_prediction = infer(argparse.Namespace(
        command="evaluate", checkpoint=checkpoint_path, data=args.data,
        output=validation_output, device=args.device,
        batch_size=args.batch_size, workers=args.workers,
    ))
    matched = bool(np.array_equal(rows, reloaded_rows) and
                   np.allclose(prediction, reloaded_prediction, rtol=1e-5, atol=1e-6))
    json_write(validation_output / "consistency.json", {
        "checkpoint_reloaded": True, "matches_training_evaluation": matched,
        "n_predictions": len(reloaded_rows), "rtol": 1e-5, "atol": 1e-6,
        "max_absolute_prediction_difference": float(np.max(np.abs(prediction - reloaded_prediction)))
        if np.array_equal(rows, reloaded_rows) else None,
    })
    if not matched:
        raise RuntimeError("Reloaded checkpoint inference differs from training evaluation; see consistency.json.")
    print(json.dumps(scores, indent=2), flush=True)
    print(f"Saved checkpoint and test results to {args.output.resolve()}")


def infer(args):
    prepare_output(args.output)
    device = resolve_device(args.device)
    model, checkpoint, x_scaler, y_scaler = restore_checkpoint(args.checkpoint, device)
    saved = checkpoint["config"]
    raw = load_debutanizer(args.data, allow_missing_y=args.command == "predict")
    features, names = make_features(raw, saved["features"])
    if list(names) != checkpoint["feature_names"]:
        raise ValueError("Feature schema differs from the checkpoint.")
    if args.command == "evaluate":
        if file_sha256(args.data) != checkpoint["training_data_sha256"]:
            raise ValueError("Evaluate requires the original training CSV; use predict for a different file.")
        splits = chronological_rows(len(raw), saved["first_row"], saved["train_size"], saved["val_size"])
        requested_rows = splits["test"]
    else:
        first = saved["first_row"] if args.start_row is None else args.start_row
        last = len(raw) if args.stop_row is None else args.stop_row
        if first < 4 or last > len(raw) or first >= last:
            raise ValueError("Require 4 <= start_row < stop_row <= number of rows.")
        requested_rows = np.arange(first, last)
    rows = requested_rows[np.isfinite(features[requested_rows]).all(axis=1)]
    if not len(rows):
        raise ValueError("No usable rows: causal13 requires four previously measured y values.")
    skipped = len(requested_rows) - len(rows)
    if args.command == "evaluate" and skipped:
        raise ValueError("Evaluation must preserve all test target rows.")
    args.context_length = saved["context_length"]
    args.seed = saved["seed"]
    args.device = str(device)
    dataset = window_dataset(x_scaler.transform(features), y_scaler.transform(raw[:, 7:8]).ravel(),
                             rows, args, model.backbone)
    indices, prediction = predict_rows(model, loader_for(dataset, args), device, y_scaler)
    # This is the mean of the training targets saved in the checkpoint, not new data.
    train_mean = float(y_scaler.mean[0])
    write_predictions(args.output / "predictions.csv", raw, indices, prediction, train_mean)
    report = {"command": args.command, "n_predictions": len(indices), "skipped_rows": skipped,
              "checkpoint": str(args.checkpoint.resolve()), "data_sha256": file_sha256(args.data),
              "independent_test": args.command == "evaluate"}
    if args.command == "evaluate":
        report["metrics"] = evaluate_predictions(raw, indices, prediction, train_mean)
    json_write(args.output / "metrics.json", report)
    if args.command == "evaluate":
        save_evaluation_plots(args.output, raw, indices, prediction, train_mean)
    print(json.dumps(report, indent=2), flush=True)
    return indices, prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Fit chronologically and evaluate the held-out test set")
    training.add_argument("--model", default="AutonLab/MOMENT-1-small", help="Hub model ID or local base weights")
    training.add_argument("--revision", default=None, help="Optional immutable Hub revision for the base model")
    training.add_argument("--local-files-only", action="store_true")
    training.add_argument("--features", choices=["process", "causal13"], default="process")
    training.add_argument("--mode", choices=["probe", "finetune"], default="probe")
    training.add_argument("--context-length", type=int, default=32)
    training.add_argument("--first-row", type=int, default=4)
    training.add_argument("--train-size", type=int, default=800)
    training.add_argument("--val-size", type=int, default=200)
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--patience", type=int, default=15)
    training.add_argument("--lr", type=float, default=1e-3, help="Regression head learning rate")
    training.add_argument("--backbone-lr", type=float, default=1e-5)
    training.add_argument("--weight-decay", type=float, default=0.01)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--no-window-stats", action="store_true", help="Ablate the level/scale bypass")
    training.add_argument("--gradient-checkpointing", action="store_true", help="Only used in full fine-tuning")
    for command in ("evaluate", "predict"):
        child = commands.add_parser(command, help="Load a saved checkpoint without downloading the base model")
        child.add_argument("--checkpoint", type=Path, required=True)
        if command == "predict":
            child.add_argument("--start-row", type=int)
            child.add_argument("--stop-row", type=int, help="Exclusive end of the requested target rows")
    for child in (training, commands.choices["evaluate"], commands.choices["predict"]):
        child.add_argument("--data", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True, help="A new or empty output directory")
        child.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
        child.add_argument("--batch-size", type=int, default=16)
        child.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Batch size must be positive and worker count nonnegative.")
    if args.command == "train":
        if (args.epochs < 1 or args.patience < 1 or not 0 <= args.seed < 2 ** 32
                or not np.isfinite([args.lr, args.backbone_lr, args.weight_decay]).all()
                or args.lr <= 0 or args.backbone_lr <= 0 or args.weight_decay < 0
                or args.context_length < 8 or args.context_length > 512 or args.context_length % 8):
            parser.error("Require positive epochs/patience/LRs, nonnegative weight decay, valid seed, and context in 8..512 by 8.")
        train(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
