"""Stage 5 – plot teacher accuracy vs watermark p-value across variants."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def read_json(path: Path) -> Dict:
    """Read a JSON file into a dictionary.

    Args:
        path: Path to a JSON file.

    Returns:
        Parsed JSON object as a dict.
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_gamma(exp_dir: Path) -> float:
    """Read gamma from an experiment's hash_config.json (fallback 0.5).

    Args:
        exp_dir: Experiment directory containing hash_seed/hash_config.json.

    Returns:
        Gamma value if available, else 0.5.
    """
    hash_cfg = exp_dir / "hash_seed" / "hash_config.json"
    if not hash_cfg.exists():
        return 0.5
    try:
        payload = read_json(hash_cfg)
    except (OSError, json.JSONDecodeError):
        return 0.5
    try:
        return float(payload.get("gamma", 0.5))
    except (TypeError, ValueError):
        return 0.5


def parse_exp_meta(exp_dir: Path) -> Tuple[str | None, str | None]:
    """Parse proxy tag and dataset from an experiment directory name.

    Args:
        exp_dir: Experiment directory with name like teacher_proxy_dataset_nN.

    Returns:
        Tuple of (proxy_tag, dataset) or (None, None) if parsing fails.
    """
    name = exp_dir.name
    if "_n" in name:
        prefix, _ = name.rsplit("_n", 1)
    else:
        prefix = name
    parts = prefix.split("_")
    if len(parts) < 3:
        return None, None
    proxy_tag = parts[1]
    dataset = "_".join(parts[2:])
    return proxy_tag, dataset


def compute_pvalue(mean: float, num_measurements: int, gamma: float) -> float:
    """
    Under H0: measurements are iid in [0,1] with mean gamma.
    Approximate p-value via Hoeffding bound.

    Args:
        mean: Observed mean of measurements.
        num_measurements: Number of measurements.
        gamma: Expected mean under H0.

    Returns:
        Approximate p-value in [0, 1].
    """
    if num_measurements <= 0:
        return 1.0
    delta = gamma - mean
    if delta >= 0:
        return 1.0
    return min(1.0, math.exp(-2 * num_measurements * delta * delta))


def _parse_metrics_name(name: str) -> Tuple[str, str, str, str]:
    """
    Returns (student_tag, method_label, lr, epochs) from a metrics subdir name.
    Expected pattern: {student}_{method_label}_lr{lr}_e{epochs}

    Args:
        name: Metrics subdirectory name.

    Returns:
        Tuple of (student_tag, method_label, lr, epochs).
    """
    if "_lr" not in name or "_e" not in name:
        raise ValueError(f"Unrecognised metrics directory format: {name}")
    left, right = name.rsplit("_lr", 1)
    lr_str, epoch_part = right.split("_e", 1)
    if "_" not in left:
        raise ValueError(f"Missing method label in metrics directory: {name}")
    student_tag, method_label = left.split("_", 1)
    return student_tag, method_label, lr_str, epoch_part


def infer_student_tag(exp_dir: Path, *, lr: str | None = None, epochs: str | None = None) -> str | None:
    """Infer a unique student tag from metrics directory names.

    Args:
        exp_dir: Experiment directory.
        lr: Optional learning rate filter.
        epochs: Optional epoch count filter.

    Returns:
        Student tag if uniquely identifiable, else None.
    """
    metrics_root = exp_dir / "metrics"
    if not metrics_root.exists():
        return None
    tags: set[str] = set()
    for subdir in metrics_root.iterdir():
        if not subdir.is_dir():
            continue
        try:
            student, _, lr_str, epoch_part = _parse_metrics_name(subdir.name)
        except ValueError:
            continue
        if lr and lr_str != lr:
            continue
        if epochs and epoch_part != str(epochs):
            continue
        tags.add(student)
    if len(tags) == 1:
        return next(iter(tags))
    return None


def _teacher_metric(teacher: Dict) -> Tuple[float, str]:
    """Extract the primary teacher metric and label from a teacher_eval payload.

    Args:
        teacher: Dict loaded from teacher_eval.json.

    Returns:
        Tuple of (metric_value, metric_label).

    Raises:
        KeyError: If no known metric field is present.
    """
    if "mean_nll" in teacher:
        return float(teacher["mean_nll"]), "Mean NLL on Original Teacher"
    if "answer_forced_accuracy" in teacher:
        return float(teacher["answer_forced_accuracy"]), "Teacher Answer-Forced Accuracy"
    if "raw_accuracy" in teacher:
        return float(teacher["raw_accuracy"]), "Teacher Raw Accuracy"
    raise KeyError("Teacher metric not found in teacher_eval.json")


def gather_points(
    exp_dir: Path,
    variant: str,
    *,
    student_tag: str | None = None,
    lr: str | None = None,
    epochs: str | None = None,
) -> Tuple[Dict[str, List[Tuple[float, float, str]]], str]:
    """
    variant in {open_supervised, open_unsupervised, closed_supervised, closed_unsupervised}
    Filters by student_tag/lr/epochs if provided.

    Args:
        exp_dir: Experiment directory to scan.
        variant: Watermark variant string.
        student_tag: Optional student tag filter.
        lr: Optional learning rate filter.
        epochs: Optional epoch count filter.

    Returns:
        Tuple of (bucketed points, x-axis label string).
    """
    metrics_root = exp_dir / "metrics"
    buckets: Dict[str, List[Tuple[float, float, str]]] = {"radioactive": [], "ads": [], "control": []}
    x_label: str | None = None
    if not metrics_root.exists():
        return buckets, "Teacher Metric"
    gamma = read_gamma(exp_dir)
    for subdir in metrics_root.iterdir():
        if not subdir.is_dir():
            continue
        try:
            student, method_label, lr_str, epoch_part = _parse_metrics_name(subdir.name)
        except ValueError:
            continue
        if student_tag and student != student_tag:
            continue
        if lr and lr_str != lr:
            continue
        if epochs and epoch_part != str(epochs):
            continue
        wm_file = subdir / f"watermark_{variant}.json"
        if not wm_file.exists():
            continue
        method_root = "training_traces" if "supervised" in variant else "alternative_traces"
        teacher_eval = exp_dir / method_root / method_label / "teacher_eval.json"
        if not teacher_eval.exists():
            continue
        teacher = read_json(teacher_eval)
        watermark = read_json(wm_file)
        try:
            teacher_metric, metric_label = _teacher_metric(teacher)
        except KeyError:
            continue
        if x_label is None:
            x_label = metric_label
        elif x_label != metric_label:
            x_label = "Teacher Metric"
        mean = float(watermark.get("mean", 0.5))
        n = int(watermark.get("num_measurements", 0))
        pval = compute_pvalue(mean, n, gamma)
        if method_label.startswith("radioactive"):
            bucket = "radioactive"
        elif method_label.startswith("ads"):
            bucket = "ads"
        elif method_label.startswith("control"):
            bucket = "control"
        else:
            raise ValueError(f"Unrecognised method label: {method_label}")
        scale_txt = method_label.split("-", 1)[-1] if "-" in method_label else method_label
        scale_txt = scale_txt.replace("delta", "").replace("lambda", "")
        scale_txt = scale_txt.replace("_", ".")
        if bucket == "radioactive":
            greek_prefix = "δ="
        elif bucket == "ads":
            greek_prefix = "λ="
        else:
            greek_prefix = ""
        label_val = f"{greek_prefix}{scale_txt}"
        buckets[bucket].append((teacher_metric, pval, label_val))
    return buckets, (x_label or "Teacher Metric")


def plot(
    exp_dir: Path,
    variant: str,
    *,
    student_tag: str | None = None,
    lr: str | None = None,
    epochs: str | None = None,
    fig_dir: Path | None = None,
    show_labels: bool = False,
) -> List[Path]:
    """Render scatter plots for a given watermark variant.

    Args:
        exp_dir: Experiment directory to scan.
        variant: Watermark variant string.
        student_tag: Optional student tag filter.
        lr: Optional learning rate filter.
        epochs: Optional epoch count filter.
        fig_dir: Optional output directory for figures.
        show_labels: If True, annotate points with delta/lambda labels.

    Returns:
        List of generated figure paths.
    """
    buckets, x_label = gather_points(exp_dir, variant, student_tag=student_tag, lr=lr, epochs=epochs)
    points = buckets["radioactive"] + buckets["ads"] + buckets.get("control", [])
    if not points:
        raise RuntimeError(f"No experiment results found for {variant} under {exp_dir}")
    styles = {
        "radioactive": {"color": "#4DBBD5", "marker": "o", "label": "Red-and-Green-List"},
        "ads": {"color": "#E64B35", "marker": "s", "label": "Antidistillation"},
        "control": {"color": "#1E8449", "marker": "D", "label": "Unfingerprinted"},
    }

    proxy_tag, dataset = parse_exp_meta(exp_dir)
    inferred_student = student_tag or infer_student_tag(exp_dir, lr=lr, epochs=epochs)
    proxy_equal = proxy_tag is not None and inferred_student is not None and proxy_tag == inferred_student
    proxy_text = "Proxy = Student" if proxy_equal else "Proxy ≠ Student"
    mode_text = "Student Open-Weight" if variant.startswith("open") else "Student Closed-Weight"
    dataset_text = dataset.upper() if dataset else "DATASET"
    title = f"{proxy_text}, {mode_text} ({dataset_text})"

    out_dir = fig_dir or (exp_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = []
    if student_tag:
        suffix.append(student_tag)
    if lr:
        suffix.append(f"lr{lr}")
    if epochs:
        suffix.append(f"e{epochs}")
    suffix_str = "_".join(suffix)
    name = f"stage5_plot_{variant}"
    if suffix_str:
        name = f"{name}_{suffix_str}"
    output_paths: List[Path] = []

    def _render(size: Tuple[int, int], size_suffix: str) -> List[Path]:
        """Render and save a plot at a given size.

        Args:
            size: Figure size in inches (width, height).
            size_suffix: Filename suffix for this size.

        Returns:
            List of file paths written for this size.
        """
        fig, ax = plt.subplots(figsize=size)
        for key, data in buckets.items():
            if not data:
                continue
            xs = [pt[0] for pt in data]
            ys = [pt[1] for pt in data]
            lbls = [pt[2] for pt in data]
            ax.scatter(
                xs,
                ys,
                c=styles[key]["color"],
                marker=styles[key]["marker"],
                alpha=0.8,
                label=styles[key]["label"],
                s=160,
            )
            if show_labels:
                for x, y, txt in zip(xs, ys, lbls):
                    ax.annotate(txt, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=10)

        ax.set_xlabel(x_label, fontsize=20, fontweight="bold")
        ax.set_ylabel("Fingerprint p-value", fontsize=20, fontweight="bold")
        ax.set_title(title, fontsize=22, fontweight="bold")
        ax.tick_params(axis="both", labelsize=18)
        ax.set_yscale("log")
        ax.axhline(0.05, color="black", linestyle="--", linewidth=2, label="p = 0.05")

        legend = ax.legend(fontsize=18)
        if legend is not None:
            for text in legend.get_texts():
                if text.get_text() == "Antidistillation":
                    text.set_fontstyle("italic")
                    text.set_fontweight("bold")

        fig.tight_layout()
        name_with_size = f"{name}{size_suffix}"
        output_path = out_dir / f"{name_with_size}.png"
        fig.savefig(output_path)
        plt.close(fig)
        return [output_path]

    output_paths.extend(_render((10, 6), ""))
    return output_paths


def main(argv: List[str] | None = None) -> None:
    """CLI entrypoint for Stage 5 plotting.

    Args:
        argv: Optional list of CLI arguments (defaults to sys.argv).

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Stage 5 – plotting")
    parser.add_argument("--exp-dir", type=Path, required=True, help="Base experiment directory")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["open_supervised", "open_unsupervised", "closed_supervised", "closed_unsupervised"],
        choices=["open_supervised", "open_unsupervised", "closed_supervised", "closed_unsupervised"],
    )
    parser.add_argument("--student-tag", type=str, default=None, help="Filter metrics to this student tag")
    parser.add_argument("--lr", type=str, default=None, help="Filter metrics to this lr tag (as encoded in folder name)")
    parser.add_argument("--epochs", type=str, default=None, help="Filter metrics to this epoch count")
    parser.add_argument("--fig-dir", type=Path, default=None, help="Output directory for figures (default: exp_dir/figures)")
    parser.add_argument("--show-labels", action="store_true", help="Annotate points with delta/lambda labels")
    args = parser.parse_args(argv)

    fig_dir = args.fig_dir or (args.exp_dir / "figures")
    for variant in args.variants:
        results = plot(
            args.exp_dir,
            variant,
            student_tag=args.student_tag,
            lr=args.lr,
            epochs=args.epochs,
            fig_dir=fig_dir,
            show_labels=args.show_labels,
        )
        for result in results:
            print(f"Wrote plot to {result}")


if __name__ == "__main__":
    main()
