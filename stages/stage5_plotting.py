"""Stage 5 – plot teacher accuracy vs watermark p-value across variants."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_gamma(exp_dir: Path) -> float:
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
    name = exp_dir.name
    if "_n" in name:
        prefix, _ = name.rsplit("_n", 1)
    else:
        prefix = name
    parts = prefix.split("_")
    if len(parts) < 3:
        return None, None
    offset = 1 if parts[0] == "efpr" else 0
    if len(parts) < offset + 3:
        return None, None
    proxy_tag = parts[offset + 1]
    dataset = "_".join(parts[offset + 2 :])
    return proxy_tag, dataset


def compute_pvalue(mean: float, num_measurements: int, gamma: float) -> float:
    """
    Under H0: measurements are iid in [0,1] with mean gamma.
    Approximate p-value via Hoeffding bound.
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
    """
    if "_lr" not in name or "_e" not in name:
        raise ValueError(f"Unrecognised metrics directory format: {name}")
    left, right = name.rsplit("_lr", 1)
    lr_str, epoch_part = right.split("_e", 1)
    if "_" not in left:
        raise ValueError(f"Missing method label in metrics directory: {name}")
    student_tag, method_label = left.split("_", 1)
    return student_tag, method_label, lr_str, epoch_part

def _parse_lm_eval_results_name(name: str) -> Tuple[str, str, str, str]:
    """
    Returns (student_tag, method_label, lr, epochs) from a lm_eval subdir name.
    Expected pattern: {prefix}{student}_{method_label}_lr{lr}_e{epochs}{suffix}
    """
    # Remove the prefix and suffix that lm-eval uses based on the full model path.
    prefix_split_str = "__models__"
    suffix_strip_str = "__student_lora"
    if prefix_split_str not in name or not name.endswith(suffix_strip_str):
        raise ValueError(f"Unrecognised lm_eval results directory format: {name}")
    _, name = name.split(prefix_split_str, 1)
    name = name.split(suffix_strip_str, 1)[0]
    # Then proceed as normal
    if "_lr" not in name or "_e" not in name:
        raise ValueError(f"Unrecognised metrics directory format: {name}")
    left, right = name.rsplit("_lr", 1)
    lr_str, epoch_part = right.split("_e", 1)
    if "_" not in left:
        raise ValueError(f"Missing method label in metrics directory: {name}")
    student_tag, method_label = left.split("_", 1)
    return student_tag, method_label, lr_str, epoch_part


def infer_student_tag(exp_dir: Path, *, lr: str | None = None, epochs: str | None = None) -> str | None:
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


def collect_student_tags(exp_dir: Path, *, lr: str | None = None, epochs: str | None = None) -> set[str]:
    metrics_root = exp_dir / "metrics"
    if not metrics_root.exists():
        return set()
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
    return tags


def _suffix_parts(
    student_tag: str | None,
    student_tags: set[str] | None,
    lr: str | None,
    epochs: str | None,
) -> List[str]:
    suffix: List[str] = []
    if student_tag:
        suffix.append(student_tag)
    elif student_tags and len(student_tags) == 1:
        suffix.append(next(iter(student_tags)))
    if lr:
        suffix.append(f"lr{lr}")
    if epochs:
        suffix.append(f"e{epochs}")
    return suffix


def _teacher_metric(teacher: Dict) -> Tuple[float, str]:
    if "mean_nll" in teacher:
        return float(teacher["mean_nll"]), "Mean NLL on Original Teacher"
    if "answer_forced_accuracy" in teacher:
        return float(teacher["answer_forced_accuracy"]), "Teacher Answer-Forced Accuracy"
    if "raw_accuracy" in teacher:
        return float(teacher["raw_accuracy"]), "Teacher Raw Accuracy"
    raise KeyError("Teacher metric not found in teacher_eval.json")


def gather_points(
    exp_dir: Path,
    variant: str | list,
    *,
    student_tag: str | None = None,
    student_tags: set[str] | None = None,
    lr: str | None = None,
    epochs: str | None = None,
) -> Tuple[Dict[str, List[Tuple[float, float, str]]], str]:
    """
    variant in {open_supervised, open_unsupervised, closed_supervised, closed_unsupervised}
    Filters by student_tag/lr/epochs if provided.
    """
    metrics_root = exp_dir / "metrics"
    buckets: Dict[str, List[Tuple[float, float, str]]] = {"radioactive": [], "ads": [], "control": []}
    x_label: str | None = None
    if not metrics_root.exists():
        return buckets, "Teacher Metric"
    gamma = read_gamma(exp_dir)

    if not isinstance(variant, list):
        variant = [variant]
    for variant in variant:
        for subdir in metrics_root.iterdir():
            if not subdir.is_dir():
                continue
            try:
                student, method_label, lr_str, epoch_part = _parse_metrics_name(subdir.name)
            except ValueError:
                continue
            if student_tag and student != student_tag:
                continue
            if student_tags and student not in student_tags:
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
            bucket = ""
            if method_label.startswith("radioactive"):
                bucket = "radioactive"
            elif method_label.startswith("ads"):
                bucket = "ads"
            elif method_label.startswith("control"):
                bucket = "control"
            else:
                raise ValueError(f"Unrecognised method label: {method_label}")
            
            scale_txt = method_label.split("-", 1)[-1] if "-" in method_label else method_label
            # also catch the trials identifier if present
            trial_txt = ""
            if "trial" in scale_txt:
                scale_txt, trial_txt = scale_txt.split("trial", 1)
                scale_txt = scale_txt.rstrip("-")
                
            scale_txt = scale_txt.replace("delta", "").replace("lambda", "")
            scale_txt = scale_txt.replace("_", ".")
            if bucket == "radioactive":
                greek_prefix = "δ=" 
            elif bucket == "ads":       
                greek_prefix = "λ="
            else:
                greek_prefix = ""
            label_val = f"{greek_prefix}{scale_txt}" + (f" trial {trial_txt}" if trial_txt else "")
            label_val = label_val.strip()
            buckets[bucket].append((teacher_metric, pval, label_val))
    return buckets, (x_label or "Teacher Metric")


def gather_lm_eval_points(
    exp_dir: Path,
    variant: str,
    task_name: str = "gsm8k_cot",
    primary_metric: str = "exact_match,flexible-extract", 
    *,
    student_tag: str | None = None,
    student_tags: set[str] | None = None,
    lr: str | None = None,
    epochs: str | None = None,
) -> Dict[str, List[Tuple[float, float, str]]]:
    """
    The is a similar function to gather_points, but looks in the lm_eval subdirectory
    """
    metrics_root = exp_dir / "metrics"
    results_root = exp_dir / "lm_eval"
    buckets: Dict[str, List[Tuple[float, float, str]]] = {"radioactive": [], "ads": [], "control": []}
    gamma = read_gamma(exp_dir)
    if not results_root.exists():
        return buckets
    # First we build tables of results and metrics, then we match them.
    result_dir_table = {}
    metric_dir_table = {}
    for result_subdir in results_root.iterdir():
        if not result_subdir.is_dir():
            continue
        try:
            student, method_label, lr_str, epoch_part = _parse_lm_eval_results_name(result_subdir.name)
        except ValueError:
            continue
        result_dir_table[(student, method_label, lr_str, epoch_part)] = result_subdir
    for metric_subdir in metrics_root.iterdir():
        if not metric_subdir.is_dir():
            continue
        try:
            student, method_label, lr_str, epoch_part = _parse_metrics_name(metric_subdir.name)
        except ValueError:
            continue
        metric_dir_table[(student, method_label, lr_str, epoch_part)] = metric_subdir
    # now we match them
    zipped_dirs = []
    for key in result_dir_table.keys():
        if key not in metric_dir_table:
            continue
        result_subdir = result_dir_table[key]
        metric_subdir = metric_dir_table[key]
        student, method_label, lr_str, epoch_part = key
        zipped_dirs.append((result_subdir, metric_subdir, student, method_label, lr_str, epoch_part))

    for result_subdir, metric_subdir, student, method_label, lr_str, epoch_part in zipped_dirs:
        if student_tag and student != student_tag:
            continue
        if student_tags and student not in student_tags:
            continue
        if lr and lr_str != lr:
            continue
        if epochs and epoch_part != str(epochs):
            continue
        wm_file = metric_subdir / f"watermark_{variant}.json"
        if not wm_file.exists():
            continue
        
        # Main difference is here as we're getting the student perf not teacher perf
        student_lm_eval = list(result_subdir.glob("results*.json"))
        if len(student_lm_eval) == 0:
            continue
        student_eval = read_json(student_lm_eval[0])
        student_eval_value = student_eval.get("results", {}).get(task_name, {}).get(primary_metric, None) # float
        if student_eval_value is None:
            continue
        student_accuracy = float(student_eval_value)

        # Then again, proceed as normal
        watermark = read_json(wm_file)
        mean = float(watermark.get("mean", 0.5))
        n = int(watermark.get("num_measurements", 0))
        pval = compute_pvalue(mean, n, gamma)
        bucket = ""
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
        buckets[bucket].append((student_accuracy, pval, label_val))
    return buckets, f"Student {primary_metric} on {task_name}"


def plot(
    exp_dir: Path,
    variant: str,
    *,
    student_tag: str | None = None,
    student_tags: set[str] | None = None,
    lr: str | None = None,
    epochs: str | None = None,
    fig_dir: Path | None = None,
    show_labels: bool = False,
) -> List[Path]:
    buckets, x_label = gather_points(
        exp_dir,
        variant,
        student_tag=student_tag,
        student_tags=student_tags,
        lr=lr,
        epochs=epochs,
    )
    points = buckets["radioactive"] + buckets["ads"] + buckets.get("control", [])
    if not points:
        raise RuntimeError(f"No experiment results found for {variant} under {exp_dir}")
    plt.figure(figsize=(10, 6))
    styles = {
        "radioactive": {"color": "#4DBBD5", "marker": "o", "label": "Red-and-Green-List"},
        "ads": {"color": "#E64B35", "marker": "s", "label": "Antidistillation"},
        "control": {"color": "#1E8449", "marker": "D", "label": "Unfingerprinted"},
    }
    for key, data in buckets.items():
        if not data:
            continue
        xs = [pt[0] for pt in data]
        ys = [pt[1] for pt in data]
        lbls = [pt[2] for pt in data]
        plt.scatter(xs, ys, c=styles[key]["color"], marker=styles[key]["marker"], alpha=0.8, label=styles[key]["label"], s=160)
        if show_labels:
            for x, y, txt in zip(xs, ys, lbls):
                plt.annotate(txt, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=10)
    plt.xlabel(x_label, fontsize=20, fontweight="bold")
    plt.ylabel("Fingerprint p-value", fontsize=20, fontweight="bold")
    proxy_tag, dataset = parse_exp_meta(exp_dir)
    inferred_student = student_tag or infer_student_tag(exp_dir, lr=lr, epochs=epochs)
    proxy_equal = proxy_tag is not None and inferred_student is not None and proxy_tag == inferred_student
    proxy_text = "Proxy = Student" if proxy_equal else "Proxy ≠ Student"
    mode_text = "Student Open-Weight" if variant.startswith("open") else "Student Closed-Weight"
    dataset_text = dataset.upper() if dataset else "DATASET"
    plt.title(f"{proxy_text}, {mode_text} ({dataset_text})", fontsize=22, fontweight="bold")
    plt.tick_params(axis="both", labelsize=18)
    plt.yscale("log")
    plt.axhline(0.05, color="black", linestyle="--", linewidth=2, label="p = 0.05")

    legend = plt.legend(fontsize=18)
    for text in legend.get_texts():
        if text.get_text() == "Antidistillation":
            text.set_fontstyle("italic")
            text.set_fontweight("bold")
    out_dir = fig_dir or (exp_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix_str = "_".join(_suffix_parts(student_tag, student_tags, lr, epochs))
    name = f"stage5_plot_{variant}"
    if suffix_str:
        name = f"{name}_{suffix_str}"
    output_paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    plt.tight_layout()
    for output_path in output_paths:
        plt.savefig(output_path)
    plt.close()
    return output_paths

def plot_roc_auc(
    exp_dir: Path,
    variant: str,
    *,
    student_tag: str | None = None,
    student_tags: set[str] | None = None,
    lr: str | None = None,
    epochs: str | None = None,
    fig_dir: Path | None = None,
    show_labels: bool = False,
    threshold_style: str = "data-dependent", # or "linear"
    min_num_thresholds: int = 100,
    balanced: bool = False, # would like to turn this on but if missing data then can't
) -> List[Path]:
    """
    This variation is to use the sets of gathered points as individual detection problem instances.
    """
    if "+" in variant:
        variant = variant.split("+")
    
    buckets, x_label = gather_points(
        exp_dir,
        variant,
        student_tag=student_tag,
        student_tags=student_tags,
        lr=lr,
        epochs=epochs,
    )

    # get labelling information
    proxy_tag, dataset = parse_exp_meta(exp_dir)
    inferred_student = student_tag or infer_student_tag(exp_dir, lr=lr, epochs=epochs)
    proxy_equal = proxy_tag is not None and inferred_student is not None and proxy_tag == inferred_student
    proxy_text = "Proxy = Student" if proxy_equal else "Proxy ≠ Student"
    print(f"Proxy tag: {proxy_tag}, Inferred student tag: {inferred_student}, Proxy equal: {proxy_equal}")

    if isinstance(variant, list):
        threat_models, supervisions = zip(*[v.split("_", 1) for v in variant])
        if all(tm == "open" for tm in threat_models):
            tm_text = "Open"
        elif all(tm == "closed" for tm in threat_models):
            tm_text = "Closed"
        else:
            raise ValueError("Mixed threat models in variant list not supported for ROC-AUC plotting")
        sup_labels = ["Supervised" if sv == "supervised" else "Unsupervised" for sv in supervisions]
        sup_texts = "/".join(dict.fromkeys(sup_labels))
        mode_text = f"{tm_text}, {sup_texts}"
    else:
        threat_model, supervision = variant.split("_", 1)
        tm_text = "Open" if threat_model == "open" else "Closed"
        sup_text = "Supervised" if supervision == "supervised" else "Unsupervised"
        mode_text = f"{tm_text}, {sup_text}"

    styles = {
        "radioactive": {"color": "#4DBBD5", "marker": "o", "label": "Red-and-Green-List"},
        "ads": {"color": "#E64B35", "marker": "s", "label": "Antidistillation"},
    }

    plt.figure(figsize=(10, 6))
    something_plotted = False
    for wm_method in ["ads","radioactive"]:

        positive_points = buckets[wm_method]
        negative_points = buckets.get("control", [])

        if not positive_points or not negative_points:
            print(f"Insufficient experiment results found for {wm_method} under {variant} under {exp_dir}")
            continue
        
        something_plotted = True
        
        all_pos_labels = [label.split(" trial ")[0] for _, _, label in positive_points]
        assert len(set(all_pos_labels)) == 1, "All positive points must have the same label/method for ROC-AUC computation"
        pos_label = all_pos_labels[0]
        
        print(f"Computing ROC-AUC for {wm_method} under {variant} with {len(positive_points)} positive and {len(negative_points)} negative points")
        
        if balanced:
            assert len(positive_points) == len(negative_points), "For balanced ROC-AUC, need equal number of positive and negative points"
        
        # Compute ROC curve
        if threshold_style == "linear":
            thresholds = [i / min_num_thresholds for i in range(min_num_thresholds + 1)]
            print(f"Using {len(thresholds)} linear thresholds for ROC-AUC computation")
        elif threshold_style == "data-dependent":
            all_pvals = [pval for _, pval, _ in positive_points] + [pval for _, pval, _ in negative_points]
            all_pvals = sorted(set(all_pvals))
            assert all_pvals[0] >= 0.0 and all_pvals[-1] <= 1.0, "p-values must be in [0,1]"
            # add the bounds and the rhs most extreme
            if all_pvals[0] != 0.0:
                all_pvals = [0.0] + all_pvals 
            if all_pvals[-1] != 1.0:
                all_pvals = all_pvals + [1.0]
            all_pvals = all_pvals + [all_pvals[-1] + 1e-6]

            thresholds = all_pvals
            print(f"Using {len(thresholds)} data-dependent thresholds for ROC-AUC computation ranging from {thresholds[0]} to {thresholds[-1]}:\n{thresholds}")
        else:
            raise ValueError(f"Unrecognised threshold style: {threshold_style}")

        tprs = []
        fprs = []
        for thresh in thresholds:
            tp = sum(1 for _, pval, _ in positive_points if pval < thresh)
            fn = sum(1 for _, pval, _ in positive_points if pval >= thresh)
            fp = sum(1 for _, pval, _ in negative_points if pval < thresh)
            tn = sum(1 for _, pval, _ in negative_points if pval >= thresh)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tprs.append(tpr)
            fprs.append(fpr)
        # Compute AUC
        auc = 0.0
        for i in range(1, len(fprs)):
            auc += (fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1]) / 2
        # Plot ROC curve
        formatted_label = pos_label.replace("λ=", "λ = ").replace("δ=", "δ = ")
        if wm_method == "ads":
            legend_label = f"Antidistillation ({formatted_label}, AUC = {auc:.3f})"
        else:
            legend_label = f"Red-and-Green-List ({formatted_label}, AUC = {auc:.3f})"
        plt.plot(fprs, tprs, color=styles[wm_method]["color"], label=legend_label, linewidth=4)
        # Add total number of pos and neg points to legend
        # plt.plot([], [], ' ', label=f"{pos_label} Num Positives: {len(positive_points)}, Num Negatives: {len(negative_points)}")
    
    if not something_plotted:
        return []

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess", linewidth=3)
    
    plt.xlabel("False Positive Rate", fontsize=20, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=20, fontweight="bold")
    plt.title(f"{proxy_text}, {mode_text}", fontsize=22, fontweight="bold")
    plt.tick_params(axis="both", labelsize=18)
    legend = plt.legend(fontsize=18)
    if legend is not None:
        for text in legend.get_texts():
            if text.get_text().startswith("Antidistillation"):
                text.set_fontstyle("italic")
                text.set_fontweight("bold")
    out_dir = fig_dir or (exp_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix_str = "_".join(_suffix_parts(student_tag, student_tags, lr, epochs))
    if isinstance(variant, list):
        variant = "+".join(variant)
    name = f"stage5_plot_roc_auc_both_wms_under_{variant}"
    if suffix_str:
        name = f"{name}_{suffix_str}"
    output_paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    plt.tight_layout()
    for output_path in output_paths:
        plt.savefig(output_path)
    plt.close()
    return output_paths


def plot_pval_vs_empirical_fpr(
    exp_dir: Path,
    variant: str,
    *,
    student_tag: str | None = None,
    student_tags: set[str] | None = None,
    lr: str | None = None,
    epochs: str | None = None,
    fig_dir: Path | None = None,
    show_labels: bool = False,
) -> List[Path]:
    """
    This variation is to plot p-value vs empirical FPR (i.e., fraction of control models
    that have p-value below threshold).
    All we need is the negatives points, and then sweep an array of pvalue thresholds
    and compute the fraction of negatives that are above that threshold.
    """

    buckets, x_label = gather_points(
        exp_dir,
        variant,
        student_tag=student_tag,
        student_tags=student_tags,
        lr=lr,
        epochs=epochs,
    )
    negative_points = buckets.get("control", [])
    if not negative_points:
        raise RuntimeError(f"No control experiment results found for {variant} under {exp_dir}")
    # compute empirical FPRs at each unique p-value in negative points
    all_pvals = sorted(set([pval for _, pval, _ in negative_points]))
    fprs = []
    for thresh in all_pvals:
        fp = sum(1 for _, pval, _ in negative_points if pval < thresh)
        tn = sum(1 for _, pval, _ in negative_points if pval >= thresh)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fprs.append(fpr)
    plt.figure(figsize=(10, 6))
    plt.scatter(all_pvals, fprs, c="#1E8449", marker="D", alpha=0.8, label="Unfingerprinted", s=160)

    # fit a curve line through the points
    plt.plot(all_pvals, fprs, color="#1E8449", linestyle="--", linewidth=2)

    # plot the line corresponding to pval = FPR
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=2, label="p-value = Empirical FPR")

    # Keep legend focused on lines only.

    # limit x and y in [0,1]
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    if show_labels:
        for x, y in zip(all_pvals, fprs):
            plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=10)
    plt.xlabel("Fingerprint p-value", fontsize=20, fontweight="bold")
    plt.ylabel("Empirical False Positive Rate", fontsize=20, fontweight="bold")
    proxy_tag, _ = parse_exp_meta(exp_dir)
    inferred_student = student_tag or infer_student_tag(exp_dir, lr=lr, epochs=epochs)
    proxy_equal = proxy_tag is not None and inferred_student is not None and proxy_tag == inferred_student
    proxy_text = "Proxy = Student" if proxy_equal else "Proxy ≠ Student"
    threat_model, supervision = variant.split("_", 1)
    mode_text = f"{'Open' if threat_model == 'open' else 'Closed'}, {'Supervised' if supervision == 'supervised' else 'Unsupervised'}"
    plt.title(f"{proxy_text}, {mode_text}", fontsize=22, fontweight="bold")
    plt.tick_params(axis="both", labelsize=18)
    plt.legend(fontsize=18)
    out_dir = fig_dir or (exp_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix_str = "_".join(_suffix_parts(student_tag, student_tags, lr, epochs))
    name = f"stage5_plot_pval_vs_empirical_fpr_{variant}"
    if suffix_str:
        name = f"{name}_{suffix_str}"
    output_paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    plt.tight_layout()
    for output_path in output_paths:
        plt.savefig(output_path)
    plt.close()
    return output_paths
    


def plot_lm_eval(
    exp_dir: Path,
    variant: str,
    task_name: str = "gsm8k_cot",
    primary_metric: str = "exact_match,flexible-extract",
    *,
    student_tag: str | None = None,
    student_tags: set[str] | None = None,
    lr: str | None = None,
    epochs: str | None = None,
    fig_dir: Path | None = None,
    show_labels: bool = False,
) -> List[Path]:
    buckets, x_label = gather_lm_eval_points(
        exp_dir,
        variant,
        task_name,
        primary_metric,
        student_tag=student_tag,
        student_tags=student_tags,
        lr=lr,
        epochs=epochs,
    )
    points = buckets["radioactive"] + buckets["ads"] + buckets.get("control", [])
    if not points:
        raise RuntimeError(f"No experiment results found for {variant} under {exp_dir}")
    plt.figure(figsize=(10, 6))
    styles = {
        "radioactive": {"color": "#4DBBD5", "marker": "o", "label": "Red-and-Green-List"},
        "ads": {"color": "#E64B35", "marker": "s", "label": "Antidistillation"},
        "control": {"color": "#1E8449", "marker": "D", "label": "Unfingerprinted"},
    }
    for key, data in buckets.items():
        if not data:
            continue
        xs = [pt[0] for pt in data]
        ys = [pt[1] for pt in data]
        lbls = [pt[2] for pt in data]
        plt.scatter(xs, ys, c=styles[key]["color"], marker=styles[key]["marker"], alpha=0.8, label=styles[key]["label"], s=160)
        if show_labels:
            for x, y, txt in zip(xs, ys, lbls):
                plt.annotate(txt, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=10)
    plt.xlabel(x_label, fontsize=20, fontweight="bold")
    plt.ylabel("Fingerprint p-value", fontsize=20, fontweight="bold")
    proxy_tag, dataset = parse_exp_meta(exp_dir)
    inferred_student = student_tag or infer_student_tag(exp_dir, lr=lr, epochs=epochs)
    proxy_equal = proxy_tag is not None and inferred_student is not None and proxy_tag == inferred_student
    proxy_text = "Proxy = Student" if proxy_equal else "Proxy ≠ Student"
    mode_text = "Student Open-Weight" if variant.startswith("open") else "Student Closed-Weight"
    dataset_text = dataset.upper() if dataset else "DATASET"
    plt.title(f"{proxy_text}, {mode_text} ({dataset_text})", fontsize=22, fontweight="bold")
    plt.tick_params(axis="both", labelsize=18)
    plt.yscale("log")
    plt.axhline(0.05, color="black", linestyle="--", linewidth=2, label="p = 0.05")

    legend = plt.legend(fontsize=18)
    for text in legend.get_texts():
        if text.get_text() == "Antidistillation":
            text.set_fontstyle("italic")
            text.set_fontweight("bold")
    out_dir = fig_dir or (exp_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix_str = "_".join(_suffix_parts(student_tag, student_tags, lr, epochs))
    name = f"stage5_plot_lm_eval_{variant}_{task_name}_{primary_metric}"
    if suffix_str:
        name = f"{name}_{suffix_str}"
    output_paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    plt.tight_layout()
    for output_path in output_paths:
        plt.savefig(output_path)
    plt.close()
    return output_paths


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 5 – plotting")
    parser.add_argument("--exp-dir", type=Path, required=True, help="Base experiment directory")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["open_supervised", "open_unsupervised", "closed_supervised", "closed_unsupervised"],
        choices=["open_supervised", "open_unsupervised", "closed_supervised", "closed_unsupervised"],
    )
    parser.add_argument(
        "--lm-eval-tasks",
        nargs="+",
        default=["gsm8k_cot"],
        choices=["gsm8k_cot"],
        help="LM eval tasks to plot. Note mode complex logic needed if not all tasks have same primary metrics.",
    )
    parser.add_argument(
        "--lm-eval-metric",
        type=str,
        default=["exact_match,strict-match","exact_match,flexible-extract"],
        help="LM eval primary metric to extract from results jsons.",
    )
    parser.add_argument("--student-tag", type=str, default=None, help="Filter metrics to this student tag")
    parser.add_argument("--lr", type=str, default=None, help="Filter metrics to this lr tag (as encoded in folder name)")
    parser.add_argument("--epochs", type=str, default=None, help="Filter metrics to this epoch count")
    parser.add_argument("--fig-dir", type=Path, default=None, help="Output directory for figures (default: exp_dir/figures)")
    parser.add_argument("--show-labels", action="store_true", help="Annotate points with delta/lambda labels")
    args = parser.parse_args(argv)

    fig_dir = args.fig_dir or (args.exp_dir / "figures")

    if args.student_tag:
        groups = [(args.student_tag, None)]
    else:
        proxy_tag, _ = parse_exp_meta(args.exp_dir)
        all_students = collect_student_tags(args.exp_dir, lr=args.lr, epochs=args.epochs)
        groups = []
        if proxy_tag and proxy_tag in all_students:
            groups.append((proxy_tag, None))
        nonproxy = {tag for tag in all_students if not proxy_tag or tag != proxy_tag}
        if nonproxy:
            groups.append((None, nonproxy))
        if not groups:
            groups = [(None, None)]

    for student_tag, student_tags in groups:
        for variant in args.variants:
            results = plot_pval_vs_empirical_fpr(
                args.exp_dir,
                variant,
                student_tag=student_tag,
                student_tags=student_tags,
                lr=args.lr,
                epochs=args.epochs,
                fig_dir=fig_dir,
                show_labels=args.show_labels,
            )
            print(f"Wrote pval vs empirical fpr plot to {results}")

        for variant in args.variants:
            results = plot_roc_auc(
                args.exp_dir,
                variant,
                student_tag=student_tag,
                student_tags=student_tags,
                lr=args.lr,
                epochs=args.epochs,
                fig_dir=fig_dir,
                show_labels=args.show_labels,
            )
            print(f"Wrote roc auc plot to {results}")

        for variant in args.variants:
            results = plot(
                args.exp_dir,
                variant,
                student_tag=student_tag,
                student_tags=student_tags,
                lr=args.lr,
                epochs=args.epochs,
                fig_dir=fig_dir,
                show_labels=args.show_labels,
            )
            print(f"Wrote plot to {results}")

        for variant in args.variants:
            for task_name in args.lm_eval_tasks:
                for primary_metric in args.lm_eval_metric:
                    try:
                        lm_eval_result = plot_lm_eval(
                            args.exp_dir,
                            variant,
                            task_name=task_name,
                            primary_metric=primary_metric,
                            student_tag=student_tag,
                            student_tags=student_tags,
                            lr=args.lr,
                            epochs=args.epochs,
                            fig_dir=fig_dir,
                        )
                        print(f"Wrote lm-eval plot to {lm_eval_result}")
                    except RuntimeError as e:
                        print(f"Skipping lm-eval plot for {variant} on {task_name} with metric {primary_metric}: {e}")


if __name__ == "__main__":
    main()
