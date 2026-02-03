"""Experiment- and stage-level configuration dataclasses and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

EvalMode = Literal["open", "closed"]
SupervisionMode = Literal["supervised", "unsupervised"]
WatermarkMethod = Literal["radioactive", "ads", "control"]
DatasetName = Literal["gsm8k", "oasst1"]


@dataclass(slots=True)
class ModelSpec:
    """Basic information for loading a model/tokenizer pair."""

    name: str
    dtype: str = "bfloat16"
    pad_token: str | None = None


@dataclass(slots=True)
class ExperimentLayout:
    """Resolve experiment- and method-specific directories."""

    root: Path
    dataset: DatasetName
    num_examples: int
    teacher_model: str
    proxy_model: str

    def __post_init__(self) -> None:
        """Validate required fields after initialization.

        Args:
            None (uses dataclass fields on self).

        Returns:
            None. Raises ValueError if num_examples is not positive.
        """
        if self.num_examples <= 0:
            raise ValueError("num_examples must be positive")

    @staticmethod
    def _abbrev(model_name: str) -> str:
        """Return a filesystem-safe model name abbreviation.

        Args:
            model_name: Full model identifier (e.g., Hugging Face repo name).

        Returns:
            Short, lowercase, alphanumeric tag suitable for directory names.
        """
        parts = model_name.split("/")
        candidate = parts[-1] if parts else model_name
        candidate = candidate.lower()
        if "deepseek-r1-distill" in candidate:
            candidate = "r1-distill-7b"
        allowed = [ch for ch in candidate if ch.isalnum() or ch in {".", "-"}]
        collapsed = "".join(allowed)
        return collapsed or "model"

    @property
    def teacher_abbrev(self) -> str:
        """Abbreviated tag for the teacher model.

        Returns:
            Lowercase, filesystem-safe model tag.
        """
        return self._abbrev(self.teacher_model)

    @property
    def proxy_abbrev(self) -> str:
        """Abbreviated tag for the proxy model.

        Returns:
            Lowercase, filesystem-safe model tag.
        """
        return self._abbrev(self.proxy_model)

    def experiment_dir(self) -> Path:
        """Experiment directory derived from model names, dataset, and size.

        Returns:
            Path to the experiment root directory.
        """
        name = f"{self.teacher_abbrev}_{self.proxy_abbrev}_{self.dataset}_n{self.num_examples}"
        return self.root / name

    def hash_dir(self) -> Path:
        """Directory containing the shared hash configuration.

        Returns:
            Path to the hash_seed directory.
        """
        return self.experiment_dir() / "hash_seed"

    def hash_path(self) -> Path:
        """Path to the hash_config.json file.

        Returns:
            Full path to the hash config JSON file.
        """
        return self.hash_dir() / "hash_config.json"

    def trace_dir(self, kind: Literal["training", "alternative"], method_label: str) -> Path:
        """Directory holding trace files for a given method and split.

        Args:
            kind: "training" or "alternative".
            method_label: Method label (e.g., "radioactive-delta2").

        Returns:
            Path to the trace directory for that method/split.
        """
        base = "training_traces" if kind == "training" else "alternative_traces"
        return self.experiment_dir() / base / method_label

    def model_dir(self, student_model: str, method_label: str, lr: float, epochs: int) -> Path:
        """Directory for the student model's LoRA outputs.

        Args:
            student_model: Student model identifier.
            method_label: Method label (e.g., "ads-lambda16").
            lr: Learning rate used in fine-tuning.
            epochs: Number of training epochs.

        Returns:
            Path to the model output directory.
        """
        student_abbrev = self._abbrev(student_model)
        lr_tag = f"{lr:g}"
        return self.experiment_dir() / "models" / f"{student_abbrev}_{method_label}_lr{lr_tag}_e{epochs}"

    def metrics_dir(self, student_model: str, method_label: str, lr: float, epochs: int) -> Path:
        """Directory for evaluation metrics for a given training run.

        Args:
            student_model: Student model identifier.
            method_label: Method label (e.g., "control").
            lr: Learning rate used in fine-tuning.
            epochs: Number of training epochs.

        Returns:
            Path to the metrics directory.
        """
        student_abbrev = self._abbrev(student_model)
        lr_tag = f"{lr:g}"
        return self.experiment_dir() / "metrics" / f"{student_abbrev}_{method_label}_lr{lr_tag}_e{epochs}"


@dataclass(slots=True)
class HashStageConfig:
    """Inputs for Stage 0."""

    teacher: ModelSpec
    exp_dir: Path
    seed: int | None = None
    gamma: float = 0.5
    output_file: Path | None = None

    def resolved_output(self) -> Path:
        """Resolve the destination path for hash_config.json.

        Returns:
            Path where the hash configuration should be written.
        """
        if self.output_file is not None:
            return self.output_file
        return self.exp_dir / "hash_config.json"


@dataclass(slots=True)
class GenerationConfig:
    """Inputs shared by radioactive/ADS teacher generation."""

    dataset: DatasetName
    split: str
    max_examples: int
    teacher: ModelSpec
    proxy: ModelSpec
    method: WatermarkMethod
    delta: float | None = None
    lam: float | None = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    batch_size: int = 16
    seed: int = 42
    output_jsonl: Path = field(default_factory=lambda: Path("traces.jsonl"))
    metadata_path: Path = field(default_factory=lambda: Path("traces_metadata.json"))

    def strength_label(self) -> str:
        """Return the method-specific strength label.

        Returns:
            String label such as "delta2" or "lambda16" or "control".

        Raises:
            ValueError: If required strength parameters are missing or unsupported.
        """
        if self.method == "radioactive":
            if self.delta is None:
                raise ValueError("delta must be set for radioactive runs")
            return f"delta{self.delta:g}".replace(".", "_")
        if self.method == "ads":
            if self.lam is None:
                raise ValueError("lam must be set for ads runs")
            return f"lambda{self.lam:g}".replace(".", "_")
        if self.method == "control":
            return "control"
        raise ValueError(f"Unsupported method: {self.method}")


@dataclass(slots=True)
class TeacherEvalConfig:
    dataset: DatasetName
    teacher: ModelSpec
    traces_jsonl: Path
    output_path: Path
    batch_size: int = 8
    max_answer_tokens: int = 32
    seed: int = 42


@dataclass(slots=True)
class FinetuneConfig:
    dataset: DatasetName
    student: ModelSpec
    traces_jsonl: Path
    output_dir: Path
    epochs: int
    batch_size: int = 1
    grad_accum: int = 4
    learning_rate: float = 2e-5
    max_seq_length: int = 4096
    rank: int = 128
    alpha: int = 128
    dropout: float = 0.05
    seed: int = 42


@dataclass(slots=True)
class WatermarkEvalConfig:
    dataset: DatasetName
    teacher: ModelSpec
    student: ModelSpec
    hash_config: Path
    traces_jsonl: Path
    lora_dir: Path
    mode: EvalMode
    supervision: SupervisionMode
    output_path: Path
    batch_size: int = 4
    seed: int = 42
    mask_chunk: int = 64
