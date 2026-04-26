from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BaseModelConfig:
    model_name_or_path: str = "meta-llama/Llama-3.2-3B"
    mode: str = "baseline"
    device: str = "auto"
    sparse_k: int = 4096
    sparse_layers: list[int] | None = None
    trust_remote_code: bool = False


@dataclass
class TrainConfig(BaseModelConfig):
    output_dir: str = "outputs/finetune"
    report_path: str = "outputs/finetune/run_metrics.json"

    dataset_name: str = "wikitext"
    dataset_config_name: str = "wikitext-2-raw-v1"
    train_split: str = "train"
    eval_split: str = "validation"
    text_column: str = "text"
    seq_len: int = 1024
    max_train_samples: int = 5000
    max_eval_samples: int = 1000
    debug_subset: bool = False

    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_train_epochs: float = 1.0
    max_steps: int = -1
    batch_size: int = 1
    eval_batch_size: int = 1
    grad_accum: int = 8
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    gradient_checkpointing: bool = False

    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    )


@dataclass
class EvalConfig(BaseModelConfig):
    adapter_path: str | None = None
    tasks: list[str] = field(default_factory=lambda: ["arc_easy"])
    batch_size: int = 4
    num_fewshot: int = 0
    limit: float | None = None
    output_path: str = "outputs/lm_eval/results.json"
