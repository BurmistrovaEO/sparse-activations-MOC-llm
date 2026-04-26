import gc
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from experiment_configs import TrainConfig
from io_utils import save_json
from model_utils import build_model, resolve_device, select_dtype


@dataclass
class RunMetrics:
    mode: str
    model_name_or_path: str
    device: str
    train_runtime_sec: float
    train_samples_per_second: float | None
    train_steps_per_second: float | None
    tokens_per_second_approx: float | None
    eval_loss: float | None
    eval_perplexity: float | None
    peak_memory_bytes: int | None


class ThroughputAndMemoryCallback(TrainerCallback):
    def __init__(
        self, seq_len: int, per_device_train_batch_size: int, grad_accum: int, device_type: str
    ):
        self.seq_len = seq_len
        self.per_device_train_batch_size = per_device_train_batch_size
        self.grad_accum = grad_accum
        self.device_type = device_type
        self.peak_memory_bytes: int | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        if self.device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        elif self.device_type == "mps" and torch.backends.mps.is_available():
            # No explicit reset API on MPS; best effort.
            pass

    def on_train_end(self, args, state, control, **kwargs):
        if self.device_type == "cuda" and torch.cuda.is_available():
            self.peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        elif self.device_type == "mps" and torch.backends.mps.is_available():
            self.peak_memory_bytes = int(torch.mps.current_allocated_memory())

    def approx_tokens_per_second(self, train_steps_per_second: float | None) -> float | None:
        if train_steps_per_second is None:
            return None
        return (
            train_steps_per_second
            * self.per_device_train_batch_size
            * self.grad_accum
            * self.seq_len
        )


def prepare_model(config: TrainConfig, dtype: torch.dtype) -> AutoModelForCausalLM:
    return build_model(
        mode=config.mode,
        model_name_or_path=config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        dtype=dtype,
        sparse_k=config.sparse_k,
        sparse_layers=config.sparse_layers,
    )


def attach_lora(model: AutoModelForCausalLM, config: TrainConfig) -> AutoModelForCausalLM:
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def _chunk_tokens(token_sequences: list[list[int]], block_size: int) -> dict[str, list[list[int]]]:
    concatenated: list[int] = []
    for sequence in token_sequences:
        concatenated.extend(sequence)
    total_len = (len(concatenated) // block_size) * block_size
    chunks = [concatenated[i : i + block_size] for i in range(0, total_len, block_size)]
    return {"input_ids": chunks, "labels": [chunk.copy() for chunk in chunks]}


def build_dataset(config: TrainConfig, tokenizer: AutoTokenizer) -> DatasetDict:
    ds = load_dataset(config.dataset_name, config.dataset_config_name)
    if config.train_split not in ds:
        raise ValueError(f"Train split '{config.train_split}' does not exist in dataset.")
    if config.eval_split not in ds:
        raise ValueError(f"Eval split '{config.eval_split}' does not exist in dataset.")

    train_ds = ds[config.train_split]
    eval_ds = ds[config.eval_split]

    if config.max_train_samples > 0:
        train_ds = train_ds.select(range(min(config.max_train_samples, len(train_ds))))
    if config.max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(config.max_eval_samples, len(eval_ds))))

    if config.debug_subset:
        train_ds = train_ds.select(range(min(256, len(train_ds))))
        eval_ds = eval_ds.select(range(min(128, len(eval_ds))))

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        texts = [t for t in batch[config.text_column] if t and t.strip()]
        tokenized = tokenizer(texts, add_special_tokens=True, truncation=False)
        return {"input_ids": tokenized["input_ids"]}

    train_tok = train_ds.map(tokenize_batch, batched=True, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(tokenize_batch, batched=True, remove_columns=eval_ds.column_names)

    train_grouped = train_tok.map(
        lambda batch: _chunk_tokens(batch["input_ids"], config.seq_len),
        batched=True,
        remove_columns=train_tok.column_names,
    )
    eval_grouped = eval_tok.map(
        lambda batch: _chunk_tokens(batch["input_ids"], config.seq_len),
        batched=True,
        remove_columns=eval_tok.column_names,
    )
    return DatasetDict({"train": train_grouped, "validation": eval_grouped})


def build_training_args(config: TrainConfig, device: torch.device) -> TrainingArguments:
    is_cuda = device.type == "cuda"
    return TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.grad_accum,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        do_eval=True,
        report_to="none",
        fp16=is_cuda and not torch.cuda.is_bf16_supported(),
        bf16=is_cuda and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=config.gradient_checkpointing,
        remove_unused_columns=False,
    )


def adapt_config_for_device(config: TrainConfig, device: torch.device) -> TrainConfig:
    runtime_config = config
    if device.type == "cuda" and torch.cuda.is_available():
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.backends.cuda.matmul.allow_tf32 = True

        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        is_low_vram_gpu = total_vram_gb <= 16.5
        if is_low_vram_gpu:
            runtime_config = replace(
                runtime_config,
                seq_len=min(runtime_config.seq_len, 256),
                batch_size=min(runtime_config.batch_size, 1),
                eval_batch_size=min(runtime_config.eval_batch_size, 1),
                gradient_checkpointing=True,
            )
            print(
                f"[CUDA low-VRAM mode] Detected {total_vram_gb:.1f}GB GPU, "
                f"using seq_len={runtime_config.seq_len}, "
                f"batch_size={runtime_config.batch_size}, "
                f"eval_batch_size={runtime_config.eval_batch_size}, "
                "gradient_checkpointing=True"
            )
    return runtime_config


def run_train(config: TrainConfig) -> dict[str, float | int | str | None]:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    device = resolve_device(config.device)
    if device.type == "cuda" and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    runtime_config = adapt_config_for_device(config, device)
    dtype = select_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(runtime_config.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model(runtime_config, dtype)
    model = attach_lora(model, runtime_config)
    # Reduces activation memory during training.
    model.config.use_cache = False
    model.to(device=device, dtype=dtype)

    datasets = build_dataset(runtime_config, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = build_training_args(runtime_config, device)

    perf_callback = ThroughputAndMemoryCallback(
        seq_len=runtime_config.seq_len,
        per_device_train_batch_size=runtime_config.batch_size,
        grad_accum=runtime_config.grad_accum,
        device_type=device.type,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        callbacks=[perf_callback],
    )

    train_start = time.time()
    train_result = trainer.train()
    train_runtime = time.time() - train_start
    eval_metrics = trainer.evaluate()

    eval_loss = eval_metrics.get("eval_loss")
    eval_perplexity = math.exp(eval_loss) if eval_loss is not None else None
    train_samples_per_second = train_result.metrics.get("train_samples_per_second")
    train_steps_per_second = train_result.metrics.get("train_steps_per_second")

    trainer.save_model(runtime_config.output_dir)
    tokenizer.save_pretrained(runtime_config.output_dir)

    metrics = RunMetrics(
        mode=runtime_config.mode,
        model_name_or_path=runtime_config.model_name_or_path,
        device=str(device),
        train_runtime_sec=train_runtime,
        train_samples_per_second=train_samples_per_second,
        train_steps_per_second=train_steps_per_second,
        tokens_per_second_approx=perf_callback.approx_tokens_per_second(train_steps_per_second),
        eval_loss=eval_loss,
        eval_perplexity=eval_perplexity,
        peak_memory_bytes=perf_callback.peak_memory_bytes,
    )

    metrics_dict = asdict(metrics)
    save_json(runtime_config.report_path, metrics_dict)
    save_json(os.path.join(runtime_config.output_dir, "train_config.json"), asdict(runtime_config))
    return metrics_dict
