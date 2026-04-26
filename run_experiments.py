from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
import gc

import torch

from experiment_configs import EvalConfig, TrainConfig
from eval_lm_eval import run_lm_eval
from train_finetune import run_train


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def cleanup_accelerator_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_experiment(name: str, train_config: TrainConfig, eval_config: EvalConfig | None) -> dict:
    print(f"[RUN] {name}")
    cleanup_accelerator_memory()
    train_metrics = run_train(train_config)
    eval_results = None

    if eval_config is not None:
        cleanup_accelerator_memory()
        eval_results = run_lm_eval(eval_config)
    cleanup_accelerator_memory()

    result = {
        "name": name,
        "train_config": asdict(train_config),
        "train_metrics": train_metrics,
        "eval_config": asdict(eval_config) if eval_config else None,
        "eval_results_path": eval_config.output_path if eval_config else None,
        "eval_task_count": len(eval_results.get("results", {})) if eval_results else 0,
    }
    return result


def build_debug_suite(output_root: str, model_name_or_path: str) -> list[tuple[str, TrainConfig, EvalConfig]]:
    base = TrainConfig(
        model_name_or_path=model_name_or_path,
        device="auto",
        debug_subset=True,
        seq_len=512,
        max_steps=20,
        batch_size=1,
        eval_batch_size=1,
        grad_accum=4,
        eval_steps=10,
        save_steps=10,
    )

    baseline = replace(
        base,
        mode="baseline",
        output_dir=f"{output_root}/debug/baseline",
        report_path=f"{output_root}/debug/baseline/train_metrics.json",
    )
    sparse = replace(
        base,
        mode="sparse",
        sparse_k=4096,
        output_dir=f"{output_root}/debug/sparse",
        report_path=f"{output_root}/debug/sparse/train_metrics.json",
    )

    baseline_eval = EvalConfig(
        model_name_or_path=model_name_or_path,
        mode="baseline",
        adapter_path=baseline.output_dir,
        tasks=["arc_easy"],
        output_path=f"{output_root}/debug/baseline/lm_eval.json",
    )
    sparse_eval = EvalConfig(
        model_name_or_path=model_name_or_path,
        mode="sparse",
        sparse_k=4096,
        adapter_path=sparse.output_dir,
        tasks=["arc_easy"],
        output_path=f"{output_root}/debug/sparse/lm_eval.json",
    )
    return [("debug_baseline", baseline, baseline_eval), ("debug_sparse", sparse, sparse_eval)]


def build_ablation_suite(output_root: str, model_name_or_path: str) -> list[tuple[str, TrainConfig, EvalConfig]]:
    base = TrainConfig(
        model_name_or_path=model_name_or_path,
        device="auto",
        debug_subset=False,
        seq_len=1024,
        max_steps=200,
        batch_size=1,
        eval_batch_size=1,
        grad_accum=8,
        eval_steps=50,
        save_steps=50,
    )

    runs: list[tuple[str, TrainConfig, EvalConfig]] = []
    baseline = replace(
        base,
        mode="baseline",
        output_dir=f"{output_root}/ablation/baseline_pair",
        report_path=f"{output_root}/ablation/baseline_pair/train_metrics.json",
    )
    runs.append(
        (
            "baseline_pair",
            baseline,
            EvalConfig(
                model_name_or_path=model_name_or_path,
                mode="baseline",
                adapter_path=baseline.output_dir,
                tasks=["arc_easy", "hellaswag"],
                output_path=f"{output_root}/ablation/baseline_pair/lm_eval.json",
            ),
        )
    )

    for k in (2048, 4096, 6144):
        train_cfg = replace(
            base,
            mode="sparse",
            sparse_k=k,
            output_dir=f"{output_root}/ablation/sparse_k_{k}",
            report_path=f"{output_root}/ablation/sparse_k_{k}/train_metrics.json",
        )
        eval_cfg = EvalConfig(
            model_name_or_path=model_name_or_path,
            mode="sparse",
            sparse_k=k,
            adapter_path=train_cfg.output_dir,
            tasks=["arc_easy", "hellaswag"],
            output_path=f"{output_root}/ablation/sparse_k_{k}/lm_eval.json",
        )
        runs.append((f"sparse_k_{k}", train_cfg, eval_cfg))

    layer_run = replace(
        base,
        mode="sparse",
        sparse_k=4096,
        sparse_layers=[8, 9, 10, 11],
        output_dir=f"{output_root}/ablation/sparse_layers_8_11",
        report_path=f"{output_root}/ablation/sparse_layers_8_11/train_metrics.json",
    )
    runs.append(
        (
            "sparse_layers_8_11",
            layer_run,
            EvalConfig(
                model_name_or_path=model_name_or_path,
                mode="sparse",
                sparse_k=4096,
                sparse_layers=[8, 9, 10, 11],
                adapter_path=layer_run.output_dir,
                tasks=["arc_easy", "hellaswag"],
                output_path=f"{output_root}/ablation/sparse_layers_8_11/lm_eval.json",
            ),
        )
    )
    return runs


def run_suite(suite_name: str, model_name_or_path: str = "meta-llama/Llama-3.2-3B", output_root: str = "outputs/experiments") -> dict:
    ensure_dir(output_root)
    if suite_name == "debug":
        plan = build_debug_suite(output_root=output_root, model_name_or_path=model_name_or_path)
    elif suite_name == "ablation":
        plan = build_ablation_suite(output_root=output_root, model_name_or_path=model_name_or_path)
    else:
        raise ValueError(f"Unknown suite_name='{suite_name}'. Use 'debug' or 'ablation'.")

    runs = []
    for name, train_config, eval_config in plan:
        runs.append(run_experiment(name=name, train_config=train_config, eval_config=eval_config))

    summary = {"suite": suite_name, "model_name_or_path": model_name_or_path, "runs": runs}
    summary_path = Path(output_root) / f"{suite_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Summary saved to {summary_path}")
    return summary

