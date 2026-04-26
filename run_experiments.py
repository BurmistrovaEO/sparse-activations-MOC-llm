from __future__ import annotations

import gc
from dataclasses import asdict, replace
from pathlib import Path

import torch

from eval_lm_eval import run_lm_eval
from experiment_configs import EvalConfig, TrainConfig
from io_utils import ensure_dir, save_json
from train_finetune import run_train


def _experiment_paths(output_root: str, suite_name: str, run_name: str) -> tuple[str, str, str]:
    run_dir = f"{output_root}/{suite_name}/{run_name}"
    return run_dir, f"{run_dir}/train_metrics.json", f"{run_dir}/lm_eval.json"


def _build_train_config(
    base: TrainConfig, output_root: str, suite_name: str, run_name: str, **overrides
) -> TrainConfig:
    output_dir, report_path, _ = _experiment_paths(output_root, suite_name, run_name)
    return replace(base, output_dir=output_dir, report_path=report_path, **overrides)


def _build_eval_config(
    *,
    model_name_or_path: str,
    mode: str,
    output_root: str,
    suite_name: str,
    run_name: str,
    adapter_path: str,
    tasks: list[str],
    sparse_k: int = 4096,
    sparse_layers: list[int] | None = None,
) -> EvalConfig:
    _, _, output_path = _experiment_paths(output_root, suite_name, run_name)
    return EvalConfig(
        model_name_or_path=model_name_or_path,
        mode=mode,
        sparse_k=sparse_k,
        sparse_layers=sparse_layers,
        adapter_path=adapter_path,
        tasks=tasks,
        output_path=output_path,
    )


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

    return {
        "name": name,
        "train_config": asdict(train_config),
        "train_metrics": train_metrics,
        "eval_config": asdict(eval_config) if eval_config else None,
        "eval_results_path": eval_config.output_path if eval_config else None,
        "eval_task_count": len(eval_results.get("results", {})) if eval_results else 0,
    }


def build_debug_suite(
    output_root: str, model_name_or_path: str
) -> list[tuple[str, TrainConfig, EvalConfig]]:
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

    baseline = _build_train_config(base, output_root, "debug", "baseline", mode="baseline")
    sparse = _build_train_config(base, output_root, "debug", "sparse", mode="sparse", sparse_k=4096)

    baseline_eval = _build_eval_config(
        model_name_or_path=model_name_or_path,
        mode="baseline",
        output_root=output_root,
        suite_name="debug",
        run_name="baseline",
        adapter_path=baseline.output_dir,
        tasks=["arc_easy"],
    )
    sparse_eval = _build_eval_config(
        model_name_or_path=model_name_or_path,
        mode="sparse",
        output_root=output_root,
        suite_name="debug",
        run_name="sparse",
        adapter_path=sparse.output_dir,
        tasks=["arc_easy"],
        sparse_k=4096,
    )
    return [("debug_baseline", baseline, baseline_eval), ("debug_sparse", sparse, sparse_eval)]


def build_ablation_suite(
    output_root: str, model_name_or_path: str
) -> list[tuple[str, TrainConfig, EvalConfig]]:
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
    baseline = _build_train_config(base, output_root, "ablation", "baseline_pair", mode="baseline")
    runs.append(
        (
            "baseline_pair",
            baseline,
            _build_eval_config(
                model_name_or_path=model_name_or_path,
                mode="baseline",
                output_root=output_root,
                suite_name="ablation",
                run_name="baseline_pair",
                adapter_path=baseline.output_dir,
                tasks=["arc_easy", "hellaswag"],
            ),
        )
    )

    for k in (2048, 4096, 6144):
        run_name = f"sparse_k_{k}"
        train_cfg = _build_train_config(
            base,
            output_root,
            "ablation",
            run_name,
            mode="sparse",
            sparse_k=k,
        )
        eval_cfg = _build_eval_config(
            model_name_or_path=model_name_or_path,
            mode="sparse",
            sparse_k=k,
            output_root=output_root,
            suite_name="ablation",
            run_name=run_name,
            adapter_path=train_cfg.output_dir,
            tasks=["arc_easy", "hellaswag"],
        )
        runs.append((run_name, train_cfg, eval_cfg))

    layer_run_name = "sparse_layers_8_11"
    layer_run = _build_train_config(
        base,
        output_root,
        "ablation",
        layer_run_name,
        mode="sparse",
        sparse_k=4096,
        sparse_layers=[8, 9, 10, 11],
    )
    runs.append(
        (
            layer_run_name,
            layer_run,
            _build_eval_config(
                model_name_or_path=model_name_or_path,
                mode="sparse",
                sparse_k=4096,
                sparse_layers=[8, 9, 10, 11],
                output_root=output_root,
                suite_name="ablation",
                run_name=layer_run_name,
                adapter_path=layer_run.output_dir,
                tasks=["arc_easy", "hellaswag"],
            ),
        )
    )
    return runs


def run_suite(
    suite_name: str,
    model_name_or_path: str = "meta-llama/Llama-3.2-3B",
    output_root: str = "outputs/experiments",
) -> dict:
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
    save_json(summary_path, summary)
    print(f"[DONE] Summary saved to {summary_path}")
    return summary
