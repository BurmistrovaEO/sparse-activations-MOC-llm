# Sparse Activations (MoC) for LLM Fine-Tuning

Репозиторий сравнивает два режима fine-tuning для Llama-подобных моделей:
- `baseline`: стандартный MLP;
- `sparse`: замена `LlamaMLP -> SparseMLP` с top-k активациями.

Основной фокус: воспроизводимый Python-пайплайн для train + lm-eval и сохранение метрик в JSON.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для gated-моделей нужен Hugging Face токен:

```bash
export HF_TOKEN=...
huggingface-cli login --token "$HF_TOKEN"
```

## Main Entry Point

Запуск серий экспериментов выполняется через `run_experiments.py`:

```python
from run_experiments import run_suite

run_suite("debug", model_name_or_path="meta-llama/Llama-3.2-3B", output_root="outputs/experiments")
run_suite("ablation", model_name_or_path="meta-llama/Llama-3.2-3B", output_root="outputs/experiments")
```

## Experiment Suites

- `debug`: короткие baseline/sparse прогоны для smoke-check.
- `ablation`:
  - `baseline_pair`
  - `sparse_k_2048`
  - `sparse_k_4096`
  - `sparse_k_6144`
  - `sparse_layers_8_11`

## Output Artifacts

Для каждого запуска:
- `<output_dir>/train_metrics.json`
- `<output_dir>/train_config.json`
- `<output_dir>/lm_eval.json`

Для всей серии:
- `outputs/experiments/<suite>_summary.json`

## Python API

- Конфиги: `TrainConfig`, `EvalConfig` в `experiment_configs.py`
- Train: `run_train(config)` в `train_finetune.py`
- Eval: `run_lm_eval(config)` в `eval_lm_eval.py`

## Low-VRAM CUDA Behavior

В `run_train(...)` при CUDA GPU с памятью <= 16.5 GB автоматически включается low-VRAM профиль:
- `gradient_checkpointing=True`
- `seq_len <= 256`
- `batch_size <= 1`
- `eval_batch_size <= 1`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `model.config.use_cache=False`

Для MPS этот режим не применяется.

## Recommended Workflow

1. Запустить `debug`.
2. Запустить `ablation`.
3. Сравнить `*_summary.json` и артефакты train/eval.

