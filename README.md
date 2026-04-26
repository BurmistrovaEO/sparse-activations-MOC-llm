# Sparse Activations (MoC) for LLM Fine-Tuning

Репозиторий содержит прототип спарсификации MLP-блоков в Llama и воспроизводимый пайплайн LoRA fine-tuning для сравнения:
- baseline (без спарсификации),
- sparse (с заменой `LlamaMLP -> SparseMLP`).

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Если используется gated-модель (например, Llama), нужен токен:

```bash
export HF_TOKEN=...
huggingface-cli login --token "$HF_TOKEN"
```

## Python-first запуск экспериментов

Основная точка входа: `run_experiments.py`.

Скрипт запускает эксперименты программно (без CLI-конфигурации), используя API:
- `run_train(config: TrainConfig)` из `train_finetune.py`
- `run_lm_eval(config: EvalConfig)` из `eval_lm_eval.py`

### Быстрый запуск debug набора из Python

```python
from run_experiments import run_suite

run_suite(
    suite_name="debug",
    model_name_or_path="meta-llama/Llama-3.2-3B",
    output_root="outputs/experiments",
)

run_suite(
    suite_name="ablation",
    model_name_or_path="meta-llama/Llama-3.2-3B",
    output_root="outputs/experiments",
)
```

### Что входит в наборы

- `debug`: короткие baseline + sparse прогоны для smoke-check.
- `ablation`: `baseline_pair`, `sparse_k_2048`, `sparse_k_4096`, `sparse_k_6144`, `sparse_layers_8_11`.

### Артефакты

- Метрики train каждого эксперимента: `<output_dir>/train_metrics.json`
- Конфиг train: `<output_dir>/train_config.json`
- LM Eval результаты: `<output_dir>/lm_eval.json`
- Итоговый агрегированный отчёт по suite: `outputs/experiments/<suite>_summary.json`

## Python API (для своих скриптов)

- `TrainConfig`, `EvalConfig`: `experiment_configs.py`
- `run_train(config)`: `train_finetune.py`
- `run_lm_eval(config)`: `eval_lm_eval.py`

### CUDA T4 / low-VRAM поведение

В `run_train(...)` добавлена авто-адаптация для CUDA GPU с небольшой памятью (включая T4):
- автоматически включается `gradient_checkpointing=True`,
- `seq_len` ограничивается до `256`,
- `batch_size` и `eval_batch_size` ограничиваются до `1`,
- устанавливается `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
- отключается `model.config.use_cache` в training.

Для MPS этот авто-режим не применяется.

## Рекомендуемый порядок экспериментов

1. Прогнать suite `debug`.
2. Прогнать suite `ablation`.
3. Выровнять гиперпараметры и длину контекста для парного сравнения.
4. Сравнить `*_summary.json` и train/eval артефакты.

