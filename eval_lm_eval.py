import json
from pathlib import Path

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from peft import PeftModel
from transformers import AutoTokenizer

from experiment_configs import EvalConfig
from model_utils import build_model, resolve_device_for_lm_eval, select_dtype


def run_lm_eval(config: EvalConfig) -> dict:
    device = resolve_device_for_lm_eval(config.device)
    device_for_dtype = torch.device("cuda") if device.startswith("cuda") else torch.device(device)
    dtype = select_dtype(device_for_dtype)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
    model = build_model(
        mode=config.mode,
        model_name_or_path=config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        dtype=dtype,
        sparse_k=config.sparse_k,
        sparse_layers=config.sparse_layers,
    )

    if config.adapter_path:
        model = PeftModel.from_pretrained(model, config.adapter_path)
        model = model.merge_and_unload()

    lm = HFLM(pretrained=model, tokenizer=tokenizer, device=device, batch_size=config.batch_size)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=config.tasks,
        num_fewshot=config.num_fewshot,
        batch_size=config.batch_size,
        limit=config.limit,
    )

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results
