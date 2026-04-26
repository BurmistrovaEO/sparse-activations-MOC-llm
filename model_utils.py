from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP

from sparsification import SparseMLP, replace_nested_module

SUPPORTED_MODES = {"baseline", "sparse"}


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device_for_lm_eval(device_arg: str) -> str:
    device = resolve_device(device_arg)
    if device.type == "cuda":
        return "cuda:0"
    return device.type


def device_string_to_torch(device: str) -> torch.device:
    return torch.device("cuda" if device.startswith("cuda") else device)


def select_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def maybe_replace_mlp_with_sparse(
    model: torch.nn.Module,
    sparse_k: int,
    sparse_layers: list[int] | None,
) -> None:
    layer_idx = -1
    replacements: list[tuple[str, LlamaMLP]] = []
    for name, module in model.named_modules():
        if isinstance(module, LlamaMLP):
            layer_idx += 1
            if sparse_layers is None or layer_idx in sparse_layers:
                replacements.append((name, module))

    if not replacements:
        raise RuntimeError("Sparse mode selected but no LlamaMLP modules matched replacement rule.")

    for name, module in replacements:
        sparse_block = SparseMLP(module, k=sparse_k)
        replace_nested_module(model, name, sparse_block)


def load_base_model(
    model_name_or_path: str,
    trust_remote_code: bool,
    dtype: torch.dtype,
) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )


def build_model(
    mode: str,
    model_name_or_path: str,
    trust_remote_code: bool,
    dtype: torch.dtype,
    sparse_k: int = 4096,
    sparse_layers: list[int] | None = None,
) -> AutoModelForCausalLM:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of: {sorted(SUPPORTED_MODES)}")

    model = load_base_model(
        model_name_or_path=model_name_or_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    )
    if mode == "sparse":
        maybe_replace_mlp_with_sparse(model, sparse_k=sparse_k, sparse_layers=sparse_layers)
    return model
