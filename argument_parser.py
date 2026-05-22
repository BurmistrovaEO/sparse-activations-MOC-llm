from tap import Tap
from pathlib import Path
from typing import List
import json


class ARGUMENT_PARSER(Tap):
    config_path: Path = None
    #  ["mps:0", "cuda:0", "cpu"] TODO: reimplement as enum
    device: str = "mps:0"
    HF_TOKEN: str = None
    model_path: str = (
        "/Users/kateburmr/.cache/huggingface/hub/"
        "models--meta-llama--Llama-3.2-3B/snapshots/"
        "13afe5124825b4f3751f836b40dafda64c1ed062"
    )
    lora_finetune: bool = False
    sparsify: bool = False
    #  ["begin", "end", "begend", "middle", "importance"]
    # TODO: reimplement as enum
    ablation_kind: str = "begin"
    importance_percentage: int = None
    #  TODO: reimplement as enum
    sparse_implementation: str = "vanilla_k"
    k_param: int = 4096
    n_param: int = 2
    m_param: int = 8

    sparse_arg_names = [
        "ablation_kind",
        "importance_percentage",
        "sparse_implementation",
        "k_param",
        "n_param",
        "m_param",
    ]

    dataset: str = "tatsu-lab/alpaca"
    lora_rank: int = 2  # 8
    lora_alpha: int = 4  # 16
    lora_dropout: float = 0.05
    target_modules: List[str] = [
        "gate_proj",
        "up_proj",
        "down_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_proj",
    ]

    train_output_dir: str = "results/res0/mt0-large-lora"
    learning_rate: float = 1e-3
    per_device_train_batch_size: int = 2  # 32
    per_device_eval_batch_size: int = 2  # 32
    num_train_epochs: int = 2
    weight_decay: float = 0.01

    ft_arg_names: List[str] = [
        "dataset",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "train_output_dir",
        "learning_rate",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "num_train_epochs",
        "weight_decay",
    ]

    hf_tasks: List[str] = [
        "hellaswag",
        "arc_challenge",
        "arc_easy",
        "boolq",
        "winogrande",
    ]


def parse_and_join_config(parsed_arguments):
    with open(parsed_arguments.config_path, "r", encoding="utf-8") as file:
        parsed_config = json.load(file)
    for key, val in parsed_config.items():
        setattr(parsed_arguments, key, val)
    return parsed_arguments
