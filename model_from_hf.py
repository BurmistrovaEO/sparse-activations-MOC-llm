import os
import torch
import transformers
from path import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from transformers.models.llama.modeling_llama import LlamaMLP
from huggingface_hub import login
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
from fine_tune_model_simple import train_model
from tap import Tap

from sparsificaiton import SparseMLP, NMsparseMLP, replace_nested_module
from typing import List


def dummy_launch(model, tokenizer):

    input_text = "What is Python?"

    tok_text = tokenizer(input_text, return_tensors="pt")
    decoded = tokenizer.decode(tok_text["input_ids"][0], skip_special_tokens=True)
    outputs = model.generate(**tok_text, max_new_tokens=100)
    decoded = tokenizer.decode(outputs[0])

    print(decoded)


class ARGUMENT_PARSER(Tap):
    config_path: Path = None
    HF_TOKEN: str = None
    model_path: str = "/Users/kateburmr/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062"
    lora_finetune: bool = False
    dataset: str = "tatsu-lab/alpaca"



    lora_rank: int = 2 # 8
    lora_alpha: int = 4 # 16
    lora_dropout: float = 0.05
    target_modules: List[str] = ['gate_proj', 'up_proj', 'down_proj', 'k_proj', 'v_proj', 'o_proj', 'q_proj']

    train_output_dir: str = "results/res0/mt0-large-lora"
    learning_rate: float = 1e-3
    per_device_train_batch_size: int = 2 # 32
    per_device_eval_batch_size: int = 2 # 32
    num_train_epochs: int = 2
    weight_decay: float = 0.01

    hf_tasks: List[str] = ["hellaswag", "arc_challenge", "arc_easy", "boolq", "winogrande"] # wikitext separately
    #hf_tasks: List[str] = ["wikitext"] # wikitext separately


def main(parsed_arguments):

    if parsed_arguments.HF_TOKEN:
        login(token=parsed_arguments.HF_TOKEN)
        print(parsed_arguments.HF_TOKEN)
        model_path = "meta-llama/Llama-3.2-3B"
    else:
        model_path = parsed_arguments.model_path
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")


    to_replace_names_modules = {}

    with torch.no_grad():
        counter = 0
        for name, module in model.named_modules():
            if isinstance(module, LlamaMLP):
                if 2 < counter < 25:
                    counter+=1
                    continue

                counter+=1
                print(name)
                to_replace_names_modules[name] = module
                

        for name, module in to_replace_names_modules.items():
            #sparseBlock = SparseMLP(module, k=1024)
            sparseBlock = NMsparseMLP(module, 8, 16)
            replace_nested_module(model, name, sparseBlock)

    if parsed_arguments.lora_finetune:
        train_model(
                model = model,
                tokenizer = tokenizer,
                lora_rank = parsed_arguments.lora_rank,
                lora_alpha = parsed_arguments.lora_alpha,
                lora_dropout = parsed_arguments.lora_dropout,
                target_modules = parsed_arguments.target_modules,
                dataset_path = parsed_arguments.dataset,
                train_output_dir = parsed_arguments.train_output_dir,
                learning_rate = parsed_arguments.learning_rate,
                per_device_train_batch_size = parsed_arguments.per_device_train_batch_size,
                per_device_eval_batch_size = parsed_arguments.per_device_eval_batch_size,
                num_train_epochs = parsed_arguments.num_train_epochs,
                weight_decay = parsed_arguments.weight_decay
            )


    # Evaluate model

    model = HFLM(
        pretrained=model,
        device="mps:0"
    )

    # Run evaluation on a single task
    results = evaluator.simple_evaluate(
        model=model,
        tasks=parsed_arguments.hf_tasks,
        num_fewshot=0,
        limit=100,
        batch_size=8
    )

    # Print accuracy
    print(f"Accuracy: {results['results']}")


if __name__ == "__main__":
    args = ARGUMENT_PARSER(underscores_to_dashes=True).parse_args()
    main(args)