import torch
from path import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaMLP
from huggingface_hub import login
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
from fine_tune_model_simple import train_model
from tap import Tap

from ablations import set_up_sparsification
from rich.console import Console

from argument_parser import ARGUMENT_PARSER, parse_and_join_config
from custom_formatter import parsed_arguments_table

def dummy_launch(model, tokenizer):

    input_text = "What is Python?"

    tok_text = tokenizer(input_text, return_tensors="pt")
    decoded = tokenizer.decode(tok_text["input_ids"][0], skip_special_tokens=True)
    outputs = model.generate(**tok_text, max_new_tokens=100)
    decoded = tokenizer.decode(outputs[0])

    print(decoded)


def main(parsed_arguments):

    console = Console(width=150)
    console.print(parsed_arguments_table(parsed_arguments))
    

    if parsed_arguments.config_path is not None:
        parsed_arguments = parse_and_join_config(parsed_arguments)

    if parsed_arguments.HF_TOKEN:
        login(token=parsed_arguments.HF_TOKEN)
        print(parsed_arguments.HF_TOKEN)
        model_path = "meta-llama/Llama-3.2-3B"
    else:
        model_path = parsed_arguments.model_path
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    if parsed_arguments.sparsify:

        set_up_sparsification(
                    model,
                    parsed_arguments.ablation_kind,
                    parsed_arguments.importance_percentage,
                    parsed_arguments.sparse_implementation,
                    parsed_arguments.device,
                    parsed_arguments.k_param,
                    parsed_arguments.n_param,
                    parsed_arguments.m_param
                )


    if parsed_arguments.lora_finetune:
        model = torch.compile(model, mode="reduce-overhead")
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

    modules_importances = []
    
    # Print accuracy
    print(f"Accuracy: {results['results']}")
    # for name, module in model.model.named_modules():
    #     if isinstance(module, LlamaMLP):
    #         modules_importances.append(torch.mean(torch.tensor(module.importances)).item())

    # list1, list2 = zip(*sorted(zip(modules_importances, [i for i in range(28)])))
    # print(list(reversed(list1)))
    # print(list(reversed(list2)))


if __name__ == "__main__":
    args = ARGUMENT_PARSER(underscores_to_dashes=True).parse_args()
    main(args)