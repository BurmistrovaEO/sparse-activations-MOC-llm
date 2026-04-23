import os
import torch
import transformers
from path import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaMLP
from huggingface_hub import login
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
from tap import Tap

from sparsificaiton import SparseMLP, replace_nested_module

#HF_TOKEN = os.environ.get('HF_TOKEN')

class ARGUMENT_PARSER(Tap):
    config_path: Path = None
    HF_TOKEN: str = None
    model_path: str = "/Users/kateburmr/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062"

#TODO rewrite as main
def main(parsed_arguments):
    #TODO check if HF_TOken is not none
    if parsed_arguments.HF_TOKEN:
        login(token=parsed_arguments.HF_TOKEN)
        print(parsed_arguments.HF_TOKEN)
        model_path = "meta-llama/Llama-3.2-3B"
    else:
        model_path = parsed_arguments.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    to_replace_names_modules = {}

    with torch.no_grad():
        counter = 0
        for name, module in model.named_modules():
            if isinstance(module, LlamaMLP):
                if counter < 10:
                    counter+=1
                    continue
                print(name)
                to_replace_names_modules[name] = module
                break

        for name, module in to_replace_names_modules.items():
            sparseBlock = SparseMLP(module)
            replace_nested_module(model, name, sparseBlock)
            

    # print(model)

    #input_text = "How to learn japanese in three easy steps before the week is over? (It's friday)"
    input_text = "What is Python?"

    tok_text = tokenizer(input_text, return_tensors="pt")

    decoded = tokenizer.decode(tok_text["input_ids"][0], skip_special_tokens=True)

    #output = model(**tok_text)
    outputs = model.generate(**tok_text, max_new_tokens=100)

    decoded = tokenizer.decode(outputs[0])

    print(decoded)

    # Evaluate model

    model = HFLM(
        pretrained=model,
        device="mps:0"
    )

    # Run evaluation on a single task
    results = evaluator.simple_evaluate(
        model=model,
        tasks=["arc_easy"],
        num_fewshot=0,
        limit=100,
        batch_size=8
    )

    # Print accuracy
    print(f"Accuracy: {results['results']['arc_easy']}")


if __name__ == "__main__":
    args = ARGUMENT_PARSER(underscores_to_dashes=True).parse_args()
    main(args)