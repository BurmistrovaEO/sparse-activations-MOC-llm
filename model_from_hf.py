import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaMLP
from huggingface_hub import login

from sparsificaiton import SparseMLP, replace_nested_module

HF_TOKEN = os.environ.get('HF_TOKEN')

if __name__ == "__main__":
    login(token=HF_TOKEN)
    print(HF_TOKEN)
    model_path = "meta-llama/Llama-3.2-3B"
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
                to_replace_names_modules[name] = module
                break

        for name, module in to_replace_names_modules.items():
            sparseBlock = SparseMLP(module)
            replace_nested_module(model, name, sparseBlock)
            

    print(model)

    #input_text = "How to learn japanese in three easy steps before the week is over? (It's friday)"
    input_text = "What is Python?"

    tok_text = tokenizer(input_text, return_tensors="pt")

    decoded = tokenizer.decode(tok_text["input_ids"][0], skip_special_tokens=True)

    #output = model(**tok_text)
    outputs = model.generate(**tok_text, max_new_tokens=100)

    decoded = tokenizer.decode(outputs[0])

    print(decoded)
