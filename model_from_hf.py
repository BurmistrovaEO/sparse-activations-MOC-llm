import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

HF_TOKEN = os.environ.get('HF_TOKEN')

if __name__ == "__main__":
    login(token=HF_TOKEN)
    print(HF_TOKEN)
    model_path = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    for name, module in model.named_modules():
        print(name)

    input_text = "How to learn japanese in three easy steps before the week is over? (It's friday)"
    #input_text = "What is Python?"

    tok_text = tokenizer(input_text, return_tensors="pt")

    decoded = tokenizer.decode(tok_text["input_ids"][0], skip_special_tokens=True)
    #print(tok_text)

    #output = model(**tok_text)
    outputs = model.generate(**tok_text, max_new_tokens=100)

    decoded = tokenizer.decode(outputs[0])
    print(decoded)
    #print(model)