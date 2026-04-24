import os
import torch
import transformers
from path import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from transformers.models.llama.modeling_llama import LlamaMLP
from huggingface_hub import login
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
from tap import Tap

from sparsificaiton import SparseMLP, replace_nested_module
from fine_tune_model_simple import prepare_data
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from alpaca_eval import evaluate



#HF_TOKEN = os.environ.get('HF_TOKEN')

def compute_token_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Token-level prediction accuracy"""
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        predictions = shift_logits.argmax(dim=-1)
        mask = shift_labels != self.ignore_index
        
        correct = (predictions == shift_labels) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        
        return accuracy

class ARGUMENT_PARSER(Tap):
    config_path: Path = None
    HF_TOKEN: str = None
    model_path: str = "/Users/kateburmr/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062"
    lora_finetune: bool = True
    dataset: str = "tatsu-lab/alpaca"

#TODO rewrite as main
def main(parsed_arguments):
    #TODO check if HF_TOken is not none
    if parsed_arguments.HF_TOKEN:
        login(token=parsed_arguments.HF_TOKEN)
        print(parsed_arguments.HF_TOKEN)
        model_path = "meta-llama/Llama-3.2-3B"
    else:
        model_path = parsed_arguments.model_path
        

    config = LoraConfig(
        # the rank of the adapter, the lower the fewer parameters you'll need to train
        r=2, #8                   
        lora_alpha=4, #16, # multiplier, usually 2*r
        bias="none",           
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        # Newer models, such as Phi-3 at time of writing, may require 
        # manually setting target modules
        target_modules=['gate_proj', 'up_proj', 'down_proj', 'k_proj', 'v_proj', 'o_proj', 'q_proj'],
    )
    
    dataset = load_dataset(parsed_arguments.dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    small_dataset = dataset["train"].select(range(1000))
    val_dataset = dataset["train"].select(range(1000, 1500))


    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(torch.mps.current_allocated_memory())
    del dataset
    torch.mps.empty_cache()
    print(torch.mps.current_allocated_memory())

    # 3. Format data for Llama
    def format_instruction(example):
        """Format Alpaca data for Llama"""
        text = f"### Instruction:\n{example['instruction']}\n"
        if example['input']:
            text += f"### Input:\n{example['input']}\n"
        text += f"### Response:\n{example['output']}"
        return {"text": text}

    formatted_dataset = small_dataset.map(format_instruction)
    formatted_val_dataset = val_dataset.map(format_instruction)

    # 4. Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
    
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "instruction", "input", "output"]
    )

    tokenized_dataset_val = formatted_val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "instruction", "input", "output"]
    )
    
    tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": x["input_ids"]})
    tokenized_dataset_val = tokenized_dataset_val.map(lambda x: {"labels": x["input_ids"]})


    model = get_peft_model(model, config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        return_tensors="pt",
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir="results/res0/mt0-large-lora",
        learning_rate=1e-3,
        per_device_train_batch_size=1, #32
        per_device_eval_batch_size=1, #32
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    #trainer.evaluate = evaluate

    trainer.train()

    return

    dataset = load_dataset(parsed_arguments.dataset, split="train")

    dataset = dataset.train_test_split(test_size=0.1)

    train_dataset = prepare_data(dataset["train"])
    test_dataset = prepare_data(dataset["test"])

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    if parsed_arguments.lora_finetune:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
        llama3_template = """{% for message in messages %}
                            {% if message['role'] ## 'system' %}
                            {{'<|system|>\n' + message['content'] + '<|end|>\n'}}
                            {% elif message['role'] ## 'user' %}
                            {{'<|user|>\n' + message['content'] + '<|end|>\n'}}
                            {% elif message['role'] ## 'assistant' %}
                            {{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}
                            {% endif %}
                        {% endfor %}
                        {% if add_generation_prompt %}
                        {{ '<|assistant|>\n' }}{% else %}{{ eos_token }}
                        {% endif %}"""
        tokenizer.chat_template = llama3_template
        print(tokenizer.chat_template)

    model = get_peft_model(model, config)

    training_args = TrainingArguments(
        output_dir="results/res0/mt0-large-lora",
        learning_rate=1e-3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_token_accuracy,
    )

    train_dataloader = trainer._get_dataloader(train_dataset, description="Training", batch_size=32)

    #print(trainer)

    #trainer.train()


    return
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

            
    model = get_peft_model(model, config)

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