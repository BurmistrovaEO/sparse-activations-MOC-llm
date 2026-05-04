import torch
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_dataset
from data import prepare_data, format_instruction


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

def train_model(
        model,
        tokenizer,
        lora_rank,
        lora_alpha,
        lora_dropout,
        target_modules,
        dataset_path,
        train_output_dir,
        learning_rate,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        num_train_epochs,
        weight_decay
    ):
    
    config = LoraConfig(
        r = lora_rank,
        lora_alpha = lora_alpha,
        bias = "none",           
        lora_dropout = lora_dropout,
        task_type = "CAUSAL_LM",
        target_modules = target_modules,
        )
    
    dataset = load_dataset(dataset_path)
    small_dataset = dataset["train"].select(range(1000))
    val_dataset = dataset["train"].select(range(1000, 1500))
    del dataset

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    formatted_dataset = small_dataset.map(format_instruction)
    formatted_val_dataset = val_dataset.map(format_instruction)


    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=64,
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
        output_dir = train_output_dir,
        learning_rate = learning_rate,
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
        num_train_epochs = num_train_epochs,
        weight_decay = weight_decay,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset_val,
        data_collator=data_collator,
    )

    trainer.train()