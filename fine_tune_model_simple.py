from datasets import load_dataset

def format_dataset(examples):
    if isinstance(examples["prompt"], list):
        output_texts = []
        for i in range(len(examples["prompt"])):
            converted_sample = [
                {"role": "user", "content": examples["prompt"][i]},
                {"role": "assistant", "content": examples["completion"][i]},
            ]
            output_texts.append(converted_sample)
        return {'messages': output_texts}
    else:
        converted_sample = [
            {"role": "user", "content": examples["prompt"]},
            {"role": "assistant", "content": examples["completion"]},
        ]
        return {'messages': converted_sample}

def prepare_data(dataset):

    dataset = dataset.rename_column("sentence", "prompt")
    dataset = dataset.rename_column("translation_extra", "completion")
    dataset = dataset.map(format_dataset)
    dataset = dataset.remove_columns(['prompt', 'completion', 'translation'])

    print(dataset[0])
    return dataset

