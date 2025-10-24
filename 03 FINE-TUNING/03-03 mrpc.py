from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)
print(raw_datasets["train"]["sentence1"])
print(raw_datasets["train"]["sentence2"])


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# Preprocessing the datasets
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)
# tokenized_dataset = tokenizer(
#     raw_datasets["train"]["sentence1"],
#     raw_datasets["train"]["sentence2"],
#     padding=True,
#     truncation=True,
# )
