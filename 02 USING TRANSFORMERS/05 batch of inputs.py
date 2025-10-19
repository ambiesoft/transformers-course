import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

# input_ids = torch.tensor(ids)
input_ids = torch.tensor([ids])
print(input_ids)

# This line will fail.
output = model(input_ids)
print("Logits", output.logits)

# too many indices for tensor of dimension 1
# IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
