import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoModel
from transformers import AutoTokenizer

# As we saw in Chapter 1, this pipeline groups together three steps:
# preprocessing, passing the input through the model, and postprocessing:

# Preprocessing with a tokeninzer
# Like other neural networks, Transformer models can't process raw text directly,
# so the first step of our pipleline is to ververt the text inputs into nubers that
# the model can make sense of. To do this we use a tokenizer, which will be
# responsible for:
# - Splitting the input into words, subwords, or symbols (like punctuation)
# that are called tokens
# - Mapping each token to an integer
# Adding additional inputs that be useful to the model
#
# All this preprocessing needs to be done in exactly the same way as shen the


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True,
                   truncation=True, return_tensors="pt")
print(inputs)
# {
#     'input_ids': tensor([
#         [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
#         [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
#     ]),
#     'attention_mask': tensor([
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
#     ])
# }

# Going through the model
if False:
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    # embeddings model and layers
    model = AutoModel.from_pretrained(checkpoint)

    # Same as following:
    # model(input_ids=tensor([[101, 2023, 2003, 1037, 2742, 102]]),
    #   attention_mask=tensor([[1, 1, 1, 1, 1, 1]]))
    outputs = model(**inputs)

    # last_hidden_state is feature representations for each token in the input
    print(outputs.last_hidden_state.shape)
    # torch.Size([2, 16, 768])

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# model with a sequence classification head
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)
# torch.Size([2, 2])
print(outputs.logits)
# tensor([[-1.5607,  1.6123],
#        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward0>)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
# tensor([[4.0195e-02, 9.5980e-01],
#        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward0>)
