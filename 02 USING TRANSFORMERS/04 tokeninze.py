from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
# ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
# [7993, 170, 13809, 23763, 2443, 1110, 3014]

decoded_string = tokenizer.decode(ids)
print(decoded_string)
