from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)
# DatasetDict({
#     train: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 3668
#     })
#     validation: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 408
#     })
#     test: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 1725
#     })
# })

raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])
# {
#     'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
#     'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
#     'label': 1,
#     'idx': 0
# }

print(raw_train_dataset.features)
# {
#     'sentence1': Value('string'), 
#     'sentence2': Value('string'), 
#     'label': ClassLabel(names=['not_equivalent', 'equivalent']), 
#     'idx': Value('int32')
# }
