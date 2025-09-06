from datasets import load_dataset

dataset = load_dataset("melisekm/natural-disasters-from-social-media")
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

print(train_data[0])
