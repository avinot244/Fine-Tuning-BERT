from transformers import AutoTokenizer
from datasets import load_dataset

def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer(examples["text"], padding="max_length", truncation=True)


if __name__ == "__main__":
    tokenizer : AutoTokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    encoded_input = tokenizer("This sentence will be encoded")

    dataset = load_dataset("yelp_review_full")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
