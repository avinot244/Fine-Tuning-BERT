from transformers import AutoTokenizer


def encode_batch(batch : list[str], padding : bool, truncation : bool):
    tokenizer : AutoTokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    encoded_batch = tokenizer(batch, padding=padding, truncation=truncation, return_tensors="pt")
    return encoded_batch