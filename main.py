from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
from transformers import Trainer

def tokenize_function(examples):
    """
    This function takes a set of sentances as a parameter and returns the tokenized sentances

    Param
    -------
        examples : list of string
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred, metric):
    """
    Compute the accuracy of a given prediction with a metrics

    Param
    --------
        eval_pred : tuple that gives the predictions and the labels of the the prediction
        metric : metric used to evaluate the accurary of our prediction
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    # Getting our tokenizer
    tokenizer : AutoTokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Loading our dataset from hugging face
    print("Getting dataset")
    dataset = load_dataset("yelp_review_full")
    print("Tokenizing dataset")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))

    print(small_train_dataset)
    print(small_eval_dataset)

    # Getting our pretrained model
    print("Getting pretrained BERT model")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

    # Getting our training hyperparameters
    print("Getting training hyperparameters from the model")
    training_args = TrainingArguments(output_dir="./test_trainer")

    # Getting our evaluate class to evaluate our model during training
    print("Getting our evaluate class for training")
    metric = evaluate.load("accuracy")

    # Creating our trainger 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics
    )

    print("Training our model")
    trainer.train()


