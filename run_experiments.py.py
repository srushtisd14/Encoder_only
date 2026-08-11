import argparse
import os
import time
import json

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


MODEL_CONFIG = {
    "bert-base": "bert-base-uncased",
    "roberta-base": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "albert-base-v2": "albert-base-v2",
    "electra-base": "google/electra-base-discriminator",
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "mpnet-base": "microsoft/mpnet-base",
}


DATASET_CONFIG = {
    "sst2": {
        "name": "glue",
        "subset": "sst2",
        "text1": "sentence",
        "text2": None,
    },

    "mrpc": {
        "name": "glue",
        "subset": "mrpc",
        "text1": "sentence1",
        "text2": "sentence2",
    },

    "qnli": {
        "name": "glue",
        "subset": "qnli",
        "text1": "question",
        "text2": "sentence",
    },

    "mnli": {
        "name": "glue",
        "subset": "mnli",
        "text1": "premise",
        "text2": "hypothesis",
    },

    "imdb": {
        "name": "stanfordnlp/imdb",
        "subset": None,
        "text1": "text",
        "text2": None,
    },

    "ag_news": {
        "name": "fancyzhx/ag_news",
        "subset": None,
        "text1": "text",
        "text2": None,
    },
}


def load_data(dataset_name):

    config = DATASET_CONFIG[dataset_name]

    if config["subset"] is not None:
        dataset = load_dataset(
            config["name"],
            config["subset"]
        )
    else:
        dataset = load_dataset(config["name"])

    return dataset, config


def get_num_labels(dataset):

    labels = dataset["train"].features["label"]

    return labels.num_classes


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CONFIG.keys()
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_CONFIG.keys()
    )

    args = parser.parse_args()

    model_name = MODEL_CONFIG[args.model]

    print("=" * 60)
    print("MODEL   :", args.model)
    print("DATASET :", args.dataset)
    print("HF MODEL:", model_name)
    print("=" * 60)

    # --------------------------------------------------
    # DATASET
    # --------------------------------------------------

    dataset, dataset_config = load_data(args.dataset)

    num_labels = get_num_labels(dataset)

    print("Number of labels:", num_labels)

    # --------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text1 = dataset_config["text1"]
    text2 = dataset_config["text2"]

    def tokenize_function(examples):

        if text2 is None:

            return tokenizer(
                examples[text1],
                truncation=True,
                max_length=256
            )

        return tokenizer(
            examples[text1],
            examples[text2],
            truncation=True,
            max_length=256
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # --------------------------------------------------
    # DATA COLLATOR
    # --------------------------------------------------

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    def compute_metrics(eval_pred):

        predictions, labels = eval_pred

        predictions = np.argmax(
            predictions,
            axis=-1
        )

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        )

        accuracy = accuracy_score(
            labels,
            predictions
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    output_dir = (
        f"outputs/{args.model}_{args.dataset}"
    )

    training_args = TrainingArguments(

        output_dir=output_dir,

        learning_rate=2e-5,

        per_device_train_batch_size=16,

        per_device_eval_batch_size=32,

        num_train_epochs=3,

        weight_decay=0.01,

        eval_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="f1",

        greater_is_better=True,

        logging_strategy="steps",

        logging_steps=100,

        report_to="none",

        fp16=torch.cuda.is_available(),

        seed=42
    )

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=tokenized["train"],

        eval_dataset=(
            tokenized["validation"]
            if "validation" in tokenized
            else tokenized["test"]
        ),

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics
    )

    # --------------------------------------------------
    # PARAMETER COUNT
    # --------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    train_start = time.perf_counter()

    train_result = trainer.train()

    train_time = (
        time.perf_counter()
        - train_start
    )

    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------

    eval_start = time.perf_counter()

    predictions = trainer.predict(
        tokenized["test"]
        if "test" in tokenized
        else tokenized["validation"]
    )

    inference_time = (
        time.perf_counter()
        - eval_start
    )

    preds = np.argmax(
        predictions.predictions,
        axis=-1
    )

    labels = predictions.label_ids

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            preds,
            average="weighted",
            zero_division=0
        )
    )

    accuracy = accuracy_score(
        labels,
        preds
    )

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    cm = confusion_matrix(
        labels,
        preds
    )

    # --------------------------------------------------
    # GPU MEMORY
    # --------------------------------------------------

    if torch.cuda.is_available():

        gpu_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

    else:

        gpu_memory = 0

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    result = {

        "model": args.model,

        "dataset": args.dataset,

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1": float(f1),

        "train_loss": float(
            train_result.training_loss
        ),

        "inference_time_seconds": float(
            inference_time
        ),

        "training_time_seconds": float(
            train_time
        ),

        "total_parameters": int(
            total_parameters
        ),

        "trainable_parameters": int(
            trainable_parameters
        ),

        "gpu_memory_mb": float(
            gpu_memory
        ),

        "confusion_matrix": cm.tolist()
    }

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    result_file = (
        f"results/"
        f"{args.model}_{args.dataset}.json"
    )

    with open(
        result_file,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )

    print("\nRESULT")
    print(json.dumps(
        result,
        indent=4
    ))


if __name__ == "__main__":
    main()