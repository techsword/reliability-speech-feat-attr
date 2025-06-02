import argparse
import os

import datasets
import evaluate
import numpy as np
import torch
from datasets import Audio, load_dataset, load_from_disk
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from utils import set_seed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets")


def compute_metrics(eval_pred):
    # All metrics are already predefined in the HF `evaluate` package
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = (
        eval_pred  # eval_pred is the tuple of predictions and labels returned by the model
    )
    predictions = np.argmax(logits, axis=-1)

    precision = precision_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )["precision"]
    recall = recall_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )["recall"]
    f1 = f1_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]

    # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores.
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def set_feature_extractor_and_model(
    dataset,
    model_type: str = "wav2vec2",
    freeze_embeddings: bool = True,
    freeze_projection_layer: bool = True,
):
    try:
        dataset_class_labels = dataset["train"].features["label"]
        num_labels = dataset_class_labels.num_classes
        label2id, id2label = {}, {}
        for classname in dataset_class_labels.names:
            label2id[classname] = dataset_class_labels.str2int(classname)
            id2label[dataset_class_labels.str2int(classname)] = classname
    except AttributeError:
        dataset_class_labels = dataset["train"].unique("label")
        num_labels = len(dataset_class_labels)
        label2id = {label: i for i, label in enumerate(dataset_class_labels)}
        id2label = {i: label for i, label in enumerate(dataset_class_labels)}

    if model_type == "wav2vec2":
        from datasets import Audio
        from transformers import (
            Wav2Vec2FeatureExtractor,
            Wav2Vec2ForSequenceClassification,
        )

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base"
        )
        hardcoded_sampling_rate = feature_extractor.sampling_rate
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/wav2vec2-base",
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        )

        dataset = dataset.cast_column(
            "audio", Audio(sampling_rate=hardcoded_sampling_rate)
        )

        def preprocess_function(examples):
            audio_arrays = [x["array"] for x in examples["audio"]]
            return feature_extractor(
                audio_arrays,
                sampling_rate=hardcoded_sampling_rate,
                max_length=hardcoded_sampling_rate * 15,
                truncation=True,
            )

        if freeze_embeddings:
            model.freeze_feature_encoder()
        if freeze_projection_layer:
            for param in model.wav2vec2.feature_projection.parameters():
                param.requires_grad = False

    elif model_type == "distilbert":
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizer,
        )

        feature_extractor = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        )

        if freeze_embeddings:
            for param in model.distilbert.embeddings.parameters():
                param.requires_grad = False

        def preprocess_function(examples):
            return feature_extractor(
                examples["text"], padding="max_length", truncation=True
            )
    else:
        raise ValueError(
            f"Model type {model_type} is not supported. Choose from 'wav2vec2' or 'distilbert'."
        )


    tokenized_datasets = dataset.map(
        preprocess_function, remove_columns=["audio"], batched=True
    )
    return (
        model,
        feature_extractor,
        tokenized_datasets,
    )


def main():
    taskname = args.taskname
    seed = args.seed
    model_type = args.model_type
    freeze_embeddings = args.freeze_embeddings
    freeze_projection_layer = args.freeze_projection_layer
    output_dir = os.path.join(args.output_dir, args.model_type)
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    no_cuda = args.no_cuda
    overwrite_output_dir = args.overwrite_output_dir
    freeze = "frozen" if freeze_embeddings else "nonfrozen"
    freeze += "_projection" if freeze_projection_layer else ""

    set_seed(seed)

    output_dir = os.path.join(
        output_dir,
        taskname,
        f"{taskname}_{freeze}_seed_{seed}",
    )

    print(
        f"Using device: {'cuda' if torch.cuda.is_available() and not no_cuda else 'cpu'}"
    )

    dataset_name_or_path = (
        os.path.join(DATASET_ROOT, taskname)
        if os.path.exists(os.path.join(DATASET_ROOT, taskname))
        else taskname
    )

    if "Voxceleb" in dataset_name_or_path:
        dataset = load_from_disk(dataset_name_or_path)
        dataset = dataset.rename_column("speaker_id", "label")

    elif "cv_genderid" in dataset_name_or_path:
        dataset = load_from_disk(
            os.path.join(
                DATASET_ROOT, dataset_name_or_path.replace("cv_genderid", "cv_spkid")
            )
        )
        dataset = dataset.remove_columns("label")
        dataset = dataset.rename_column("gender", "label")
    else:
        dataset = (
            load_from_disk(dataset_name_or_path)
            if os.path.exists(dataset_name_or_path)
            else load_dataset(dataset_name_or_path)
        )

    # Resample dataset to 16000 Hz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Split the train set into train and validation without shuffling
    if not isinstance(dataset, datasets.dataset_dict.DatasetDict):
        dataset = dataset.train_test_split(test_size=0.2, seed=42)

    model, feature_extractor, tokenized_datasets = set_feature_extractor_and_model(
        dataset,
        model_type=model_type,
        freeze_embeddings=freeze_embeddings,
        freeze_projection_layer=freeze_projection_layer,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        # gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        warmup_ratio=0.1,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=seed,
        lr_scheduler_type="linear",
        data_seed=42,
        overwrite_output_dir=overwrite_output_dir,
        save_total_limit=1,
        weight_decay=0.001,
        use_cpu=no_cuda,
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune wav2vec2 on audio classification"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{PROJECT_ROOT}/models",
        help="Base output directory for the model",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="wav2vec2",
        help="Model to use for fine-tuning, choose from wav2vec2 or distilbert",
    )
    parser.add_argument(
        "--taskname",
        type=str,
        default="iemocap",
        help="",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Disable CUDA even if available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the output directory",
    )
    parser.add_argument(
        "--freeze_embeddings", action="store_false", help="Freeze the embedding layers"
    )
    parser.add_argument(
        "--freeze_projection_layer",
        action="store_false",
        help="Freeze the projection layers after CNN embeddings",
    )
    args = parser.parse_args()

    main()
