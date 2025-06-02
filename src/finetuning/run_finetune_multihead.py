import argparse
import os
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import ClassLabel, load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2Config,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Processor,
)
from transformers.utils import ModelOutput

from utils import set_seed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits1, logits2, logits3 = logits
    labels1, labels2, labels3 = labels

    # Convert logits to predictions
    preds1 = np.argmax(logits1, axis=-1)
    preds2 = np.argmax(logits2, axis=-1)
    preds3 = np.argmax(logits3, axis=-1)

    # Calculate metrics for each task
    metrics = {}

    for name, preds, labels in [
        ("action", preds1, labels1),
        ("object", preds2, labels2),
        ("location", preds3, labels3),
    ]:
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted"
        )

        metrics.update(
            {
                f"{name}_accuracy": accuracy,
                f"{name}_f1": f1,
                f"{name}_precision": precision,
                f"{name}_recall": recall,
            }
        )

    # Add average metrics
    metrics["accuracy"] = np.mean(
        [
            metrics["action_accuracy"],
            metrics["object_accuracy"],
            metrics["location_accuracy"],
        ]
    )

    return metrics


@dataclass
class Wav2Vec2MultiLabelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits1: torch.FloatTensor = None
    logits2: torch.FloatTensor = None
    logits3: torch.FloatTensor = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class Wav2Vec2ForMultiLabelClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.classifier1 = nn.Linear(config.hidden_size, config.num_labels_1)
        self.classifier2 = nn.Linear(config.hidden_size, config.num_labels_2)
        self.classifier3 = nn.Linear(config.hidden_size, config.num_labels_3)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_cnn_projection(self):
        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values,
        attention_mask=None,
        labels1=None,
        labels2=None,
        labels3=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        hidden_states = torch.mean(hidden_states, dim=1)

        logits1 = self.classifier1(hidden_states)
        logits2 = self.classifier2(hidden_states)
        logits3 = self.classifier3(hidden_states)

        loss = None
        if labels1 is not None and labels2 is not None and labels3 is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss1 = loss_fct(
                logits1.view(-1, self.config.num_labels_1), labels1.view(-1)
            )
            loss2 = loss_fct(
                logits2.view(-1, self.config.num_labels_2), labels2.view(-1)
            )
            loss3 = loss_fct(
                logits3.view(-1, self.config.num_labels_3), labels3.view(-1)
            )
            loss = loss1 + loss2 + loss3

        return Wav2Vec2MultiLabelOutput(
            loss=loss,
            logits1=logits1,
            logits2=logits2,
            logits3=logits3,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class DataCollatorCTCWithMultipleLabels:
    processor: Any
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(
        self, features: list[dict[str, Union[list[int], torch.Tensor]]]
    ) -> dict[str, torch.Tensor]:
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels1"] = torch.tensor([f["labels1"] for f in features])
        batch["labels2"] = torch.tensor([f["labels2"] for f in features])
        batch["labels3"] = torch.tensor([f["labels3"] for f in features])

        return batch


class MultiLabelWav2Vec2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels1 = inputs.pop("labels1")
        labels2 = inputs.pop("labels2")
        labels3 = inputs.pop("labels3")

        outputs = model(**inputs, labels1=labels1, labels2=labels2, labels3=labels3)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def main():
    # Load arguments
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

    # Load processor and setup config
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")

    # Get label counts from your dataset
    dataset = load_from_disk(f"{DATASET_ROOT}/fsc-ic")

    action_classes = ClassLabel(names=sorted(dataset["train"].unique("action")))
    object_classes = ClassLabel(names=sorted(dataset["train"].unique("object")))
    location_classes = ClassLabel(names=sorted(dataset["train"].unique("location")))

    dataset = dataset.cast_column("action", action_classes)
    dataset = dataset.cast_column("object", object_classes)
    dataset = dataset.cast_column("location", location_classes)

    # Set number of labels in config
    config.num_labels_1 = action_classes.num_classes
    config.num_labels_2 = object_classes.num_classes
    config.num_labels_3 = location_classes.num_classes

    # Create model
    model = Wav2Vec2ForMultiLabelClassification.from_pretrained(
        "facebook/wav2vec2-base",
        config=config,
    )

    # Freeze feature extractor and cnn projection layer
    model.freeze_feature_extractor()
    model.freeze_cnn_projection()

    # Setup training

    column_renaming_dict = {
        "action": "labels1",
        "object": "labels2",
        "location": "labels3",
    }

    dataset = dataset.rename_columns(column_renaming_dict)
    train_dataset_encoded = dataset["train"]
    eval_dataset_encoded = dataset["valid"]
    test_dataset_encoded = dataset["test"]

    training_args = TrainingArguments(
        # output_dir="./wav2vec2-multi-label",
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_steps=100,
        eval_steps=100,
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        warmup_ratio=0.1,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        weight_decay=0.001,
        push_to_hub=False,
        metric_for_best_model="accuracy",
        lr_scheduler_type="linear",
        save_total_limit=1,
        load_best_model_at_end=True,
        seed=seed,
        data_seed=42,
        overwrite_output_dir=overwrite_output_dir,
        use_cpu=no_cuda,
    )

    data_collator = DataCollatorCTCWithMultipleLabels(processor=processor, padding=True)

    trainer = MultiLabelWav2Vec2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_encoded,
        eval_dataset=eval_dataset_encoded,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate(eval_dataset_encoded)
    print(eval_results)

    # Save trained model
    trainer.save_model(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune wav2vec2 or distilbert on multilabel classification"
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
        default="fsc-ic",
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
