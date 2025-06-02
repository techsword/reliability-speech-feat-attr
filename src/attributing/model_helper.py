import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import textgrids
import torch
from torch import nn
from tqdm import tqdm
from transformers import (
    DistilBertTokenizer,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Processor,
)
from transformers.utils import ModelOutput

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WrapperModel(nn.Module):
    def __init__(self, model, input_type="embedding", output_type="logit"):
        """_summary_

        Args:
            model (_type_): _description_
            input_type (str, optional): choose between "input", "spec", or "embedding". Defaults to "embedding".
            output_type (str, optional): _description_. Defaults to "logit".
        """
        super().__init__()
        self.model = model
        self.input_type = input_type
        self.output_type = output_type
        if "wav2vec" in model.config.model_type:
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        elif "distilbert" in model.config.model_type:
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )

    def _forward_emb(self, embeddings):
        if "wav2vec" in self.model.config.model_type:
            hidden_states = self.model.wav2vec2.encoder(embeddings)

            hidden_states = self.model.projector(hidden_states[0])
            pooled_output = hidden_states.mean(dim=1)
            logits = self.model.classifier(pooled_output)

        elif "distilbert" in self.model.config.model_type:
            logits = self.model(inputs_embeds=embeddings).logits

        return logits

    def _forward_input(self, input_values):
        if "wav2vec" in self.model.config.model_type:
            outputs = self.model(input_values, return_dict=True)
        elif "distilbert" in self.model.config.model_type:
            outputs = self.model(input_values, return_dict=True)
        logits = outputs["logits"]

        return logits

    def _forward_spec(self, input_values):
        # Use the istft function to convert the input waveform to spectrogram
        # First check if need to shift dim
        if input_values.shape[1] != 201:
            # Swap the time and frequency axis
            input_values = input_values.transpose(1, 2)
        input_values = torch.istft(
            input_values,
            n_fft=400,
            window=torch.hann_window(400, device=input_values.device),
            hop_length=320,
            win_length=400,
            normalized=False,
            return_complex=False,
        ).to(DEVICE, dtype=torch.float32)
        if len(input_values.shape) == 1:
            input_values = input_values.unsqueeze(0)
        logits = self.model(input_values).logits
        return logits

    def forward(self, input):
        if self.input_type == "embedding":
            logits = self._forward_emb(input)
        elif self.input_type == "input":
            logits = self._forward_input(input)
        elif self.input_type == "spec":
            logits = self._forward_spec(input)
        else:
            raise ValueError("Invalid mode")

        if self.output_type == "logit":
            return logits
        elif self.output_type == "prob":
            return torch.nn.functional.softmax(logits, dim=-1)
        else:
            raise ValueError("Invalid output type")

    def process_input(self, example, baseline_mode=False):
        if "wav2vec" in self.model.config.model_type:
            if "input_values" in example:
                input_values = example["input_values"]
            else:
                input_values = self.processor(
                    example["audio"]["array"],
                    sampling_rate=example["audio"]["sampling_rate"],
                ).input_values[0]
        elif "distilbert" in self.model.config.model_type:
            if "input_ids" in example:
                input_values = example["input_ids"]
            else:
                input_values = self.tokenizer(
                    example["transcription"], padding="max_length", truncation=True
                )["input_ids"]

        processed_input = torch.tensor(input_values, device=DEVICE).unsqueeze(0)
        if baseline_mode:
            # Create a silent waveform the same shape as the input
            processed_input = processed_input * 0

        if self.input_type == "embedding":
            # processed_input should have shape (batch_size, seq_len, feature_dim)
            processed_input, _ = self.extract_embedding(processed_input)
        elif self.input_type == "spec":
            # processed_input should have shape (batch_size, seq_len, feature_dim)
            processed_input = self.extract_spec(processed_input)
        elif self.input_type == "input":
            pass
        else:
            raise ValueError("Invalid mode")

        return processed_input

    def extract_embedding(self, input_values):
        # Extract the CNN embedding or token embedding from the model
        with torch.no_grad():
            # forward cnn part (before transformer)
            if "wav2vec" in self.model.config.model_type:
                extract_features = self.model.wav2vec2.feature_extractor(input_values)
                extract_features = extract_features.transpose(1, 2)
                embeddings, extract_features = self.model.wav2vec2.feature_projection(
                    extract_features
                )
            elif "distilbert" in self.model.config.model_type:
                embeddings = self.model.get_input_embeddings()(input_values)
                extract_features = None
            else:
                raise NotImplementedError(
                    f"Model type {self.model.config.model_type} not supported"
                )

        return embeddings, extract_features

    def extract_spec(self, input_values):
        # Use the stft function to convert the input waveform to spectrogram
        return torch.stft(
            input_values,
            n_fft=400,
            window=torch.hann_window(400, device=input_values.device),
            hop_length=320,
            win_length=400,
            normalized=False,
            return_complex=True,
        ).moveaxis(-1, 1)


class WrapperModelMultiHead(WrapperModel):
    # Inherit the WrapperModel class and modify for multi-head classification
    def __init__(self, model, input_type="embedding", output_type="logit"):
        super().__init__(model, input_type, output_type)

    def _forward_emb(self, embeddings):
        if "wav2vec" in self.model.config.model_type:
            hidden_states = self.model.wav2vec2.encoder(embeddings)

            # hidden_states = self.model.projector(hidden_states[0])
            hidden_states = self.model.dropout(hidden_states[0])
            pooled_output = hidden_states.mean(dim=1)
            logits1 = self.model.classifier1(pooled_output)
            logits2 = self.model.classifier2(pooled_output)
            logits3 = self.model.classifier3(pooled_output)

        elif "distilbert" in self.model.config.model_type:
            # logits = self.model(inputs_embeds=embeddings).logits
            raise NotImplementedError("DistilBert model not supported")

        return (logits1, logits2, logits3)

    def _forward_input(self, input_values):
        if "wav2vec" in self.model.config.model_type:
            outputs = self.model(input_values, return_dict=True)
        elif "distilbert" in self.model.config.model_type:
            outputs = self.model(input_values, return_dict=True)
        logits = (outputs["logits1"], outputs["logits2"], outputs["logits3"])

        return logits

    def _forward_spec(self, input_values):
        if input_values.shape[1] != 201:
            # Swap the time and frequency axis
            input_values = input_values.transpose(1, 2)
        input_values = torch.istft(
            input_values,
            n_fft=400,
            window=torch.hann_window(400, device=input_values.device),
            hop_length=320,
            win_length=400,
            normalized=False,
            return_complex=False,
        ).to(DEVICE, dtype=torch.float32)
        if len(input_values.shape) == 1:
            input_values = input_values.unsqueeze(0)
        output = self.model(input_values, return_dict=True)
        logits = (output["logits1"], output["logits2"], output["logits3"])
        return logits

    def forward(self, input, classifier_idx=0):
        if self.input_type == "embedding":
            logits = self._forward_emb(input)
        elif self.input_type == "input":
            logits = self._forward_input(input)
        elif self.input_type == "spec":
            logits = self._forward_spec(input)
        else:
            raise ValueError("Invalid mode")

        if self.output_type == "logit":
            return logits[classifier_idx]
        elif self.output_type == "prob":
            return torch.nn.functional.softmax(logits[classifier_idx], dim=-1)
        else:
            raise ValueError("Invalid output type")

    def full_forward(self, input):
        if self.input_type == "embedding":
            logits = self._forward_emb(input)
        elif self.input_type == "input":
            logits = self._forward_input(input)
        elif self.input_type == "spec":
            logits = self._forward_spec(input)
        else:
            raise ValueError("Invalid mode")

        if self.output_type == "logit":
            return logits
        elif self.output_type == "prob":
            return torch.nn.functional.softmax(logits, dim=-1)
        else:
            raise ValueError("Invalid output type")


def align_score_with_forced_alignment(example, attributions, sr=16000):
    """Use the forced alignment to align the attributions with the words in the textgrid

    Args:
        example (_type_): an entry from the dataset
        attributions (_type_): attribution scores
        sr (int, optional): sampling rate. Defaults to 16000.

    Returns:
        _type_: aggregated scores on the word level
    """
    waveform = example["input_values"]

    tg = textgrids.TextGrid(example["textgrid_path"])
    alignment = []
    for interval in tg["words"]:
        alignment.append((interval.text, interval.xmin, interval.xmax))
    sec_in_frame = len(waveform) / sr / len(attributions)
    # Compute the alignment scores
    aggre_score = []
    for word, start, end in alignment:
        word = "[SIL]" if word == "" else word
        frame_start = int(start / sec_in_frame)
        frame_end = int(end / sec_in_frame)
        score = attributions[frame_start:frame_end].mean()
        aggre_score.append((word, score))
    return aggre_score


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


def choose_attr_method(method="saliency", forward_func=None):
    from captum.attr import (
        FeatureAblation,
        IntegratedGradients,
        Lime,
        Occlusion,
        Saliency,
    )

    method = method.lower()
    if method == "saliency":
        attr_func = Saliency(forward_func)
    elif method == "ig":
        attr_func = IntegratedGradients(forward_func)
    elif method == "lime":
        # attr_func = Lime(forward_func)
        from captum._utils.models.linear_model import (
            SkLearnLasso,
            SkLearnLinearRegression,
        )

        attr_func = Lime(forward_func, interpretable_model=SkLearnLinearRegression())
    elif method == "occlusion":
        attr_func = Occlusion(forward_func)
    elif method == "featureablation":
        attr_func = FeatureAblation(forward_func)
    else:
        print(
            f"Invalid method {method}. Please choose from saliency, ig, lime, occlusion"
        )
        return None
    return attr_func
