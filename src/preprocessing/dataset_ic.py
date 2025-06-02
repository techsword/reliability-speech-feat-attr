import argparse
import os

import pandas as pd
from datasets import Audio, Dataset, DatasetDict
from transformers import (
    DistilBertTokenizer,
    Wav2Vec2FeatureExtractor,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def encode_fsc_dataset(corpus_path="~/corpora/FSC/"):
    FSC_ROOT = os.path.expanduser(corpus_path)
    splits = ["train", "valid", "test"]
    fsc = DatasetDict()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base"
    )
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    for split in splits:
        csv_file = os.path.join(FSC_ROOT, f"data/{split}_data.csv")
        df = pd.read_csv(csv_file, index_col=0)
        df["path"] = df["path"].apply(lambda x: os.path.join(FSC_ROOT, x))
        df["textgrid_path"] = df["path"].apply(
            lambda x: x.replace(".wav", ".TextGrid").replace("wavs", "aligned")
        )
        # add suffix to object and location
        df["object"] = df["object"].apply(lambda x: x + "_object" if x == "none" else x)
        df["location"] = df["location"].apply(
            lambda x: x + "_location" if x == "none" else x
        )

        fsc[split] = Dataset.from_pandas(df, preserve_index=False).rename_column(
            "path", "audio"
        )

    ALL_LABELS = []
    ALL_LABELS.extend(fsc["train"].unique("action"))
    ALL_LABELS.extend(fsc["train"].unique("object"))
    ALL_LABELS.extend(fsc["train"].unique("location"))
    id2label = {k: l for k, l in enumerate(ALL_LABELS)}
    label2id = {l: k for k, l in enumerate(ALL_LABELS)}

    fsc = fsc.cast_column("audio", Audio(sampling_rate=16000))
    fsc = fsc.class_encode_column("action")
    fsc = fsc.class_encode_column("object")
    fsc = fsc.class_encode_column("location")

    def preprocess_function(examples):
        labels = []
        for i in range(len(examples["action"])):
            label = [0] * len(id2label)
            for k, l in id2label.items():
                if (
                    l == examples["action"][i]
                    or l == examples["object"][i]
                    or l == examples["location"][i]
                ):
                    label[k] = 1
            labels.append(label)
        examples["labels"] = labels

        # encode audio array
        audio_arrays = [x["array"] for x in examples["audio"]]
        audioinputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
        )
        examples["input_values"] = audioinputs["input_values"]

        textinputs = tokenizer(
            examples["transcription"], padding="max_length", truncation=True
        )
        examples["input_ids"] = textinputs["input_ids"]
        return examples

    encoded_dataset = fsc.map(
        preprocess_function,
        # remove_columns=[
        #     "audio",
        # ],
        batched=True,
        num_proc=8,
    )

    return encoded_dataset


def main(args):
    save_path = os.path.expanduser(args.save_path)

    if os.path.exists(save_path) and not args.overwrite:
        print(f"{save_path} exists already! not overwriting")
        print("use --overwrite flag if needed")
    else:
        dataset = encode_fsc_dataset(args.corpus_path)
        dataset.save_to_disk(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Saving encoded FSC dataset for intent classification"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite the saved arrow dataset"
    )
    parser.add_argument(
        "--corpus_path",
        default="~/corpora/FSC/",
        help="Path to the FSC corpus, default is ~/corpora/FSC/",
    )
    parser.add_argument(
        "--save_path",
        default=f"{PROJECT_ROOT}/datasets/fsc-ic",
        help="Where to save the arrow dataset",
    )
    parser.add_argument(
        "--balance", action="store_false", help="Not implemented for this file"
    )
    args = parser.parse_args()
    main(args)
