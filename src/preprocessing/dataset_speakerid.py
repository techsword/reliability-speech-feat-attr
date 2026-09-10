import os
import re

import numpy as np
from datasets import (
    ClassLabel,
    concatenate_datasets,
    load_dataset,
)

SEED = 42
MAX_SECOND = 14.0
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets")


def balance(data, key):
    unique_labels, label_counts = np.unique(data[key], return_counts=True)
    min_count = label_counts.min()
    balanced_datasets = []
    for l in unique_labels:
        selected_indices = np.where(np.array(data[key]) == l)[0]
        np.random.shuffle(selected_indices)
        one_class_data = data.select(selected_indices[:min_count])
        balanced_datasets.append(one_class_data)
    balanced_data = concatenate_datasets(balanced_datasets, axis=0).shuffle(seed=SEED)
    return balanced_data


commonvoice = load_dataset(
    "mozilla-foundation/common_voice_17_0", "en", split="validated"
)

# choose samples with defined gender and speaker id info (with more than 200 freq)
unique_ids, counts = np.unique(np.array(commonvoice["client_id"]), return_counts=True)
freq_dict = dict(zip(unique_ids, counts))
freqs = np.vectorize(freq_dict.get)(np.array(commonvoice["client_id"]))

selected_indices = np.where(
    (
        (np.array(commonvoice["gender"]) == "male_masculine")
        | (np.array(commonvoice["gender"]) == "female_feminine")
    )
    & (freqs > 200)
)[0]
commonvoice = commonvoice.select(selected_indices)

# renaming and removing and casting
commonvoice = commonvoice.remove_columns(
    ["path", "up_votes", "down_votes", "accent", "age", "locale", "segment", "variant"]
)
commonvoice = commonvoice.rename_column("sentence", "text")


# keep only 24 unique speakers (12 male and 12 female)
male_speakers = commonvoice.select(
    np.where(np.array(commonvoice["gender"]) == "male_masculine")[0]
)
unique_male_ids, male_counts = np.unique(
    np.array(male_speakers["client_id"]), return_counts=True
)
del male_speakers
female_speakers = commonvoice.select(
    np.where(np.array(commonvoice["gender"]) == "female_feminine")[0]
)
unique_female_ids, female_counts = np.unique(
    np.array(female_speakers["client_id"]), return_counts=True
)
del female_speakers
# select speakers for which there are at least 300 samples
eligible_male_ids = unique_male_ids[male_counts > 300]
eligible_female_ids = unique_female_ids[female_counts > 300]

# select small subset of speakers randomly
np.random.seed(SEED)
selected_male_speakers = np.random.choice(eligible_male_ids, 20, replace=False)
selected_female_speakers = np.random.choice(eligible_female_ids, 20, replace=False)
# filter dataset based on selected speakers
selected_speakers = np.concatenate((selected_male_speakers, selected_female_speakers))
selected_indices = [
    i
    for i, client_id in enumerate(commonvoice["client_id"])
    if client_id in selected_speakers
]
commonvoice = commonvoice.select(selected_indices)


# filtering out long audios (> MAX_SECOND)
commonvoice = commonvoice.filter(
    lambda example: len(example["audio"]["array"]) / example["audio"]["sampling_rate"]
    <= MAX_SECOND,
    num_proc=8,
)

# balance by down sampling
commonvoice = balance(commonvoice, key="client_id")

# clean transcriptions and make them uppercase
chars_to_ignore_regex = r"[\,\?\.\!\-\;\:\"]"
commonvoice = commonvoice.map(
    lambda example: {
        "text": re.sub(chars_to_ignore_regex, "", example["text"]).strip().upper()
    }
)


# Process columns into classlabels
commonvoice = commonvoice.cast_column(
    "client_id",
    ClassLabel(
        names=commonvoice.unique("client_id"),
        num_classes=len(commonvoice.unique("client_id")),
    ),
)
commonvoice = commonvoice.cast_column(
    "gender",
    ClassLabel(
        names=commonvoice.unique("gender"),
        num_classes=len(commonvoice.unique("gender")),
    ),
)


def make_stratify_column(example):
    example["stratify_column"] = (
        str(example["gender"]) + "_" + str(example["client_id"])
    )
    return example


commonvoice = commonvoice.map(make_stratify_column)
commonvoice = commonvoice.cast_column(
    "stratify_column",
    ClassLabel(
        names=commonvoice.unique("stratify_column"),
        num_classes=len(commonvoice.unique("stratify_column")),
    ),
)

commonvoice = commonvoice.rename_column("client_id", "label")

commonvoice_dataset = commonvoice.train_test_split(
    test_size=0.2, stratify_by_column="stratify_column", seed=SEED
)


commonvoice_dataset.save_to_disk(os.path.join(DATASET_ROOT, "cv_spkid"))


# The following functions are for preparing commonvoice dataset to be used with
# Montreal Forced Aligner (MFA) for alignment.


def copy_audio_to_new_directory(
    dataset,
    new_dir=os.path.join(DATASET_ROOT, "commonvoice_subset"),
):
    import os
    import shutil

    from tqdm.auto import tqdm

    # Make sure new directory exists
    new_dir = os.path.expanduser(new_dir)
    os.makedirs(new_dir, exist_ok=True)

    for example in tqdm(dataset):
        audiofile = example["audio"]["path"]
        spklabel = dataset.features["label"].int2str(example["label"])

        new_audiofile = os.path.join(
            new_dir, f"{spklabel}/{os.path.basename(audiofile)}"
        )
        os.makedirs(os.path.dirname(new_audiofile), exist_ok=True)
        shutil.copyfile(audiofile, new_audiofile)

        # Write transcription to text file under the same name
        with open(new_audiofile.replace(".mp3", ".txt"), "w") as f:
            f.write(example["text"])


def flatten_tg_dir_structure():
    import glob
    import os
    import shutil

    from tqdm.auto import tqdm

    tg_root = os.path.join(DATASET_ROOT, "commonvoice_subset_aligned")
    new_dir = os.path.join(DATASET_ROOT, "cv_tg", "test")
    os.makedirs(new_dir, exist_ok=True)
    tg_files = glob.glob(f"{tg_root}/**/*.TextGrid", recursive=True)
    for tg_file in tqdm(tg_files):
        shutil.copyfile(tg_file, os.path.join(new_dir, os.path.basename(tg_file)))
