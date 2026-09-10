import argparse
import glob
import os

import datasets
import pandas as pd
from datasets import Audio, Dataset, load_from_disk
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_librispeech_as_gender_classification_dataset(
    dataset_audio_path="/corpora/LibriSpeech/LibriSpeech/",
    split="test-clean",
    overwrite=False,
    sr=16_000,
    no_samples=None,
):
    save_dir = os.path.join(
        PROJECT_ROOT, "datasets", f"librispeech-{split}-speaker-gender"
    )
    if no_samples:
        save_dir += f"-{no_samples}"

    if os.path.exists(save_dir) and not overwrite:
        try:
            hf_dataset = load_from_disk(save_dir)
            return hf_dataset
        except:
            print(f"cannot load dataset from {save_dir}, regenerating dataset")

    audio_absolute_path = glob.glob(
        f"{os.path.join(dataset_audio_path, split)}/**/*.flac", recursive=True
    )
    dataset_pd = pd.DataFrame(audio_absolute_path, columns=["absolute_path"])

    dataset_pd["filename"] = dataset_pd["absolute_path"].map(
        lambda x: os.path.basename(x)
    )
    dataset_pd["speaker_ID"] = dataset_pd["filename"].map(
        lambda x: int(x.split("-")[0])
    )
    dataset_pd["chapter_ID"] = dataset_pd["filename"].map(
        lambda x: int(x.split("-")[1])
    )

    # Loading speaker information
    speaker_metainfo = pd.read_csv(
        os.path.join(PROJECT_ROOT, "..", "data", "SPEAKERS.TXT"),
        sep="|",
        skiprows=12,
        skipinitialspace=True,
        engine="python",
        names=["ID", "gender", "split", "MINUTES", "NAME"],
    )
    speaker_metainfo
    # Look up gender based on speaker ID
    dataset_pd["gender"] = dataset_pd["speaker_ID"].map(
        lambda x: speaker_metainfo.loc[speaker_metainfo.ID == x, "gender"]
        .item()
        .strip()
    )

    # Find the minimum count between the two labels
    counts = dataset_pd["gender"].value_counts()
    min_count = min(counts)
    dataset_pd["file_ID"] = dataset_pd["filename"].map(lambda x: x.split(".")[0])

    # Reading transcriptions for the audio files
    unique_directories = (
        dataset_pd["absolute_path"].map(lambda x: os.path.dirname(x)).unique()
    )
    data = []
    for unique_directory in tqdm(unique_directories):
        txt_file = glob.glob(unique_directory + "/*.txt")[0]
        # print(txt_file)
        with open(txt_file, "r") as file:
            for line in file:
                # Strip any leading/trailing spaces
                line = line.strip()

                # Split on the first space (audio ID and transcription)
                audio_id, transcription = line.split(" ", 1)  # Split at first space
                data.append((audio_id, transcription))

    transcription_df = pd.DataFrame(data, columns=["file_ID", "text"])

    # Merge the transcriptions with the dataset
    dataset_pd = pd.merge(dataset_pd, transcription_df, on="file_ID", how="left")

    if no_samples:
        samples_per_label = int(no_samples / 2)
        if samples_per_label > min_count:
            print(
                f"this split does not have {samples_per_label} samples per label, using the balanced {min_count} spl instead"
            )
            # Select random samples from each label with the minimum count
            subset_dataset_pd = pd.concat(
                [
                    dataset_pd[dataset_pd["gender"] == label].sample(min_count)
                    for label in counts.index
                ]
            ).reset_index(drop=True)
        else:
            subset_dataset_pd = pd.concat(
                [
                    dataset_pd[dataset_pd["gender"] == label].sample(samples_per_label)
                    for label in counts.index
                ]
            ).reset_index(drop=True)
    else:
        subset_dataset_pd = pd.concat(
            [
                dataset_pd[dataset_pd["gender"] == label].sample(min_count)
                for label in counts.index
            ]
        ).reset_index(drop=True)

    # Load pandas dataframe as HF Dataset object
    hf_dataset = Dataset.from_pandas(subset_dataset_pd)

    class_labels = datasets.ClassLabel(names=sorted(hf_dataset.unique("gender")))
    hf_dataset = hf_dataset.rename_column("gender", "label")
    hf_dataset = hf_dataset.cast_column("label", class_labels)
    hf_dataset = hf_dataset.rename_column("absolute_path", "audio")
    hf_dataset = hf_dataset.cast_column("audio", Audio(sampling_rate=sr))
    hf_dataset = hf_dataset.remove_columns(["filename", "speaker_ID", "chapter_ID"])

    hf_dataset.save_to_disk(save_dir)

    return hf_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Load LibriSpeech dataset for gender classification."
    )
    parser.add_argument(
        "--dataset_audio_path",
        type=str,
        default="/corpora/LibriSpeech/LibriSpeech/",
        help="Path to the LibriSpeech dataset.",
    )
    parser.add_argument(
        "--split", type=str, default="test-clean", help="Dataset split to use."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing dataset."
    )
    parser.add_argument(
        "--sr", type=int, default=16000, help="Sampling rate for audio files."
    )
    parser.add_argument(
        "--no_samples", type=int, default=None, help="Number of samples to use."
    )

    args = parser.parse_args()

    dataset = load_librispeech_as_gender_classification_dataset(
        dataset_audio_path=args.dataset_audio_path,
        split=args.split,
        overwrite=args.overwrite,
        sr=args.sr,
        no_samples=args.no_samples,
    )

    print(dataset)


if __name__ == "__main__":
    main()
