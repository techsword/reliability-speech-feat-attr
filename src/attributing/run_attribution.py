import argparse
import logging
import os
import pickle
from typing import NamedTuple

import torch
from datasets import Audio, load_from_disk
from model_helper import (
    Wav2Vec2ForMultiLabelClassification,
    WrapperModel,
    WrapperModelMultiHead,
    choose_attr_method,
)
from textgrids import TextGrid
from tqdm.auto import trange
from transformers import (
    Wav2Vec2ForSequenceClassification,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wdseg_tg = NamedTuple(
    "wdseg_tg", [("xmin", float), ("xmax", float), ("SegAScr", int), ("text", str)]
)


def read_wdseg(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    alignments = []
    for line in lines[1:-1]:
        xmin, xmax, SegAScr, label = line.split()
        alignments.append(
            wdseg_tg(float(xmin) / 100, float(xmax) / 100, int(SegAScr), label)
        )
    return alignments


def attribute(
    attr_func, example, batch_size=8, additional_forward_args=None, word_level=False
):
    C = example["label"]
    attr_input = attr_func.forward_func.process_input(example)
    if attr_func.get_name() == "Saliency":
        scores = attr_func.attribute(
            attr_input,
            target=C,
            abs=False,
            additional_forward_args=additional_forward_args,
        )
    else:
        # First construct baseline based on the input type
        baselines = attr_func.forward_func.process_input(example, baseline_mode=True)
        input_shape = attr_input.shape

        if attr_func.get_name() == "Integrated Gradients":
            scores = attr_func.attribute(
                attr_input,
                baselines=baselines,
                target=C,
                internal_batch_size=batch_size,
                additional_forward_args=additional_forward_args,
            )
        elif attr_func.get_name() == "Occlusion":
            # Change sliding window shape based on the input type
            if len(input_shape) == 3:
                # Set sliding window shape to (1, feature_dim) for spec and embedding input
                sliding_window_shapes = (1, input_shape[2])
                strides = 1
            else:
                # Set sliding window shape to 320 samples for waveform input
                sliding_window_shapes = (320,)
                strides = 320

            scores = attr_func.attribute(
                attr_input,
                baselines=baselines,
                target=C,
                sliding_window_shapes=sliding_window_shapes,
                strides=strides,
                perturbations_per_eval=batch_size,
                additional_forward_args=additional_forward_args,
            )
        else:
            if not word_level:
                if len(input_shape) == 3:
                    # Set feature mask to be the feature dimension for spec and embedding input
                    feature_mask = torch.stack(
                        [
                            torch.full((input_shape[2],), i, device=device)
                            for i in range(input_shape[1])
                        ]
                    ).unsqueeze(0)
                else:
                    # Construct feature mask so every 10ms frame is assigned the same feature
                    frame_size = 160
                    number_of_frames = input_shape[1] // frame_size
                    feature_mask = torch.repeat_interleave(
                        torch.arange(number_of_frames, device=device),
                        repeats=frame_size,
                    )
                    # Pad the feature mask to the same length as the input at the end
                    feature_mask = torch.cat(
                        (
                            feature_mask,
                            torch.full(
                                (input_shape[1] % frame_size,),
                                number_of_frames,
                                device=device,
                            ),
                        )
                    ).unsqueeze(0)

            elif word_level:
                feature_mask = torch.ones_like(
                    attr_input, dtype=torch.long, device=device
                )
                # Read the segmentation files
                if "textgrid_path" in example.keys():
                    alignment = TextGrid(example["textgrid_path"])
                    interval_tier = "words"
                    intervals = alignment[interval_tier]
                elif "wdseg" in example.keys():
                    intervals = read_wdseg(example["wdseg"])
                else:
                    raise ValueError(
                        "No segmentation file found for word-level attribution"
                    )
                # Group intervals into features using feature mask
                for i, interval in enumerate(intervals):
                    # Compute the start and end index of the interval
                    start = int(interval.xmin // 0.02)
                    end = int(interval.xmax // 0.02)
                    if len(attr_input.shape) == 2:
                        # Times 320 samples for waveform input
                        start = int(start * 320)
                        end = int(end * 320)
                    # Assign the feature mask to the interval
                    feature_mask[:, start:end] = i
                # Check if there's remaining frames
                if end < feature_mask.shape[1]:
                    feature_mask[:, end:] = i + 1

            if (
                attr_func.get_name() == "Lime"
                or attr_func.get_name() == "Feature Ablation"
            ):
                scores = attr_func.attribute(
                    attr_input,
                    baselines=baselines,
                    target=C,
                    feature_mask=feature_mask,
                    perturbations_per_eval=batch_size,
                    additional_forward_args=additional_forward_args,
                )

            else:
                raise NotImplementedError(
                    f"Attribution method {attr_func.get_name()} not implemented"
                )

    return scores.detach().cpu().numpy().real


def parse_cmdline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modelroot", type=str, default="models", help="Root directory for the model"
    )
    parser.add_argument("--modeltype", type=str, default="wav2vec2", help="Model type")
    parser.add_argument("--taskname", type=str, default="cv_genderid", help="Task name")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the model")
    parser.add_argument(
        "--datasplit", type=str, default="test", help="Data split to use"
    )
    parser.add_argument(
        "--method", type=str, default="saliency", help="Attribution method to use"
    )
    parser.add_argument(
        "--inputtype", type=str, default="input", help="Input type for the model"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--subtask",
        type=str,
        default="action",
        help="Subtask for the FSC intent classification model",
    )

    args = parser.parse_args()

    # turn args into a dictionary
    args = vars(args)
    return args


def main(args):
    # Set up the paths
    PROJECTROOT = os.path.expanduser("~/work_dir/grad-attr-speech")
    MODELROOT = os.path.join(PROJECTROOT, "models")
    DATAROOT = os.path.join(PROJECTROOT, "datasets")

    # Set up the parameters
    MODELTYPE = args.get("modeltype", "wav2vec2")
    TASKNAME = args.get("taskname", "cv_genderid")
    SEED = args.get("seed", 42)
    DATASPLIT = args.get("datasplit", "test")
    ATTRMETHOD = args.get("attrmethod", "saliency")
    INPUTTYPE = args.get("inputtype", "input")
    WORD_LEVEL = args.get("word_level", False)
    # Set WORD_LEVEL to false if not using the LIME or Feature Ablation method
    if ATTRMETHOD not in ["lime", "featureablation"]:
        assert not WORD_LEVEL, (
            "Word level attribution only supported for LIME and Feature Ablation"
        )
    SUBTASK = args.get(
        "subtask", None
    )  # Should be a string or None, only needed for multi-label classification
    # Make sure SUBTASK is none if not using the FSC model
    if "fsc" not in TASKNAME:
        SUBTASK = None

    # Set up the save directory and output name
    taskname = TASKNAME.replace("_", "-")
    if SUBTASK:
        taskname = f"{taskname}-{SUBTASK}"

    # Load model and processor
    modelname = (
        f"{MODELROOT}/{MODELTYPE}/{TASKNAME}/{TASKNAME}_frozen_projection_seed_{SEED}"
    )
    logging.info(f"Loading model from {modelname}")
    assert os.path.exists(modelname), f"Model {modelname} not found"

    if "fsc" not in modelname:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(modelname)
    else:
        model = Wav2Vec2ForMultiLabelClassification.from_pretrained(modelname)

    logging.info(f"Loaded model {modelname}")
    logging.info(f"Using device {device}")
    logging.info(f"Saving attribution scores to {args['outputname']}")

    model.to(device)
    model.eval()

    # Load dataset
    if "genderid" in TASKNAME:
        dataset_taskname = "cv_spkid"
    else:
        dataset_taskname = TASKNAME
    datasetname = (
        f"{DATAROOT}/{dataset_taskname}/{DATASPLIT}"
        if DATASPLIT
        else f"{DATAROOT}/{dataset_taskname}"
    )
    assert os.path.exists(datasetname), f"Dataset {datasetname} not found"

    dataset = load_from_disk(datasetname)
    logging.info(f"Loaded dataset {datasetname}")

    if "genderid" in TASKNAME:
        # Rename the label column to spkid and rename the gender column to label
        rename_dict = {"label": "spkid", "gender": "label"}
        dataset = dataset.rename_columns(rename_dict)
    elif "fsc" in TASKNAME:
        # Make sure subtask is defined, else set it to the action column
        if SUBTASK is None:
            SUBTASK = "action"
        # Rename subtask column to label
        dataset = dataset.rename_column(SUBTASK, "label")

    # Cast the audio column to 16000 Hz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    # Add textgrid path to dataset
    if "cv" in taskname:
        # Add textgrid path to the dataset
        commonvoice_tg_root = os.path.expanduser(
            "~/work_dir/grad-attr-speech/datasets/cv_tg/test"
        )

        filenames = [x["audio"]["path"] for x in dataset]
        textgrid_path = [
            os.path.join(commonvoice_tg_root, x.replace(".mp3", ".TextGrid"))
            for x in filenames
        ]
        dataset = dataset.add_column("textgrid_path", textgrid_path)

    logging.info(f"Preprocessed dataset {datasetname}")

    # Initialize the model wrapper
    if "fsc" in modelname:
        wrapped_model = WrapperModelMultiHead(model, input_type=INPUTTYPE)
        classifier_to_idx = {"action": 0, "object": 1, "location": 2}
        additional_forward_args = (classifier_to_idx[SUBTASK],)

        # wrapped_model = lambda x: wrapped_model.forward(x, classifier_idx=classifier_to_idx[SUBTASK])
    else:
        wrapped_model = WrapperModel(model, input_type=INPUTTYPE)
        additional_forward_args = None

    logging.info(f"Wrapped model with input type {INPUTTYPE}")
    # Choose the attribution method
    attr_func = choose_attr_method(ATTRMETHOD, wrapped_model)
    logging.info(f"Chose attribution method {ATTRMETHOD}")

    attr_scores = []
    no_samples = len(dataset)

    for i in trange(no_samples, desc="Attributing samples from dataset"):
        example = dataset[i]
        # Make sure the alignment file exists
        if WORD_LEVEL:
            if "textgrid_path" in example.keys():
                alignment_file = example["textgrid_path"]
            elif "wdseg" in example.keys():
                alignment_file = example["wdseg"]
            else:
                raise ValueError(
                    "No segmentation file found for word-level attribution"
                )
            if not os.path.isfile(alignment_file):
                logger.warning(
                    f"Segmentation file not found for {alignment_file}. Skipping..."
                )
                continue
        example_attr_score = attribute(
            attr_func,
            example,
            additional_forward_args=additional_forward_args,
            word_level=WORD_LEVEL,
        )
        # Save the attribution scores together with some metadata from the example
        # keep everything but the audio array

        example_attr_metadata = {k: v for k, v in example.items() if k != "audio"}
        example_attr_metadata["index"] = i
        example_attr_metadata["path"] = example["audio"]["path"]
        example_attr_metadata["scores"] = example_attr_score
        attr_scores.append(example_attr_metadata)

    # Save the attribution scores
    overall_metadata = {
        "model": modelname,
        "taskname": taskname,
        "dataset": datasetname,
        "attr_method": ATTRMETHOD,
        "input_type": INPUTTYPE,
        "seed": SEED,
    }

    with open(args["outputname"], "wb") as f:
        pickle.dump((overall_metadata, attr_scores), f)
    print(f"Saved attribution scores to {args['outputname']}")


def set_output_name(args):
    # Set up the parameters
    MODELTYPE = args.get("modeltype", "wav2vec2")
    TASKNAME = args.get("taskname", "cv_genderid")
    SEED = args.get("seed", 42)
    DATASPLIT = args.get("datasplit", "test")
    ATTRMETHOD = args.get("attrmethod", "saliency")
    INPUTTYPE = args.get("inputtype", "input")
    WORD_LEVEL = args.get("word_level", False)

    SUBTASK = args.get(
        "subtask", None
    )  # Should be a string or None, only needed for multi-label classification

    # Set up the save directory and output name
    fulltaskname = TASKNAME.replace("_", "-")
    if SUBTASK:
        fulltaskname = f"{fulltaskname}-{SUBTASK}"

    savedir = f"{SAVEROOT}/{MODELTYPE}/{TASKNAME}"
    os.makedirs(savedir, exist_ok=True)
    outputname = f"{savedir}/{fulltaskname}_{DATASPLIT}_{ATTRMETHOD}_{INPUTTYPE}_{SEED}_attr-scores.pkl"
    if WORD_LEVEL:
        outputname = outputname.replace(".pkl", "_word-level.pkl")
    return outputname


def submitit_main():
    import os

    import submitit

    global PROJECTROOT, SAVEROOT, outputname
    PROJECTROOT = os.path.expanduser("~/work_dir/grad-attr-speech")
    SAVEROOT = os.path.join(PROJECTROOT, "attribution_scores")

    # seeds = [42, 666, 2024]
    seeds = [42, 666, 2024, 2025, 2026, 2027, 0, 1, 2]
    attr_methods = ["saliency", "ig", "lime", "occlusion", "featureablation"]
    input_types = ["spec", "embedding", "input"]
    modeltype = "wav2vec2"
    overwrite = False

    all_args = []
    # For loop to gather arguments for jobs to calculate attribution scores for single-label classification models
    for taskname in ["speech_commands", "cv_spkid", "cv_genderid", "fsc-ic"]:  # iemocap
        for attr_method in attr_methods:
            for input_type in input_types:
                for seed in seeds:
                    batch_size = 16 if "ig" in attr_method else 64
                    if "fsc" in taskname:
                        subtasks = ["action", "object", "location"]
                    else:
                        subtasks = [None]

                    for subtask in subtasks:
                        args = {
                            "modeltype": modeltype,
                            "taskname": taskname,
                            "datasplit": "test",
                            "attrmethod": attr_method,
                            "inputtype": input_type,
                            "seed": seed,
                            "batch_size": batch_size,
                            "overwrite": overwrite,
                            "subtask": subtask,
                            "word_level": False,
                        }

                        outputname = set_output_name(args)
                        args["outputname"] = outputname
                        all_args.append(args)

                        if (
                            attr_method in ["lime", "featureablation"]
                            and taskname != "speech_commands"
                        ):
                            args_copy = args.copy()
                            args_copy["word_level"] = True
                            outputname = set_output_name(args_copy)
                            args_copy["outputname"] = outputname
                            all_args.append(args_copy)

    submitting_args = []
    for args in all_args:
        if os.path.exists(args["outputname"]) and not args["overwrite"]:
            logger.info(f"Output file {args['outputname']} exists, skipping")
            continue
        else:
            # Filter out feature ablation but word level is False
            # if args["attrmethod"] == "featureablation" and not args["word_level"]:
            #     logger.info(f"Skipping feature ablation with word level {args['word_level']}")
            #     continue
            submitting_args.append(args)

    logger.info(f"Number of jobs: {len(submitting_args)}")
    # submitit_logdir = "logs/submitit_logs/grad_attr"
    submitit_logdir = f"{PROJECTROOT}/slurm_logs/attribution/%A"
    executor = submitit.AutoExecutor(folder=submitit_logdir)

    executor.update_parameters(
        timeout_min=180,
        slurm_partition="GPU",
        nodes=1,
        slurm_array_parallelism=7,
        slurm_gres="gpu:A40",
        name="attributing",
    )
    with executor.batch():
        for args in submitting_args:
            executor.submit(main, args)


if __name__ == "__main__":
    # args = parse_cmdline_args()

    # main(args)
    submitit_main()
