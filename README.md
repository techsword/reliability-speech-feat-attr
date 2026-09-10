# reliability-speech-feat-attr

This is the public repository for the experiment code for the Interspeech 2025 paper
[“On the reliability of feature attribution methods for speech classification”](https://arxiv.org/abs/2505.16406).

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{shen2025reliability,
    title = {On the reliability of feature attribution methods for speech classification},
    author = {Shen, Gaofei and Mohebbi, Hosein and Bisazza, Arianna and Alishahi, Afra and Chrupa{\l}a, Grzegorz},
    booktitle = {Proceedings of Interspeech 2025},
    pages = {266--270},
    year = {2025},
    doi = {10.21437/Interspeech.2025-1911},
    note = {arXiv:2505.16406},
    url = {https://arxiv.org/abs/2505.16406},
}
```

## Python Environment

`uv` is recommended to set up a Python environment.

PyTorch is provided through an extra, so select exactly one extra matching your accelerator. On a CPU-only machine run:

```bash
uv sync --extra cpu
```

On a machine with a CUDA 12.1 NVIDIA GPU run:

```bash
uv sync --extra cu121
```

The two extras conflict; install only one. See [uv documentation](https://docs.astral.sh/uv/guides/integration/pytorch/) for more.

## Repository layout

- `src/attributing/` — attribution scoring (`run_attribution.py`) and model wrappers (`model_helper.py`)
- `src/finetuning/` — fine-tuning scripts plus shared helpers (`utils.py`). `run_finetune.py` supports wav2vec2 and distilbert. `run_finetune_multihead.py` is wav2vec2-only.
- `src/preprocessing/` — dataset builders for Common Voice speaker-id (`dataset_speakerid.py`), LibriSpeech gender (`dataset_gender.py`), and FSC intent classification (`dataset_ic.py`)
- `data/` — external input files, currently `SPEAKERS.TXT` for LibriSpeech gender preprocessing (see `data/README.md`)
- `scripts/` — Slurm launcher wrappers for the attribution sweep and fine-tuning

Scripts resolve their data and model directories relative to their own location, so datasets and fine-tuned models live under `src/datasets/` and `src/models/` by default.

## Usage

Run scripts with `uv run`.

### Attribution

`src/attributing/run_attribution.py` computes attribution scores (saliency, integrated gradients, LIME, occlusion, feature ablation) for a trained model.

Running the script directly calls `submitit_main()`. That function ignores the command line, builds a hardcoded sweep over tasks, seeds, attribution methods, and input types, and submits the jobs to a Slurm cluster with `submitit`. It needs a Slurm cluster and the `submitit` package.

```bash
uv run src/attributing/run_attribution.py
```

`scripts/run_attribution_sweep.sh` wraps the same command and activates `.venv` first. It takes no arguments.

`parse_cmdline_args()` defines an argparse interface, but `__main__` does not call it. The sweep calls `main(args)` directly with a dictionary. To call `main(args)` yourself, supply these keys:

- `modeltype` — model type (default `wav2vec2`)
- `taskname` — task name (default `cv_genderid`)
- `seed` — model seed (default `42`)
- `datasplit` — data split (default `test`)
- `attrmethod` — attribution method (default `saliency`)
- `inputtype` — input type (default `input`)
- `subtask` — subtask for the FSC intent-classification model, or `None` (default `None`)
- `word_level` — word-level attribution, only for `lime` and `featureablation` (default `False`)
- `outputname` — output path (required; `main()` reads it to save the scores)

`main()` reads the model root from `src/models` and ignores `modelroot` and `overwrite`. The argparse options are vestigial: `--method` does not set `attrmethod`, and `parse_cmdline_args()` does not define `attrmethod`, `word_level`, or `outputname`.

### Fine-tuning

`src/finetuning/run_finetune.py` fine-tunes a wav2vec2 or distilbert classifier on a single-label audio task.

```bash
uv run src/finetuning/run_finetune.py --seed 42 --model_type wav2vec2 --taskname iemocap --num_epochs 5 --overwrite_output_dir
```

`scripts/run_finetune.sh` wraps this command, activates `.venv` first, and forwards its arguments to `run_finetune.py`.

`src/finetuning/run_finetune_multihead.py` fine-tunes a wav2vec2 model with three classification heads on the FSC intent-classification task (action, object, location). This script is wav2vec2-only: it always builds a wav2vec2 model and ignores `--model_type` for architecture selection.

```bash
uv run src/finetuning/run_finetune_multihead.py --seed 42 --model_type wav2vec2 --taskname fsc-ic
```

Common fine-tuning arguments for `run_finetune.py`: `--seed` (default `42`), `--batch_size` (default `64`), `--num_epochs` (default `10`), `--output_dir` (default `src/models`), `--model_type` (`wav2vec2` or `distilbert`), `--taskname`, `--no_cuda` (flag), `--overwrite_output_dir` (flag), `--freeze_embeddings` (flag), `--freeze_projection_layer` (flag). `run_finetune_multihead.py` accepts the same arguments, but its `--model_type` only changes the output directory name.

### Preprocessing

The dataset builders run as standalone scripts:

```bash
uv run src/preprocessing/dataset_speakerid.py
uv run src/preprocessing/dataset_gender.py
uv run src/preprocessing/dataset_ic.py
```

Notes:

- `dataset_speakerid.py` loads the gated `mozilla-foundation/common_voice_17_0` dataset at module import time, so its standalone command needs Hugging Face authentication and dataset access approval.
- `dataset_gender.py` reads `data/SPEAKERS.TXT` (the LibriSpeech speaker list). This file is not shipped with the repository; you must supply it. See `data/README.md`.
