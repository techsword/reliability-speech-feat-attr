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
    year = {2025},
    note = {arXiv:2505.16406},
    url = {https://arxiv.org/abs/2505.16406},
}
```

## Python Environment

`uv` is recommended to set up a Python environment.

First run `uv sync` to install most of the required Python packages. Then install PyTorch according to your accelerator setup. If you just have a standard NVIDIA GPU then the following line should suffice. See [uv documentation](https://docs.astral.sh/uv/guides/integration/pytorch/) for more.

```bash
UV_TORCH_BACKEND=auto uv pip install torch
```

## Repository layout

- `src/attributing/` — attribution scoring (`run_attribution.py`) and model wrappers (`model_helper.py`)
- `src/finetuning/` — fine-tuning scripts for wav2vec2 and distilbert (`run_finetune.py`, `run_finetune_multihead.py`) plus shared helpers (`utils.py`)
- `src/preprocessing/` — dataset builders for Common Voice speaker-id (`dataset_speakerid.py`), LibriSpeech gender (`dataset_gender.py`), and FSC intent classification (`dataset_ic.py`)

Scripts resolve their data and model directories relative to their own location, so datasets and fine-tuned models live under `src/datasets/` and `src/models/` by default.

## Usage

Run scripts with `uv run`.

### Attribution

`src/attributing/run_attribution.py` computes attribution scores (saliency, integrated gradients, LIME, occlusion, feature ablation) for a trained model. Running it directly launches the `submitit_main` job sweep, which submits Slurm jobs. The argparse interface defines these options:

- `--modelroot` — model root directory (default `models`)
- `--modeltype` — model type (default `wav2vec2`)
- `--taskname` — task name (default `cv_genderid`)
- `--seed` — model seed (default `42`)
- `--datasplit` — data split (default `test`)
- `--method` — attribution method (default `saliency`)
- `--inputtype` — input type (default `input`)
- `--overwrite` — overwrite existing files (flag)
- `--subtask` — subtask for the FSC intent-classification model (default `action`)

```bash
uv run src/attributing/run_attribution.py
```

### Fine-tuning

`src/finetuning/run_finetune.py` fine-tunes a wav2vec2 or distilbert classifier on a single-label audio task.

```bash
uv run src/finetuning/run_finetune.py --seed 42 --model_type wav2vec2 --taskname iemocap --num_epochs 5 --overwrite_output_dir
```

`src/finetuning/run_finetune_multihead.py` fine-tunes a wav2vec2 model with three classification heads on the FSC intent-classification task (action, object, location).

```bash
uv run src/finetuning/run_finetune_multihead.py --seed 42 --model_type wav2vec2 --taskname fsc-ic
```

Common fine-tuning arguments: `--seed` (default `42`), `--batch_size` (default `64`), `--num_epochs` (default `10`), `--output_dir` (default `src/models`), `--model_type` (`wav2vec2` or `distilbert`), `--taskname`, `--no_cuda` (flag), `--overwrite_output_dir` (flag), `--freeze_embeddings` (flag), `--freeze_projection_layer` (flag).

### Preprocessing

The dataset builders run as standalone scripts:

```bash
uv run src/preprocessing/dataset_speakerid.py
uv run src/preprocessing/dataset_gender.py
uv run src/preprocessing/dataset_ic.py
```
