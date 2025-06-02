# reliability-speech-feat-attr
This is the public repository for the experiment code for the paper “ On the reliability of feature attribution methods for speech classification”

## Python Environment
`uv` is recommended to setup a python environment

First run `uv sync` to install most of the required python packages. Then install PyTorch according to your accelerator setup. If you just have a standard nVidia GPU then the following line should suffice. See [uv documnetation](https://docs.astral.sh/uv/guides/integration/pytorch/) for more.

``` bash
UV_TORCH_BACKEND=auto uv pip install torch
```

