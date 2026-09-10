# External data

This directory holds external input files that are not shipped with the
repository.

## `SPEAKERS.TXT`

`src/preprocessing/dataset_gender.py` reads `data/SPEAKERS.TXT`, the LibriSpeech
speaker list. Download it from the LibriSpeech corpus and place it here:

- LibriSpeech: <https://www.openslr.org/12>

The file is required only for the LibriSpeech gender preprocessing step. The
repository does not redistribute it.
