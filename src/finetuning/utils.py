import os
import random

import numpy as np
import torch
import transformers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_last_checkpoint(model_name_or_path):
    from transformers import (
        AutoModelForAudioClassification,
    )

    if not os.path.exists(model_name_or_path):
        if isinstance(
            model_name_or_path,
            transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification,
        ):
            model = model_name_or_path
            return model
        try:
            print(f"Loading model from huggingface hub with {model_name_or_path}")
            model = AutoModelForAudioClassification.from_pretrained(model_name_or_path)
        except Exception:
            print(f"Loading model from huggingface hub failed")
            model = model_name_or_path

    else:
        ckpt_dirs = os.listdir(model_name_or_path)
        ckpt_dirs = [x for x in ckpt_dirs if "checkpoint" in x]
        ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split("-")[1]))
        last_ckpt = ckpt_dirs[-1]
        model_path = os.path.join(model_name_or_path, last_ckpt)
        model = AutoModelForAudioClassification.from_pretrained(model_path)
    return model


def compare_models(model_1, model_2, model_component=None):
    models_differ = 0
    state_dict1, state_dict2 = model_1.state_dict(), model_2.state_dict()
    for state_dict_key in state_dict1.keys():
        if model_component and model_component not in state_dict_key:
            continue
        if torch.equal(state_dict1[state_dict_key], state_dict2[state_dict_key]):
            print(
                f"{state_dict_key} in {model_1.name_or_path} and {model_2.name_or_path} are the same"
            )
            pass
        else:
            models_differ += 1
            print("Mismtach found at", state_dict_key)
    if models_differ == 0:
        print(
            f"Model components including {model_component} match perfectly! :)"
        ) if model_component else print("Models match perfectly! :)")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
