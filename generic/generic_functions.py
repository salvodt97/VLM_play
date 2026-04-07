from final_model import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Carica il modello da huggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Trova i file *.safetensors per caricare i pesi, e poi li mette in dizionario 
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Carica la configurazione del modello
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # crea il modello con quella configurazione
    model = PaliGemmaForConditionalGeneration(config).to(device)
    load_result = model.load_state_dict(tensors, strict=False)
    # if load_result.missing_keys or load_result.unexpected_keys:
    #     print("WARNING: state_dict mismatch while loading the checkpoint.")
    #     print(f"Missing keys: {len(load_result.missing_keys)}")
    #     if load_result.missing_keys:
    #         print("First missing keys:", load_result.missing_keys[:10])
    #     print(f"Unexpected keys: {len(load_result.unexpected_keys)}")
    #     if load_result.unexpected_keys:
    #         print("First unexpected keys:", load_result.unexpected_keys[:10])
    # copia i pesi
    model.tie_weights()

    return (model, tokenizer)
