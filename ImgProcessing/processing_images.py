from typing import List
import numpy as np
import torch
from PIL import Image

from ImgProcessing.generic_functions import process_image

# Valori di huggingface per la normalizzazione delle immagini, usati anche in CLIP
IMAGE_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGE_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    # Qui image tokens sono i placeholder che vengono inseriti prima dei token del testo
    # <bos_token> indica il token di inizio sequenza, e viene inserito prima dei token del testo, ma dopo i token dell'immagine
    # prefiz_prompt è il prompt dell'utente
    # \n è il token separatore (newline), che va tokenizzato separatamente dal resto
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


class ImageProcessor:
    
    # Placeolder per i token delle immagini, affiancati ai token di testo
    IMAGE_TOKEN = "<image>"
    
    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        self.image_seq_lenght = num_image_tokens
        self.image_size = image_size
        
        # Il checkpoint PaliGemma include gia' <image>; lo aggiungiamo solo se manca.
        if tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN) == tokenizer.unk_token_id:
            tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
            tokenizer.add_special_tokens(tokens_to_add)
        ############# EXTRA SATNDARD TOKENS NOT USEFUL ######################
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]   # Tokens for object detection (bounding boxes) 
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]   # Tokens for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        #########################################################################
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        
        self.tokenizer = tokenizer
        
    def __call__(self, text: List[str], images: List[Image.Image], padding: str = "longest", truncation: bool = True) -> dict:
        # Esempio con un'immagine e un prompt 
        assert len (images) == 1 and len(text) == 1, f"received {len(images)} images and {len(text)} prompts"
        
        pixel_values = process_image(
            images,
            size = (self.image_size, self.image_size),
            resample = Image.Resampling.BICUBIC,
            rescale_factor = 1/255.0,
            image_mean = IMAGE_STANDARD_MEAN,
            image_std = IMAGE_STANDARD_STD,
        )

        # Pixel values è un vettore di nunpy. così ottengo invece dei batches, e poi converto in formato torch
        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.tensor(pixel_values)

        # crea i token per il testo, e i placeholder dell'immagine
        input_string = [
            add_image_tokens_to_prompt(
                prefix_prompt = prompt,
                bos_token = self.tokenizer.bos_token,
                image_seq_len = self.image_seq_lenght,
                image_token = self.IMAGE_TOKEN
            )
            for prompt in text
        ]
        
        # non sono veri e propri embedding, ma input_id e attention:mask
        inputs = self.tokenizer(input_string, return_tensors="pt", padding=padding, truncation=truncation)
        
        return_data = {"pixel_values": pixel_values, **inputs}
        
        return return_data
