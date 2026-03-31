from typing import Dict, List, Optional, Tuple, Union, Iterable
import nunmpy as np
import torch
from PIL import Image

# Valori di huggingface per la normalizzazione delle immagini, usati anche in CLIP
IMAGE_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGE_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    # Qui image tokens sono i placeholder che vengono inseriti prima dei token del testo
    # <bos_token> indica il token di inizio sequenza, e viene inserito prima dei token del testo, ma dopo i token dell'immagine
    # prefiz_prompt è il prompt dell'utente
    # \n è il token separatore (newline), che va tokenizzato separatamente dal resto
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def resize(
    image: Image.Image, 
    size: Tuple[int, int], 
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
    ) -> np.ndarray:
    height, width = size
    resized_image = image.resize((width, height), resample=resample, reducing_gap=reducing_gap)
    
    return resized_image


def rescale(image: np.ndarray, scale: float, dtype: np.dtype = np.float32) -> np.ndarray:
    rescale_image = image * scale
    rescaled_image = rescale_image.astype(dtype)
    
    return rescaled_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    mean = np.array(mean, dtype = image.dtype)
    std = np.array(std, dtype = image.dtype)  
    normalized_image = (image - mean) / std
    
    return normalized_image


def process_image(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample:  Image.Resampling = None, 
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [resize(image=image, size=(height, width), resample=resample) for image in images]
    images = [np.array(image) for image in images]
    images = [rescale(image, scale=rescale_factor) for image in images]
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    images = [image.transpose(2, 0, 1) for image in images] # channel dimension to be first dimension [channel, hight, width]
    
    return images



class ImageProcessor:
    
    # Placeolder per i token delle immagini, affiancati ai token di testo
    IMAGE_TOKEN = "<image>"
    
    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        self.image_seq_lenght = num_image_tokens
        self.image_size = image_size
        
        # Gemma Tokenizer
        tokens_to_add = {"additional_special_tokens:" [self.IMAGE_TOKEN]}
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
        
        return_data = {pixel_values: pixel_values, **inputs}
        
        return return_data