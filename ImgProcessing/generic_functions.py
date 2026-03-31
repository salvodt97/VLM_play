from typing import Dict, List, Optional, Tuple, Union, Iterable
import numpy as np
from PIL import Image


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