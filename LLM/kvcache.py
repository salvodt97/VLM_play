import torch
from typing import Optional, Tuple, List


class KVCache():
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # ritrono la sequence lenght
            return self.key_cache[0].shape[-2]
        
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # stiamo creando la KVCache di quel layer
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # concateniamo quello che abbiamo col nuovo token arriva, lungo la dimensione della sequence lenght
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim = -2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim = -2)
            
        return self.key_cache[layer_idx], self.value_cache[layer_idx]