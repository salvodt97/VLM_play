from msilib import sequence
from optparse import Option
from re import S
from tkinter import HIDDEN
from turtle import position

from numpy import dtype, pad
import torch
from torch import device, nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, List, final
import math

from ViT.siglip import SiglipVisionConfig, SiglipVisionModel



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
            self.value_cache.append(self.value_cache)
        else:
            # concateniamo quello che abbiamo col nuovo token arriva, lungo la dimensione della sequence lenght
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim = -2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim = -2)
            
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class GemmaConfig:
    def __init__(
        # i valori hardcodadi vengono da PaliGemma di huggingface
        self,
        vocab_size,
        hidden_size,                    # dimensione degli embedding di ciascun token
        intermediate_size,              # parte del feed forward network
        num_hidden_layers,
        num_attention_heads,            # heads per le query
        num_key_value_heads,            # heads per le key e value
        head_dim = 256,
        max_position_embeddings = 8192, 
        rms_norm_eps = 1e-6,            # è per la normalizzazione rms
        rope_theta = 100000.0,          # base frequency per la rotatory positional encoding
        attention_bias = False,         # se usare o meno i bias nelle matrizi per Q, K e V
        attention_dropout = 0.0,
        pad_token_id = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id



class PaliGemmaConfig:
    # contiene le configurazioni sia del ViT, sia di Gemma
    def __init__(
        self,
        vision_config = None,
        text_config = None,
        ignore_index = -100,
        image_token_index = 256000, # Per i placehoder dei token delle immagini, che vanno inseriti prima dei token del testo
        vocab_size = 257152,        
        projection_dim = 2048,      # dimensione in cui va fatto il resize dell'immagine     
        hidden_size = 2048,         # embedding size della LLM
        pad_token_id = None,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id
        
        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config
        
        self.text_config = GemmaConfig(**text_config, pad_token_id = self.pad_token_id)
        self.vocab_size = self.text_config.vocab_size
        
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim
     
     
     
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    
    def _norm(self, x):
        # Aggiungo eps per evitare divisione per 0 (o quasi 0)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
        
    def forward(self, x):
        output = self._norm(x)
        output = output * (1 + self.weight.float())
        
        return output.type_as(x)
    
    

class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # layer usato dalla funzione di attivazione di Gemma
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias = False)
        # layer che espande l'embedding
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias = False)
        # layer che ritorna alla dimensione iniziale
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias = False)
        
        
    def forward(self, x):
        
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
    
    
    
class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx  # serve per sapere quale kvcache usare, perchè pgni layer ha la sua kvcache
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_casual = True
        
        assert self.hidden_size % self.num_heads == 0
        # diverso da siglip, il numero di head per le query è maggiore del numero di head per K e V
        # Es: number of heads = 8, hidden_size = 1024, head_dim = 1024/8 = 128
        # Wq = [1024, 8 * 128] = [1024, 1024]
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias = config.attention_bias)
        # Wk = [1024, 1 * 128] = [1024, 1024]
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        # Wv = [1024, 1 * 128] = [1024, 1024]
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias = config.attention_bias)
        
        self.rotary_emb = GemmaRotatoryEmbedding(
            self.head_dim,
            max_position_embeddings = self.max_position_embeddings,
            base = self.rope_theta,
        )
        
        
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            kv_cache: Optional[KVCache] = None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            bsz, q_len = hidden_states.size()
            query_states = self.q_proj(hidden_states)
            key_states = self.q_proj(hidden_states)
            value_states = self.q_proj(hidden_states)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            # si aggiunge qualche info per query e key che codifica la loro posizione
            cos, sin = self.rotary_emb(value_states, position_ids, seq_len = None)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
            if kv_cache is not None:
                key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)
    

class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config = config, layer_idx = layer_idx)
        self.mlp = GemmaMLP(config)
        
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            kv_cache: Optional[KVCache] = None,
        ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)  
            hideen_states, _ = self.self_attn(
                hidden_states = hidden_states,
                attention_mask = attention_mask,
                position_ids = position_ids,
                kv_cache = kv_cache,
            )
            hidden_states = residual + hideen_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
            return hidden_states
        
        
        
class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)


    def get_input_embeddings(self):
        return self.embed_tokens
    
    
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        hidden_states = inputs_embeds
        normalizer = torch.Tensor(self.config.hidden_size ** 0.5, dtype = hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states = hidden_states,
                attention_mask = attention_mask,
                position_ids = position_ids,
                kv_cache = kv_cache,
            )
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states



class GemmaLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        
        
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight


    def forard(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        outputs = self.model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            kv_cache = kv_cache,
        )
        
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        return_data = {"logits": logits}
        
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias = True)
        
    def forward(self, image_features) -> torch.Tensor:
        projected_image_features = self.linear(image_features)
        return projected_image_features
    


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        
        language_model = GemmaLLM(config.text_config)
        self.language_model = language_model
        
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        
        
        def tie_weights(self):
            # è una tecnica che permette di usare i pesi di un layer in un altro layer
            # Nello specifico, permette di condividere  i pesi tra il layer che crea gli embedding (in input alla parte
            # decoder della LLM) e il layer lineare che proietta i contextualized embeddings nello spazio dei token (vocab size)
            # Questo perchè i 2 layer fanno la stessa operazione ma all'opposto
            return self.language_model.tie_weights()
        
        
        def _merge_inputs_ids_and_image_features(
            self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache: Optional[KVCache] = None
        ):
            _, _, embed_dim = image_features.shape
            batch_size, sequence_lenght = input_ids.shape
            dtype, device = inputs_embeds.dtype, inputs_embeds.device
            scaled_image_features = image_features / (self.config.hidden_size ** 0.5)
            # combina ora gli embedding delle immagini, i ext tokens e la mask, tutto con i token di padding
            # sequence_lenght è il numero di iputs_ids (concatenazione di placeolder delle immagini, token di testo, prompt)
            # quindi creo sequence_lenght embeddings di dimensione embed_dim
            final_embedding = torch.zeros(batch_size, sequence_lenght, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            
            text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
            image_mask = input_ids == self.config.image_token_index
            pad_mask = input_ids == self.pad_token_id   # non la useremo
            # si espandono le maschere, aggiumgendo una dimensione 
            text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
            image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
            pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
            
            final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
            # sostituisco i placeolder
            final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
            final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding) # metto 0 perchè non ho padding
            
            #### GESTIONE KVCACHE ####
            dtype, device, inputs_embeds.dtype, inputs_embeds.device
            min_dtype = torch.finfo(dtype).min
            q_len = inputs_embeds.shape[1]
            
            if kv_cache is None or kv_cache.num_items() == 0:
                # fase di prefill, non maschero nessun token
                # non aggiungo nessun valore di - inf (per dove voglio attention sore = 0) perchè, stando al paper di paligemma
                # si vogliono tenere tutti i token del prompt, essendo che descrivono il task
                # ho la casualità, quindi, solo nella generazione dei token, non nel prefill della kvcache
                casual_mask = torch.full((batch_size, q_len, q_len), fill_value=0, device=device)
            else:
                # qui genero tokens, quindi query (matrice Q) contiene una sola riga
                assert q_len == 1
                kv_len = kv_cache.num_items() + q_len
            # Aggiungo la head dimension
            casual_mask = casual_mask.unsqueeze(1)
            
            if kv_cache is not None and kv_cache.num_items() > 0:
                position_ids = attention_mask.cumsum(-1)[:, -1]
                if position_ids.dim() == 1:
                    position_ids = position_ids.unsqueeze(0)
            else:
                 position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)
                 
            return final_embedding, casual_mask, position_ids
        
        
        
        
        def forward(
            self,
            input_ids: torch.LongTensor = None,             # token di testo (prompt) + placeholder per le immagini
            pixel_values: torch.FloatTensor = None,         # immafgini processate (rescaled ecc...)
            attention_mask: Optional[torch.Tensor] = None,  # data dal tokenizer
            kv_cache: Optional[KVCache] = None,             
        ) -> Tuple:
            assert torch.all(attention_mask == 1), "No padding token allowed in the input"
            
            # Si trasformano i token di testo + image placeholder in embedding
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            # da i contextualized embedding delle patch delle immagini
            # [batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim]
            selected_image_features = self.vision_model(pixel_values.to(inputs_embeds.dtype))
            # resize embedding delle immagini nella dimensione dei embedding dei token
            # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, hidden_size (text_embed_dim)]
            image_features = self.multi_modal_projector(selected_image_features)
            
            # merge image embeddings (sostituendoli ai placeolder) e text tokens
            inputs_embeds, attention_mask, position_ids = self._merge_inputs_ids_and_image_features(
                image_features, 
                inputs_embeds, 
                input_ids, 
                attention_mask, 
                kv_cache,
                )
            
            outputs = self.language_model(
                attention_mask = attention_mask,
                position_ids = position_ids,
                inputs_embeds = inputs_embeds,
                kv_cache = kv_cache,
            )
            
            return outputs 
            
