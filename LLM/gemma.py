import torch
from torch import nn
from typing import Optional, Tuple, List, final
import math

from LLM.kvcache import KVCache


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
        rope_theta = 10000.0,          # base frequency per la rotatory positional encoding
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
    
    
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # si ripete quello che viene dopo le prima dimensioni un numero di volte pari a n_rep
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # si riduce la dimensione aggiuntiva, moltiplicando
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



class GemmaRotatoryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device = None):
        super().__init__()
        self.dim = dim   # head dimension (si modifica la head attention, ohni head ha la sua positional encoding)
        self.max_position_embeddings = max_position_embeddings # numero massimo di posizioni da codificare
        self.base = base # per il calcolo dell'angolo theta
        # calcolo proprio l'angolo, preso dal paper originale
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype = torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)
    
    
    @torch.no_grad
    def forward(self, x, position_ids, seq_len=None):
        # funzione per il calcolo degli embedding col positional encoding
        # x: [batch_size, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        # inv_freq_expanded: [Batch_Size, Head_Dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, -1)
        # position_ids_expanded: [Batch_Size, 1, Seq_Len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # si disabilita la mixed precision
            # freqs: [Batch_Size, Head_Dim // 2, 1] @ [Batch_Size, 1, Seq_Len] --> [Batch_Size, Seq_Len, Head_Dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # facendo così in realtà non uso angoli diversi per ogni embedding, bensì ripeto degli angoli
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
    
    
def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim) # Add the head dimension
    # Apply the formula of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
            
     

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
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # si aggiunge qualche info per query e key che codifica la loro posizione
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len = None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)
        
        # metodo che ripete le head mancanti di k e v per avere lo stesso numero delle Q.
        # è come se non facessi la Grouped Query Attention (GQA), ma serve ad evitare l'implementazione di un CUDA Core custom
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        attn_weghts = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        assert attention_mask is not None
        attn_weghts = attn_weghts + attention_mask
        attn_weghts = nn.functional.softmax(attn_weghts, dim = -1, dtype=torch.float32).to(query_states.dtype)
        attn_weghts = nn.functional.dropout(attn_weghts, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weghts, value_states)
        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"attn_output should be of zise { (bsz, self.num_heads, q_len, self.head_dim)}, not {attn_output.size()}"
            )
        
        # transpongo all'indietro come nel caso di siglip
        attn_output = attn_output.transpose(1,2).contiguous()
        # concateno le head
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weghts


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
        hidden_states, _ = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            position_ids = position_ids,
            kv_cache = kv_cache,
        )
        hidden_states = residual + hidden_states
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
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype = hidden_states.dtype)
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
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight


    def forward(
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
            
