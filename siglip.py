from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    # configurazione per tutti i moduli della parte ViT della VLM
    def __init__(
        self,
        hidden_size=768,                # dimensione del vettore di embedding del ViT
        intermediate_size=3072,         # dimensione del layer lineare della feed-forward network all'interno di ogni blocco del ViT
        num_hidden_layers=12,           # numero di layer del ViT
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout_prob=0.0,
        num_image_tokens : int = None,  # numero di embeddings per immagine
        **kwargs    
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout_prob = attention_dropout_prob
        self.num_image_tokens = num_image_tokens
        


class SiglipVisionEmbeddings(nn.Module):
    # prende l'immagine, la diviode in patch, la flattenizza e ci aggiunge i positional encoding
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size     # patch di 16x16 pixel
        
        # estrazione di feature patch per patch senza overlpping
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"         # no padding
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2 # patch quadrata
        self.num_positions = self.num_patches   # sarebbe il numero di positional encoding, ovviamente uno per patch
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim)  # il position embencoding è un vettore appreso
        self.register_buffer(
            # si salvano gli id delle posizioni nel SiglipVisionEmbeddings
            "position_ids", 
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
            )  
    
    def forward(self, pixel_values) -> torch.Tensor:
        _, _, height, width = pixel_values.shape  # valori ritornati da nunpy
        # [batch_size, embed_dim, num_patches_h, num_patches_w] -> [batch_size, embed_dim, num_patches]
        patch_embeds = self.patch_embedding(pixel_values)  # converte l'immagine in embeddings patch per patch
        embeddings = patch_embeds.flatten(2) # flatten delle patch dopo la convoluzione per ottnere le patch
        embeddings = embeddings.transpose(1, 2) # [batch_size, embed_dim, num_patches], si traspone, perchè voglio un batch di sequenze di embeddings
        embeddings = embeddings + self.position_embeddings(self.position_ids)  # aggiungo il positional encoding alle patch
       
        return embeddings       
        
        

class SiglipVisionEncoder(nn.Module):
    # è la parte che calcola i contextualized embeddings. Sarebbe quindi l'encoder
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiflipAttention(config)        # modulo di self attention
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)                     # modulo multi-layer perceptron
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states     # parte per la skip connection
        hidden_states = self.layer_norm1(hidden_states)  # normalizzazione delle dimensioni batch_size, nmber_of_patches, embed_dim
        hidden_states, _ = self.self_attn(hidden_states)  # calcolo della self attention
        hidden_states = residual + hidden_states          # skip connection
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)  # normalizzazione num 2
        hidden_states = self.mlp(hidden_states)          # calcolo del multi-layer perceptron: layer lineari che trasforma ciascun embedding indipendentemente dagli altri
        hidden_states = residual + hidden_states          # skip connection num 2
        
        return hidden_states
        
        
class SiglipVisionTransformer(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size      
        self.embeddings = SiglipVisionEmbeddings(config)                # modulo che divide l'immagine in patch 
        self.encoder = SiglipVisionEncoder(config)                      # modulo che implementa i layer del ViT   
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # addestramento più stabile
        
    def forward(self, pixel_values) -> Tuple:
        # pixel_values: [batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim]
        hideen_state = self.embeddings(pixel_values)    # estrazione delle patch (con la convoluzione e flatten)
        last_hidden_state = self.encoder(hideen_state)
        # last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        return last_hidden_state
        
        
        
class SiglipVisionModel(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
        
    def forward(self, pixel_values) -> Tuple:
        # funzione che passa le immagini attraverso il ViT e restituisce i relativi embeddings
        # [batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim]
        return self.vision_model(pixel_values=pixel_values)