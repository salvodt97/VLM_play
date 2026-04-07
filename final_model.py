import torch
from torch import nn
from typing import Optional, Tuple


from ViT.siglip import SiglipVisionConfig, SiglipVisionModel
from LLM.gemma import GemmaConfig, GemmaLLM
from LLM.kvcache import KVCache




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
            casual_mask = torch.full((batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device)
        else:
            # qui genero tokens, quindi query (matrice Q) contiene una sola riga
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            casual_mask = torch.full((batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device)
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
            

     