from transformers.adapters import RobertaAdapterModel

from .utils import extract_text_embed


class RobertaDense(RobertaAdapterModel):
    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None, return_dict=False):
        pooling = getattr(self.config, "pooling")
        outputs = self.roberta(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids, 
            position_ids = position_ids,
            return_dict = True,
        )
        similarity_metric = getattr(self.config, "similarity_metric")
        text_embeds = extract_text_embed(
            last_hidden_state = outputs.last_hidden_state, 
            attention_mask = attention_mask,
            similarity_metric = similarity_metric, 
            pooling = pooling,
        )
        if return_dict:
            outputs.embedding = text_embeds
            return outputs
        else:
            return text_embeds