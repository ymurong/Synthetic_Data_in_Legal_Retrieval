from transformers.adapters import BertAdapterModel

from .utils import extract_text_embed


class BertDense(BertAdapterModel):
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                head=None,
                output_adapter_gating_scores=False,
                output_adapter_fusion_attentions=False,
                **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=True,
        )
        pooling = getattr(self.config, "pooling")
        similarity_metric = getattr(self.config, "similarity_metric")
        text_embeds = extract_text_embed(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
            similarity_metric=similarity_metric,
            pooling=pooling,
        )
        if return_dict:
            outputs.embedding = text_embeds
            return outputs
        else:
            return text_embeds
