import torch
import torch.nn as nn

from transformers import ElectraModel

#=======================================================
class PretrainedModel(nn.Module):
#=======================================================\
    def __init__(self, tokenizer):
        super(PretrainedModel, self).__init__()
        self.model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", output_hidden_states=True)
        self.model.resize_token_embeddings(len(tokenizer))
        self.config = self.model.config
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                label_ids=None, subj_start_id=None, obj_start_id=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        return outputs # (loss), logits, (hidden_states), (attentions)

