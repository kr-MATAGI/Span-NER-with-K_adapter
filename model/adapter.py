import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformer_enc import TransformerEncoder, TransformerConfig

#========================================================
class AdapterConfig(object):
#========================================================
    def __init__(self,
                 pretrained_model_config,
                 adapter_transformer_layers: int = 2,
                 adapter_size: int = 768,
                 vocab_size: int = 50265):

        self.project_hidden_size: int = pretrained_model_config.hidden_size
        self.hidden_act: str = "gelu"
        self.adapter_size: int = adapter_size  # 64
        self.adapter_initializer_range: float = 0.0002
        self.is_decoder: bool = False
        self.attention_probs_dropout_prob: float = 0.1
        self.hidden_dropout_prob: float = 0.1
        self.hidden_size: int = 768
        self.initializer_range: float = 0.02
        self.intermediate_size: int = 3072
        self.layer_norm_eps: float = 1e-05
        self.max_position_embeddings: int = 514
        self.num_attention_heads: int = 12
        self.num_hidden_layers: int = adapter_transformer_layers
        self.num_labels: int = 2
        self.output_attentions: bool = False
        self.output_hidden_states: bool = False
        self.torchscript: bool = False
        self.type_vocab_size: int = 1
        self.vocab_size: int = vocab_size

#========================================================
class Adapter(nn.Module):
#========================================================
    def __init__(self, adapter_config):
        super(Adapter, self).__init__()
        self.adapter_config = adapter_config

        self.down_project = nn.Linear(
            self.adapter_config.project_hidden_size,
            self.adapter_config.adapter_size
        )

        self.encoder = TransformerEncoder(self.adapter_config)
        self.up_project = nn.Linear(self.adapter_config.adapter_size, adapter_config.project_hidden_size)

        self.init_weights()

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)

        input_shape = down_projected.size()[:-1]
        attention_mask = torch.ones(input_shape, device=self.args.device)
        # encoder_attention_mask = torch.ones(input_shape, device=self.args.device)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.adapter_config.num_hidden_layers
        encoder_outputs = self.encoder(down_projected,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask)

        up_projected = self.up_project(encoder_outputs[0])
        return hidden_states + up_projected

    def init_weights(self):
        self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.up_project.bias.data.zero_()

#========================================================
class AdapterModel(nn.Module):
#========================================================
    def __init__(self, pretrained_model_config):
        super(AdapterModel, self).__init__()

        self.config = pretrained_model_config
        self.adapter_size = 768 # 768
        self.adapter_skip_layers = 2 # 2
        self.adapter_list = [0, 11, 23] # based Large
        self.adapter_num = len(self.adapter_list)
        self.adapter = nn.ModuleList([Adapter(args, AdapterConfig) for _ in range(self.adapter_num)])

        self.com_dense = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.dense = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.config.hidden_size, self.num_labels)

    def forward(self, pretrained_model_outputs, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None,
                labels=None, subj_special_start_id=None, obj_special_start_id=None):
        outputs = pretrained_model_outputs
        sequence_output = outputs[0]
        # pooler_output = outputs[1]
        hidden_states = outputs[2]
        hidden_state_num = len(hidden_states)
        hidden_states_last = torch.zeros(sequence_output.size()).to(self.args.device)

        adapter_hidden_states = []
        adapter_hidden_states_count = 0
        for i, adapter_module in enumerate(self.adapter):
            fusion_state = hidden_states[self.adapter_list[i]] + hidden_states_last
            hidden_states_last = adapter_module(fusion_state)
            adapter_hidden_states.append(hidden_states_last)
            adapter_hidden_states_count += 1
            if self.adapter_skip_layers >= 1:  # if adapter_skip_layers>=1, skip connection
                if adapter_hidden_states_count % self.adapter_skip_layers == 0:
                    hidden_states_last = hidden_states_last + adapter_hidden_states[
                        int(adapter_hidden_states_count / self.adapter_skip_layers)]

        ##### drop below parameters when doing downstream tasks
        com_features = self.com_dense(torch.cat([sequence_output, hidden_states_last], dim=2))

        subj_special_start_id = subj_special_start_id.unsqueeze(1)
        subj_output = torch.bmm(subj_special_start_id, com_features)
        obj_special_start_id = obj_special_start_id.unsqueeze(1)
        obj_output = torch.bmm(obj_special_start_id, com_features)
        logits = self.out_proj(
            self.dropout(self.dense(torch.cat((subj_output.squeeze(1), obj_output.squeeze(1)), dim=1))))

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)

    def save_pretrained(self, save_directory, logger):
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self
        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Saving model checkpoint to %s", save_directory)