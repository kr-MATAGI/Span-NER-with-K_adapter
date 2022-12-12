import os
import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.transformer_enc import TransformerEncoder, TransformerConfig
from definition.klue_dp import pos_labels

from typing import Optional

#========================================================
class DPResult:
#========================================================
    """Result object for DataParallel"""

    def __init__(self, heads: torch.Tensor, types: torch.Tensor) -> None:
        self.heads = heads
        self.types = types

#========================================================
class AdapterConfig(object):
#========================================================
    def __init__(self,
                 args,
                 pretrained_model_config):
        # from args
        self.adapter_size = args.adapter_size
        self.num_hidden_layers = args.adapter_transformer_layers
        self.adapter_skip_layers = args.adapter_skip_layers
        self.adapter_num = 3  # [0, 5, 11]

        self.project_hidden_size: int = pretrained_model_config.hidden_size
        self.hidden_act: str = "gelu"
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
        self.num_labels: int = 2
        self.output_attentions: bool = False
        self.output_hidden_states: bool = False
        self.torchscript: bool = False
        self.type_vocab_size: int = 1
        self.vocab_size: int = pretrained_model_config.vocab_size


# ========================================================
class Adapter(nn.Module):
    # ========================================================
    def __init__(self, adapter_config):
        super(Adapter, self).__init__()
        self.adapter_config = adapter_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.down_project = nn.Linear(
            self.adapter_config.project_hidden_size,
            self.adapter_config.adapter_size
        )

        '''
            vocab_size: 35004,
            project_hidden_size: 768,
            adapter_size: 128
        '''
        self.encoder_config = TransformerConfig(adapter_config.vocab_size, hidden_size=self.adapter_config.adapter_size)
        self.encoder_config.num_heads = 8
        self.encoder = TransformerEncoder(self.encoder_config)
        self.up_project = nn.Linear(self.adapter_config.adapter_size, adapter_config.project_hidden_size)

        self.init_weights()

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)

        input_shape = down_projected.size()[:-1]
        attention_mask = torch.ones(input_shape, device=self.device)
        # encoder_attention_mask = torch.ones(input_shape, device=self.args.device)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # head_mask = [None] * self.adapter_config.num_hidden_layers
        encoder_outputs = self.encoder(down_projected, attention_mask=extended_attention_mask)

        up_projected = self.up_project(encoder_outputs[0])
        return hidden_states + up_projected

    def init_weights(self):
        self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.up_project.bias.data.zero_()


#========================================================
class DpAdapterModel(nn.Module):
#========================================================
    def __init__(self, args, pretrained_model_config, num_pos_labels, num_dp_labels):
        super(DpAdapterModel, self).__init__()

        self.num_pos_labels = num_pos_labels
        self.num_dp_labels = num_dp_labels

        self.config = pretrained_model_config
        self.adapter_config = AdapterConfig(args, pretrained_model_config)

        self.adapter_skip_layers = args.adapter_skip_layers
        self.adapter_list = [0, 5, 11]  # Base Model
        self.adapter_num = len(self.adapter_list)

        self.adapter = nn.ModuleList([Adapter(self.adapter_config) for _ in range(self.adapter_num)])
        self.com_dense = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)

        # DP
        self.pos_dim = 100
        if args.no_pos:
            self.pos_embedding = None
        else:
            self.pos_embedding = nn.Embedding(self.num_pos_labels + 1, self.pos_dim)

        enc_dim = self.config.hidden_size * 2
        if self.pos_embedding is not None:
            enc_dim += self.pos_dim

        self.enc_layers = 1
        self.encoder = nn.LSTM(
            enc_dim,
            self.config.hidden_size,
            self.enc_layers,
            batch_first=True,
            dropout=0.33,
            bidirectional=True
        )
        self.dec_layers = 1
        self.decoder = nn.LSTM(
            self.config.hidden_size, self.config.hidden_size, self.dec_layers, batch_first=True, dropout=0.33
        )

        self.dropout = nn.Dropout2d(p=0.33)

        self.src_dense = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.hx_dense = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)

        self.arc_space = 512
        self.type_space = 256
        self.arc_c = nn.Linear(self.config.hidden_size * 2, self.arc_space)
        self.type_c = nn.Linear(self.config.hidden_size * 2, self.type_space)
        self.arc_h = nn.Linear(self.config.hidden_size, self.arc_space)
        self.type_h = nn.Linear(self.config.hidden_size, self.type_space)

        self.attention = BiAttention(self.arc_space, self.arc_space, 1)
        self.bilinear = BiLinear(self.type_space, self.type_space, self.num_dp_labels)

    def forward(self, pretrained_model_outputs, input_ids,
                max_word_length: int, batch_index: torch.Tensor,
                bpe_head_mask=None, bpe_tail_mask=None,
                attention_mask=None, pos_ids=None, head_ids=None, type_ids=None,
                mask_e=None, mask_d=None, is_training: bool = True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        outputs = pretrained_model_outputs

        sequence_output = outputs.last_hidden_state
        # pooler_output = outputs[1]
        hidden_states = outputs.hidden_states
        hidden_state_num = len(hidden_states)
        hidden_states_last = torch.zeros(sequence_output.size()).to(device)

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
        # [batch, seq_len, hidden_size] -> [64, 128, 768]
        com_features = self.com_dense(torch.cat([sequence_output, hidden_states_last], dim=2))

        # DP Pre-train
        com_features, sent_len = self.resize_outputs(com_features, bpe_head_mask, bpe_tail_mask, max_word_length)

        if self.pos_embedding is not None:
            pos_outputs = self.pos_embedding(pos_ids)
            pos_outputs = self.dropout(pos_outputs)
            com_features = torch.cat([com_features, pos_outputs], dim=2)

        # encoder
        packed_outputs = pack_padded_sequence(com_features, sent_len, batch_first=True, enforce_sorted=False)
        encoder_outputs, hn = self.encoder(packed_outputs)
        encoder_outputs, outputs_len = pad_packed_sequence(encoder_outputs, batch_first=True)
        encoder_outputs = self.dropout(encoder_outputs.transpose(1, 2)).transpose(1, 2)  # apply dropout for last layer
        hn = self._transform_decoder_init_state(hn)

        # decoder
        src_encoding = F.elu(self.src_dense(encoder_outputs[:, 1:]))
        sent_len = [i - 1 for i in sent_len]
        packed_outputs = pack_padded_sequence(src_encoding, sent_len, batch_first=True, enforce_sorted=False)
        decoder_outputs, _ = self.decoder(packed_outputs, hn)
        decoder_outputs, outputs_len = pad_packed_sequence(decoder_outputs, batch_first=True)
        decoder_outputs = self.dropout(decoder_outputs.transpose(1, 2)).transpose(1, 2)  # apply dropout for last layer

        # compute output for arc and type
        arc_c = F.elu(self.arc_c(encoder_outputs))
        type_c = F.elu(self.type_c(encoder_outputs))

        arc_h = F.elu(self.arc_h(decoder_outputs))
        type_h = F.elu(self.type_h(decoder_outputs))

        out_arc = self.attention(arc_h, arc_c, mask_d=mask_d, mask_e=mask_e).squeeze(dim=1)

        # use predicted head_ids when validation step
        if not is_training:
            head_ids = torch.argmax(out_arc, dim=2)
        type_c = type_c[batch_index, head_ids.data.t()].transpose(0, 1).contiguous()
        out_type = self.bilinear(type_h, type_c)

        if is_training:
            loss = self._compute_loss(out_arc, out_type, max_word_length, mask_e, mask_d,
                                      batch_index, head_ids, type_ids)
            return loss
        else:
            heads = torch.argmax(out_arc, dim=2)
            types = torch.argmax(out_type, dim=2)

            preds = DPResult(heads, types)
            labels = DPResult(head_ids, type_ids)

            return {"preds": preds, "labels": labels}

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory), "Saving path should be a directory where the model and configuration can be saved"
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self
        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        print("Saving model checkpoint to %s", save_directory)

    def _compute_loss(self, out_arc, out_type, max_word_length, mask_e, mask_d,
                      batch_index, head_ids, type_ids):
        batch_size = head_ids.size()[0]
        head_index = (
            torch.arange(0, max_word_length).view(max_word_length, 1).expand(max_word_length, batch_size).long()
        )

        minus_inf = -1e8

        minus_mask_d = (1 - mask_d) * minus_inf
        minus_mask_e = (1 - mask_e) * minus_inf
        out_arc = out_arc + minus_mask_d.unsqueeze(2) + minus_mask_e.unsqueeze(1)

        loss_arc = F.log_softmax(out_arc, dim=2)
        loss_type = F.log_softmax(out_type, dim=2)

        loss_arc = loss_arc * mask_d.unsqueeze(2) * mask_e.unsqueeze(1)
        loss_type = loss_type * mask_d.unsqueeze(2)
        num = mask_d.sum()

        loss_arc = loss_arc[batch_index, head_index, head_ids.data.t()].transpose(0, 1)
        loss_type = loss_type[batch_index, head_index, type_ids.data.t()].transpose(0, 1)
        loss_arc = -loss_arc.sum() / num
        loss_type = -loss_type.sum() / num
        loss = loss_arc + loss_type

        # print("train/loss_arc", loss_arc)
        # print("train/loss_type", loss_type)
        # print("train/loss", loss)

        return loss

    def resize_outputs(
            self, outputs: torch.Tensor,
            bpe_head_mask: torch.Tensor, bpe_tail_mask: torch.Tensor, max_word_length: int):
        """Resize output of pre-trained transformers (bsz, max_token_length, hidden_dim) to word-level outputs (bsz, max_word_length, hidden_dim*2). """
        batch_size, input_size, hidden_size = outputs.size()
        word_outputs = torch.zeros(batch_size, max_word_length + 1, hidden_size * 2).to(outputs.device)
        sent_len = list()

        for batch_id in range(batch_size):
            head_ids = [i for i, token in enumerate(bpe_head_mask[batch_id]) if token == 1]
            tail_ids = [i for i, token in enumerate(bpe_tail_mask[batch_id]) if token == 1]
            assert len(head_ids) == len(tail_ids)

            word_outputs[batch_id][0] = torch.cat(
                (outputs[batch_id][0], outputs[batch_id][0])
            )  # replace root with [CLS]
            for i, (head, tail) in enumerate(zip(head_ids, tail_ids)):
                word_outputs[batch_id][i + 1] = torch.cat((outputs[batch_id][head], outputs[batch_id][tail]))
            sent_len.append(i + 2)

        return word_outputs, sent_len

    def _transform_decoder_init_state(self, hn: torch.Tensor) -> torch.Tensor:
        hn, cn = hn
        cn = cn[-2:]  # take the last layer
        _, batch_size, hidden_size = cn.size()
        cn = cn.transpose(0, 1).contiguous()
        cn = cn.view(batch_size, 1, 2 * hidden_size).transpose(0, 1)
        cn = self.hx_dense(cn)
        if self.decoder.num_layers > 1:
            cn = torch.cat(
                [
                    cn,
                    torch.autograd.Variable(cn.data.new(self.decoder.num_layers - 1, batch_size, hidden_size).zero_()),
                ],
                dim=0,
            )
        hn = torch.tanh(cn)
        hn = (hn, cn)
        return hn

#========================================================
class BiAttention(nn.Module):
#========================================================
    def __init__(  # type: ignore[no-untyped-def]
        self, input_size_encoder: int, input_size_decoder: int, num_labels: int, biaffine: bool = True, **kwargs
    ) -> None:
        super(BiAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.num_labels = num_labels
        self.biaffine = biaffine

        self.W_e = nn.Parameter(torch.Tensor(self.num_labels, self.input_size_encoder))
        self.W_d = nn.Parameter(torch.Tensor(self.num_labels, self.input_size_decoder))
        self.b = nn.Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = nn.Parameter(torch.Tensor(self.num_labels, self.input_size_decoder, self.input_size_encoder))
        else:
            self.register_parameter("U", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_e)
        nn.init.xavier_uniform_(self.W_d)
        nn.init.constant_(self.b, 0.0)
        if self.biaffine:
            nn.init.xavier_uniform_(self.U)

    def forward(
        self,
        input_d: torch.Tensor,
        input_e: torch.Tensor,
        mask_d: Optional[torch.Tensor] = None,
        mask_e: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert input_d.size(0) == input_e.size(0)
        batch, length_decoder, _ = input_d.size()
        _, length_encoder, _ = input_e.size()

        out_d = torch.matmul(self.W_d, input_d.transpose(1, 2)).unsqueeze(3)
        out_e = torch.matmul(self.W_e, input_e.transpose(1, 2)).unsqueeze(2)

        if self.biaffine:
            output = torch.matmul(input_d.unsqueeze(1), self.U)
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))
            output = output + out_d + out_e + self.b
        else:
            output = out_d + out_d + self.b

        if mask_d is not None:
            output = output * mask_d.unsqueeze(1).unsqueeze(3) * mask_e.unsqueeze(1).unsqueeze(2)

        return output

#========================================================
class BiLinear(nn.Module):
#========================================================
    def __init__(self, left_features: int, right_features: int, out_features: int):
        super(BiLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = nn.Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.W_l = nn.Parameter(torch.Tensor(self.out_features, self.left_features))
        self.W_r = nn.Parameter(torch.Tensor(self.out_features, self.left_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.constant_(self.bias, 0.0)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left: torch.Tensor, input_right: torch.Tensor) -> torch.Tensor:
        left_size = input_left.size()
        right_size = input_right.size()
        assert left_size[:-1] == right_size[:-1], "batch size of left and right inputs mis-match: (%s, %s)" % (
            left_size[:-1],
            right_size[:-1],
        )
        batch = int(np.prod(left_size[:-1]))

        input_left = input_left.contiguous().view(batch, self.left_features)
        input_right = input_right.contiguous().view(batch, self.right_features)

        output = F.bilinear(input_left, input_right, self.U, self.bias)
        output = output + F.linear(input_left, self.W_l, None) + F.linear(input_right, self.W_r, None)
        return output.view(left_size[:-1] + (self.out_features,))