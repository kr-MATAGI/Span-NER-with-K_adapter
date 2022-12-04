# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import logging
import math
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss, Dropout, Embedding, Softmax, MSELoss

logger = logging.getLogger(__name__)

#========================================================
def Linear(i_dim, o_dim, bias=True):
#========================================================
    m = nn.Linear(i_dim, o_dim, bias)
    nn.init.normal_(m.weight, std=0.02)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

#========================================================
def gelu(x):
#========================================================
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

#========================================================
def swish(x):
#========================================================
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

#========================================================
class SelfOutput(nn.Module):
#========================================================
    def __init__(self, config):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

#========================================================
class Attention(nn.Module):
#========================================================
    def __init__(self, config):
        super(Attention, self).__init__()
        self.self_attn = SelfAttention(config)
        self.output = SelfOutput(config)
        self.pruned_heads = set()

    def prune_linear_layer(self, layer, index, dim=0):
        """ Prune a linear layer (a model parameters) to keep only entries in index.
            Return the pruned layer as a new layer with requires_grad=True.
            Used to remove heads.
        """
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self_attn.num_attention_heads, self.self_attn.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self_attn.query = self.prune_linear_layer(self.self_attn.query, index)
        self.self_attn.key = self.prune_linear_layer(self.self_attn.key, index)
        self.self_attn.value = self.prune_linear_layer(self.self_attn.value, index)
        self.output.dense = self.prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self_attn.num_attention_heads = self.self_attn.num_attention_heads - len(heads)
        self.self_attn.all_head_size = self.self_attn.attention_head_size * self.self_attn.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self_attn(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

#========================================================
class SelfAttention(nn.Module):
#========================================================
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.hidden_size / config.num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.o_proj = Linear(config.hidden_size, config.hidden_size)
        self.dropout = Dropout(config.dropout_prob)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # attention_scores : [64, 8, 25, 25]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.o_proj(context_layer)

        return attention_output

#========================================================
class PositionWiseFeedForward(nn.Module):
#========================================================
    def __init__(self, config):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.ff_dim)
        self.fc2 = Linear(config.ff_dim, config.hidden_size)
        self.act_fn = ACT2FN[config.act_fn]
        self.dropout = Dropout(config.dropout_prob)

    def forward(self, input):
        intermediate = self.fc1(input)
        ff_out = self.dropout(self.fc2(self.act_fn(intermediate)))
        return ff_out

#========================================================
class Block(nn.Module):
#========================================================
    def __init__(self, config):
        super(Block, self).__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.ffn = PositionWiseFeedForward(config)
        self.attn = SelfAttention(config)

    def forward(self, x, attention_mask):
        # Attention
        h = x
        x = self.attn(x, attention_mask)
        x = h + x
        x = self.attention_norm(x)

        # FFN
        h = x
        x = self.ffn(x)
        x = x + h
        x = self.ffn_norm(x)

        return x

#========================================================
class TransformerEncoder(nn.Module):
#========================================================
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.layer = nn.ModuleList()
        for l in range(config.num_hidden_layers):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers

#========================================================
class TransformerConfig(object):
#========================================================
    def __init__(self,
                 vocab_size_or_config_json_file,
                 act_fn="gelu",
                 hidden_size=768,
                 num_hidden_layers=12,
                 ff_dim=3072,
                 num_heads=12,
                 dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02
                 ):
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.act_fn = act_fn
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.ff_dim = ff_dim
            self.num_heads = num_heads
            self.dropout_prob = dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        config = TransformerConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

#========================================================
class Embeddings(nn.Module):
#========================================================
    """
        Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = Dropout(config.dropout_prob)

        self.init_weights(config)

    def init_weights(self, config):
        self.word_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.position_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.token_type_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings