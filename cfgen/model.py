# coding=utf-8
# Copyright (C) The Zhejiang University DMAC Lab Authors. team - All Rights Reserved
#
# Written by Xiangji Zeng <xijiz@qq.com>, March 2020
"""Model Zoo"""

__all__ = ["LSTMTagger", "BERTTagger"]

import math
from collections import OrderedDict
import torch
import torch.nn as nn
from torch import FloatTensor


def gelu(inputs: torch.FloatTensor) -> torch.FloatTensor:
    """
    Gaussian Error Linear Units (GELUs)

    References:
        [1] https://arxiv.org/abs/1606.08415

    Args:
        inputs (torch.FloatTensor): Inputs.
    """
    return inputs * 0.5 * (1.0 + torch.erf(inputs / math.sqrt(2.0)))


class BERTEmbeddings(nn.Module):
    """
    BERTEmbedding: token_embedding + position_embedding + token_type_embedding.

    Args:
        n_tokens (int): The total number of tokens.
        hidden_dim (int): The dimension of embedding.
        max_seq_len (int): The maximum length of the input sequence.
        n_token_types (int): The number of token types.
        empty_id (int): The ID of the special token [EMPTY], we will conduct a DO operation on empty id.
        padding_id (int): PAD ID.
        hidden_dropout (float): Dropout probability for hidden states.
        layer_norm_eps (float): Layer normalization eps.
    """

    def __init__(
            self,
            n_tokens: int,
            hidden_dim: int,
            max_seq_len: int,
            n_token_types: int,
            empty_id: int,
            padding_id: int,
            hidden_dropout: float,
            layer_norm_eps: float
    ):
        super(BERTEmbeddings, self).__init__()
        self.empty_id = empty_id
        self.token_embeddings = nn.Embedding(n_tokens, hidden_dim, padding_id)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_dim)
        self.token_type_embeddings = nn.Embedding(n_token_types, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        """
        Return BERT embeddings with do operation.

        Args:
            input_ids (torch.LongTensor): Batch sentence inputs with shape (batch_size, seq_len).
            token_type_ids (torch.LongTensor): Batch sentence token types with shape (batch_size, seq_len).
            position_ids (torch.LongTensor): Batch sentence token positions with shape (batch_size, seq_len).
        """
        seq_len = input_ids.shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)

        token_embeds = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = token_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings[input_ids == self.empty_id] = 0.0

        return embeddings


class BERTSelfAttention(nn.Module):
    """
    BERTSelfAttention apllies multi-head attention to get a context-aware token representations.

    Args:
        hidden_dim (int): The dimension of hidden states.
        n_heads (int): The number of attention heads.
        attention_dropout (float): Attention dropout probability.
    """
    def __init__(self, hidden_dim: int, n_heads: int, attention_dropout: float):
        super(BERTSelfAttention, self).__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, hidden_states, mask=None):
        """
        Return a context-aware token representations.

        Args:
            hidden_states (torch.FloatTensor): Batch hidden states with shape (batch_size, seq_len, hidden_dim).
            mask (torch.FloatTensor): Batch masks with shape (batch_size, seq_len. hidden_dim).
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        attention_shape = (batch_size, seq_len, self.n_heads, self.head_dim)
        query = self.query(hidden_states).view(attention_shape).permute(0, 2, 1, 3)
        key = self.key(hidden_states).view(attention_shape).permute(0, 2, 1, 3)
        value = self.value(hidden_states).view(attention_shape).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        if mask is not None:
            attention_scores = attention_scores + mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(attention_probs, value).permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, hidden_dim)

        outputs = (context, attention_probs)

        return outputs


class BERTSelfOutput(nn.Module):
    """
    BERTSelfOutput processes the attention outputs.

    Args:
        hidden_dim (int): The dimension of hidden states.
        layer_norm_eps (float): Layer normalization epsilon.
        hidden_dropout (float): Hidden dropout probability.
    """

    def __init__(self, hidden_dim: int, layer_norm_eps: float, hidden_dropout: float):
        super(BERTSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states


class BERTAttention(nn.Module):
    """
    BERTAttention applies attention mechanism to get context-aware representations.

    Args:
        hidden_dim (int): The dimension of hidden states.
        n_heads (int): The number of attention heads.
        attention_dropout (float): Attention dropout probability.
        layer_norm_eps (float): Layer normalization epsilon.
        hidden_dropout (float): Hidden dropout probability.
    """

    def __init__(self, hidden_dim: int, n_heads: int, attention_dropout: float, layer_norm_eps: float, hidden_dropout: float):
        super(BERTAttention, self).__init__()
        self.self = BERTSelfAttention(hidden_dim, n_heads, attention_dropout)
        self.output = BERTSelfOutput(hidden_dim, layer_norm_eps, hidden_dropout)

    def forward(self, hidden_states, mask=None):
        """
        Return a context-aware token representations.

        Args:
            hidden_states (torch.FloatTensor): Batch hidden states with shape (batch_size, seq_len, hidden_dim).
            mask (torch.FloatTensor): Batch masks with shape (batch_size, seq_len. hidden_dim).
        """
        self_outputs = self.self(hidden_states, mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]

        return outputs


class BERTIntermediate(nn.Module):
    """
    BERTIntermediate transforms the outputs of attention module.

    Args:
        hidden_dim (int): The dimension of hidden states.
        intermediate_size (int): The size of feedforward intermediate.
    """

    def __init__(self, hidden_dim: int, intermediate_size: int):
        super(BERTIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_dim, intermediate_size)

    def forward(self, hidden_states):
        """
        Return the transformed representations.

        Args:
            hidden_states (torch.FloatTensor): Batch hidden states with shape (batch_size, seq_len, hidden_dim).
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)

        return hidden_states


class BERTOutput(nn.Module):
    """
    BERTOutput processes the feedforward outputs.

    Args:
        hidden_dim (int): The dimension of hidden states.
        intermediate_size (int): The size of feedforward intermediate.
        layer_norm_eps (float): Layer normalization epsilon.
        hidden_dropout (float): Hidden dropout probability.
    """

    def __init__(self, hidden_dim: int, intermediate_size: int, layer_norm_eps: float, hidden_dropout: float):
        super(BERTOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states, input_tensor):
        """
        Return a context-aware token representations.

        Args:
            hidden_states (torch.FloatTensor): Batch hidden states with shape (batch_size, seq_len, hidden_dim).
            input_tensor (torch.FloatTensor): Batch input tensor with shape (batch_size, seq_len. hidden_dim).
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states


class BERTLayer(nn.Module):
    """
    BERTLayer applies self-attention to token embeddings of a given sentence.

    Args:
        hidden_dim (int): The dimension of hidden states.
        n_heads (int): The number of attention heads.
        intermediate_size (int): The size of feedforward intermediate.
        attention_dropout (float): Attention dropout probability.
        layer_norm_eps (float): Layer normalization epsilon.
        hidden_dropout (float): Hidden dropout probability.
    """

    def __init__(
            self,
            hidden_dim: int,
            n_heads: int,
            attention_dropout: float,
            layer_norm_eps: float,
            hidden_dropout: float,
            intermediate_size: int
    ):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(hidden_dim, n_heads, attention_dropout, layer_norm_eps, hidden_dropout)
        self.intermediate = BERTIntermediate(hidden_dim, intermediate_size)
        self.output = BERTOutput(hidden_dim, intermediate_size, layer_norm_eps, hidden_dropout)

    def forward(self, hidden_states, mask=None):
        """
        Return a context-aware token representations.

        Args:
            hidden_states (torch.FloatTensor): Batch hidden states with shape (batch_size, seq_len, hidden_dim).
            mask (torch.FloatTensor): Batch masks with shape (batch_size, seq_len. hidden_dim).
        """
        self_attention_outputs = self.attention(hidden_states, mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        return outputs


class BERTEncoder(nn.Module):
    """
    BERTEncoder integrates multiple transformer layers into a module for calculating the context-aware
    representations.

    Args:
        hidden_dim (int): The dimension of hidden states.
        n_heads (int): The number of attention heads.
        intermediate_size (int): The size of feedforward intermediate.
        n_layers (int): The number of transformer layers.
        attention_dropout (float): Attention dropout probability.
        layer_norm_eps (float): Layer normalization epsilon.
        hidden_dropout (float): Hidden dropout probability.
    """

    def __init__(
            self,
            hidden_dim: int,
            n_heads: int,
            attention_dropout: float,
            layer_norm_eps: float,
            hidden_dropout: float,
            intermediate_size: int,
            n_layers: int
    ):
        super(BERTEncoder, self).__init__()
        self.layer = nn.ModuleList(
            [
                BERTLayer(hidden_dim, n_heads, attention_dropout, layer_norm_eps, hidden_dropout, intermediate_size)
                for _ in range(n_layers)
            ]
        )

    def forward(self, hidden_states, mask=None):
        """
        Return a context-aware token representations.

        Args:
            hidden_states (torch.FloatTensor): Batch hidden states with shape (batch_size, seq_len, hidden_dim).
            mask (torch.FloatTensor): Batch masks with shape (batch_size, seq_len. hidden_dim).
        """
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states, mask)
            hidden_states = layer_outputs[0]
            all_attentions = all_attentions + (layer_outputs[1],)
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class BERTPooler(nn.Module):
    """
    BERTPooler pools the first token output of BERT outputs.

    Args:
        hidden_dim (int): The dimension of hidden states.
    """

    def __init__(self, hidden_dim: int):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        """
        Return the pooled outputs.

        Args:
            hidden_states (torch.FloatTensor): Batch hidden states with shape (batch_size, seq_len, hidden_dim).
        """
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BERTModel(nn.Module):
    """
    The implementation of the paper "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding".

    Args:
        n_tokens (int): The total number of tokens.
        hidden_dim (int): The dimension of hidden states.
        max_seq_len (int): The maximum length of sentences in a batch.
        n_token_types (int): The number of total token types.
        empty_id (int): The ID of the special token [EMPTY], we will conduct a DO operation on empty id..
        padding_id (int): The ID of the special token [PAD].
        n_heads (int): The number of attention heads.
        intermediate_size (int): The size of feedforward intermediate.
        n_layers (int): The number of transformer layers.
        attention_dropout (float): Attention dropout probability.
        layer_norm_eps (float): Layer normalization epsilon.
        hidden_dropout (float): Hidden dropout probability.
    """

    def __init__(
            self,
            n_tokens: int,
            hidden_dim: int,
            max_seq_len: int,
            n_token_types: int,
            empty_id: int,
            padding_id: int,
            n_heads: int,
            intermediate_size: int,
            n_layers: int,
            attention_dropout: float,
            layer_norm_eps: float,
            hidden_dropout: float
    ):
        super(BERTModel, self).__init__()
        self.embeddings = BERTEmbeddings(
            n_tokens, hidden_dim, max_seq_len, n_token_types, empty_id, padding_id, hidden_dropout, layer_norm_eps
        )
        self.encoder = BERTEncoder(
            hidden_dim, n_heads, attention_dropout, layer_norm_eps, hidden_dropout, intermediate_size, n_layers
        )
        self.pooler = BERTPooler(hidden_dim)

    def forward(self, input_ids, mask=None, token_type_ids=None, position_ids=None):
        """
        Return BERT representations.

        Args:
            input_ids (torch.LongTensor): Inputs with shape (batch_size, seq_len).
            mask (torch.LongTensor): Mask with shape (batch_size, seq_len).
            token_type_ids (torch.LongTensor): Token types with shape (batch_size, seq_len).
            position_ids (torch.LongTensor): Token positions with shape (batch_size, seq_len).
        """
        if mask is None:
            mask = torch.ones(input_ids.size(), device=input_ids.device)
        if mask.dim() == 3:
            mask = mask[:, None, :, :]
        elif mask.dim() == 2:
            mask = mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_ids.size(), mask.shape
                )
            )
        # (batch_size, n_heads, seq_len, head_dim)
        mask = (1.0 - mask) * -10000.0

        embedding_output = self.embeddings(input_ids, position_ids, token_type_ids)
        encoder_outputs = self.encoder(embedding_output, mask)
        pooled_output = self.pooler(encoder_outputs[0])
        outputs = (pooled_output,) + encoder_outputs

        return outputs


class BERTTagger(nn.Module):
    """
    A BERT-based Tagger for the task sequence labeling.

    Args:
        n_tags (int): The number of tags.
        token_embeddings (OrderedDict): The pretrained BERTModel weights.
    """
    def __init__(
            self,
            n_tokens: int,
            hidden_dim: int,
            max_seq_len: int,
            n_token_types: int,
            empty_id: int,
            padding_id: int,
            n_heads: int,
            intermediate_size: int,
            n_layers: int,
            attention_dropout: float,
            layer_norm_eps: float,
            hidden_dropout: float,
            n_tags: int,
            token_embeddings: OrderedDict = None
    ):
        super(BERTTagger, self).__init__()
        self.bert = BERTModel(
            n_tokens, hidden_dim, max_seq_len, n_token_types, empty_id, padding_id, n_heads,
            intermediate_size, n_layers, attention_dropout, layer_norm_eps, hidden_dropout
        )
        if token_embeddings is not None:
            self.bert.load_state_dict(token_embeddings)
        self.dropout = nn.Dropout(p=hidden_dropout)
        self.tagger = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.PReLU(), nn.Linear(hidden_dim*2, n_tags))

    def forward(self, inputs, mask, *args):
        """
        Compute tagging logits by BERTTagger.

        Args:
            inputs (torch.LongTensor): Batch sequence inputs with expected shape (batch_size, seq_len).
            mask (torch.LongTensor): Mask with shape (batch_size, seq_len).
        """
        bert_outputs = self.bert(inputs, mask)
        outputs = self.dropout(bert_outputs[1])
        logits = self.tagger(outputs)

        return logits


class LSTMTagger(nn.Module):
    """
    A bidirectional LSTMTagger for the task sequence labeling.

    Args:
        n_tokens (int): The number of total tokens.
        token_dim (int): The dimension of token embedding.
        hidden_dim (int): The dimension of hidden states.
        n_layers (int): The number of LSTM layers.
        n_tags (int): The number of tags.
        empty_id (int): The ID of the special token [EMPTY], we will conduct a DO operation on empty id.
        dropout (float): Dropout probability.
        token_embeddings (FloatTensor): Token embeddings.
    """

    def __init__(
            self,
            n_tokens: int,
            token_dim: int,
            hidden_dim: int,
            n_layers: int,
            n_tags: int,
            empty_id: int,
            dropout: float,
            token_embeddings: FloatTensor = None
    ):
        super(LSTMTagger, self).__init__()
        self.empty_id = empty_id
        if token_embeddings is None:
            self.embedding = nn.Embedding(n_tokens, token_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(token_embeddings)
        self.rnn = nn.LSTM(token_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=dropout)
        self.tagger = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim*2), nn.PReLU(), nn.Linear(hidden_dim*2, n_tags))

    def forward(self, inputs, *args):
        """
        Compute tagging logits by LSTMTagger.

        Args:
            inputs (torch.LongTensor): Batch sequence inputs with expected shape (batch_size, seq_len).
        """
        embeds = self.embedding(inputs)
        embeds[inputs == self.empty_id] = 0.0
        self.rnn.flatten_parameters()
        rnn_output, _ = self.rnn(embeds)
        outputs = self.dropout(rnn_output)
        logits = self.tagger(outputs)

        return logits
