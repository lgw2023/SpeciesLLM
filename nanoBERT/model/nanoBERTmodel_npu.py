"""
Full definition of a BERT Language Model, all of it in this single file.
References:
1) the official BERT TensorFlow implementation released by Google:
https://github.com/google-research/bert
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
"""

import math
from typing import Dict, Mapping, Optional, Tuple, Any, Union
import inspect
from dataclasses import dataclass
from packaging import version

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch_npu
from transformers.activations import ACT2FN
from transformers.utils import get_torch_version, ModelOutput
from transformers.modeling_outputs import MaskedLMOutput
from transformers.pytorch_utils import apply_chunking_to_forward

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class CellTokenEncoder(nn.Module):
    def __init__(
            self,
            num_embeddings: int,   ## ntoken
            embedding_dim: int,    ## n_embd
            dropout: float=0.0,
    ):
        super().__init__()
        #self.dropout = nn.Dropout(p = dropout)
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx = None
        )
        self.enc_norm = LayerNorm(embedding_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x
    
class GeneEncoder(nn.Module):
    def __init__(
            self,
            num_embeddings: int,   ## ntoken
            embedding_dim: int,    ## n_embd
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx = None
        )
        self.enc_norm = LayerNorm(embedding_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.1, len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout)
        position = torch.arange(len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd)
        )
        pe = torch.zeros(len, 1, n_embd)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]  ## x : Tensor, shape [seq_len, batch_size, embedding_dim]
        return self.dropout(x)
    
class ContinuousValueEncoder(nn.Module):
    """
    Encode real number nvalues to a vector using neural nets projection.
    Apated from scGPT
    """
    def __init__(self, n_embd: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, n_embd)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(n_embd, n_embd)
        self.norm = nn.LayerNorm(n_embd)
        self.max_value = max_value
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x:Tensor, shape is [batch_size, seq_len]
        """
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max = self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)

class BERTSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention"
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            output_attentions: Optional[bool] = False,    
        ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer, )
        return outputs
    
class BertSdpaSelfAttention(BERTSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.dropout_prob = config.attention_probs_dropout_prob
        self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")

    def forward(self, hidden_states, attention_mask=None, output_attentions: Optional[bool] = False) -> Tuple:
        if output_attentions:
            return super().forward(hidden_states, attention_mask, output_attentions)
        bsz, tgt_len, _ = hidden_states.size()
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        if self.require_contiguous_qkv and query_layer.device.type == "cuda" and attention_mask is not None:
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask = attention_mask,
            dropout_p = self.dropout_prob if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        outputs = (attn_output, )
        return outputs
    
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
BERT_SELF_ATTENTION_CLASSES = {
    "eager": BERTSelfAttention,
    "sdpa": BertSdpaSelfAttention,
}

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BERT_SELF_ATTENTION_CLASSES[config._attn_implementation](config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output, ) + self_outputs[1:]
        return outputs
    
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    
class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions= output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output, ) + outputs
        return outputs
    
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class ExprDecoder(nn.Module):
    def __init__(
            self,
            d_in: int,
            n_embd: int,
            explicit_zero_prob: bool = False,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_in, n_embd),
            nn.LeakyReLU(),
            nn.Linear(n_embd, n_embd),
            nn.LeakyReLU(),
            nn.Linear(n_embd, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, n_embd),
                nn.LeakyReLU(),
                nn.Linear(n_embd, n_embd),
                nn.LeakyReLU(),
                nn.Linear(n_embd, 1),
            )
        
    def forward(self, x : torch.Tensor) -> Dict[str, torch.Tensor]:
        """ x is the output of the transformer, shape is (batch, seq_len, n_embd)"""
        pred_value = self.fc(x).squeeze(-1) # shape is (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)
    
class MVCDecoder(nn.Module):
    def __init__(
            self,
            dim_in: int,
            n_embd: int,
            arch_style: str = "inner product",
            query_activation: nn.Module = nn.Sigmoid,
            hidden_activation: nn.Module = nn.PReLU,
            explicit_zero_prob: bool = False,
    ) -> None:
        super().__init__()
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(n_embd, n_embd)
            self.query_activation = query_activation()
            self.W = nn.Linear(n_embd, dim_in, bias=False)
            if explicit_zero_prob:
                self.W_zero_logit = nn.Linear(n_embd, dim_in)
        elif arch_style  == "concat query":
            self.gene2query = nn.Linear(n_embd, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(n_embd + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")
        
        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
            self, cell_emb: torch.Tensor, gene_emb: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
        cell_emb: Tensor, shape (batch, embsize = n_embd)
        gene_emb: Tensor, shape (batch, seq_len, embsize = n_embd)
        """
        gene_emb = gene_emb.detach() if self.do_detach else gene_emb
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_emb))
            cell_emb = cell_emb.unsqueeze(2) # (batch, embsize, 1)
            # the pred gene expr values, shape (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the cell_emb
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred = pred_value, zero_probs= zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_emb))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_emb.shape[1], -1)
            h = self.hidden_activation(self.fc1(torch.cat([cell_emb, query_vecs], dim=2)))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2) # (batch, seq_len)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_emb))
            cell_emb = cell_emb.unsqueeze(1)
            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2) # (batch, seq_len)
    
class ClsDecoder(nn.Module):
    """ Decoder for cell token classification task."""
    def __init__(
            self,
            n_embd: int,
            n_cls: int,
            n_layers: int = 3,
            activation: callable = nn.ReLU,
    ):
        super().__init__()
        self._decoder = nn.ModuleList()
        for i in range(n_layers - 1):
            self._decoder.append(nn.Linear(n_embd, n_embd))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(n_embd))
        self.out_layer = nn.Linear(n_embd, n_cls)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            x: Tensor, shape is [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

@dataclass
class BERTConfig:
    vocab_size: int
    vocab: Dict[str, int] = None
    hidden_size: int = 1280
    num_hidden_layers: int = 12
    num_attention_heads: int = 10
    intermediate_size: int = 5120
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    cell_hidden_size: int = 128
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 1196
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    use_batch_labels: bool = False
    num_batch_labels: Optional[int] = None
    use_species_labels: bool = False
    num_species_labels: Optional[int] = None
    use_tissue_labels: bool = False
    num_tissue_labels: Optional[int] = None
    use_seqmethod_labels: bool = False
    num_seqmethod_labels: Optional[int] = None
    use_disease_labels: bool = False
    num_disease_labels: Optional[int] = None
    use_sex_labels: bool = False
    num_sex_labels: Optional[int] = None
    use_age_labels: bool = False
    num_age_labels: Optional[int] = None
    cell_emb_style: str = "cls"
    do_mvc: bool = True
    do_cls: bool = False
    _attn_implementation: str = "eager"   # attention implementation, either "eager" or "sdpa"
    mvc_decoder_style: str = "inner product"
    num_cls: int = 1
    ## these parameters need further confirm
    chunk_size_feed_forward: int = None
    explicit_zero_prob: bool = False


class BERTModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.max_position_embeddings is not None
        self.config = config
        self.gradient_checkpointing = False

        self.gene_encoder = GeneEncoder(config.vocab_size, config.hidden_size)
        self.value_encoder = ContinuousValueEncoder(config.hidden_size, dropout=config.hidden_dropout_prob)
        #self.position_encoder = PositionalEncoding(config.hidden_size, dropout=config.hidden_dropout_prob, len=config.max_position_embeddings)
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        self.h = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        # init all weights
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
            self,
            src: torch.Tensor,
            values: torch.Tensor,
            embeddings: torch.Tensor,  #(batch, seq_len, emb_size)
            batch_labels: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
    ) -> torch.Tensor:
        device = src.device
        b, seq_len = src.size()
        assert seq_len <= self.config.max_position_embeddings, f"Cannot forward sequence of length {seq_len}, block size is only {self.config.max_position_embeddings}"

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        #self._check_batch_labels(batch_labels)

        # generate embeddings
        src = self.gene_encoder(src)
        self.cur_gene_token_embs = src

        values = self.value_encoder(values)  # (batch, seq_len, emb_size)
        x = src + values + embeddings

        x = self.drop(x)

        # input into BertAttention layers
        for block in self.h:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x, )
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    x,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = block(
                    x,
                    attention_mask,
                    output_attentions,
                )
            x = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
            
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)
        return x, all_hidden_states, all_self_attentions
    
    
    def _check_batch_labels(self, batch_labels: torch.Tensor) -> None:
        if self.use_cell_labels:
            assert batch_labels is not None
        elif batch_labels is not None:
            raise ValueError(
                f"labels should be provided when 'self.use_batch_labels' is True"
            )

class BERTForPreTraining(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.max_position_embeddings is not None
        self.config = config
        self.gradient_checkpointing = False

        self.bert = BERTModel(config)

        # These encoders act as cell token encoder
        if config.use_batch_labels:
            self.batch_encoder = CellTokenEncoder(config.num_batch_labels, config.cell_hidden_size)
        if config.use_species_labels:
            self.species_encoder = CellTokenEncoder(config.num_species_labels, config.cell_hidden_size)
        if config.use_tissue_labels:
            self.tissue_encoder = CellTokenEncoder(config.num_tissue_labels, config.cell_hidden_size)
        if config.use_seqmethod_labels:
            self.seqmethod_encoder = CellTokenEncoder(config.num_seqmethod_labels, config.cell_hidden_size)
        if config.use_disease_labels:
            self.disease_encoder = CellTokenEncoder(config.num_disease_labels, config.cell_hidden_size)
        # Add sex and age encoders
        if config.use_sex_labels:
            self.sex_encoder = CellTokenEncoder(config.num_sex_labels, config.cell_hidden_size)
        if config.use_age_labels:
            self.age_encoder = CellTokenEncoder(config.num_age_labels, config.cell_hidden_size)

        cell_attributes = [
            config.use_batch_labels,
            config.use_species_labels,
            config.use_tissue_labels,
            config.use_seqmethod_labels,
            config.use_disease_labels,
            config.use_sex_labels,
            config.use_age_labels,
        ]
        # Expression Decocer acts as BertOnlyMLMHead
        self.decoder = ExprDecoder(
            config.hidden_size + config.cell_hidden_size if any(cell_attributes) else config.hidden_size,
            config.hidden_size,
            config.explicit_zero_prob,
        )
        if config.do_mvc:
            self.mvc_decoder = MVCDecoder(
                config.hidden_size + config.cell_hidden_size if any(cell_attributes) else config.hidden_size,
                config.hidden_size,
                arch_style=config.mvc_decoder_style,
                explicit_zero_prob=config.explicit_zero_prob,
            )
        if config.do_cls:
            self.cls_decoder = ClsDecoder(
                config.hidden_size + config.cell_hidden_size if any(cell_attributes) else config.hidden_size,
                config.num_cls
            )

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        #self.gene_encoder.weight = self.decoder.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        
        return n_params
    #return x, all_hidden_states, all_self_attentions
    
    def _get_cell_emb_from_layer(
            self, layer_output: torch.Tensor, weights: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.config.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]
        elif self.config.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim = 1)
        elif self.config.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim = 1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1) # shape [batch, embsize]
        return cell_emb
    
    def _check_batch_labels(self, batch_labels: torch.Tensor) -> None:
        if self.use_cell_labels:
            assert batch_labels is not None
        elif batch_labels is not None:
            raise ValueError(
                f"labels should be provided when 'self.use_batch_labels' is True"
            )
        
    def get_output_embeddings(self):
        return self.decoder.fc[-1]
    
    def set_output_embeddings(self, new_embeddings):
        self.decoder.fc[-1] = new_embeddings

    def _encode(
            self,
            src: torch.Tensor,
            values: torch.Tensor,
            embeddings: torch.Tensor,
            batch_labels: Optional[torch.Tensor] = None,
            species_labels: Optional[torch.Tensor] = None,
            tissue_labels: Optional[torch.Tensor] = None,
            seqmethod_labels: Optional[torch.Tensor] = None,
            disease_labels: Optional[torch.Tensor] = None,
            sex_labels: Optional[torch.Tensor] = None,
            age_labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
    ) -> torch.Tensor:
        sequence_output, bert_hidden_states, bert_attentions = self.bert(
            src, values, embeddings, batch_labels, None, output_attentions, output_hidden_states
        )

        cell_token_embs = []
        cell_attributes = [
            (self.config.use_batch_labels, batch_labels, self.batch_encoder if hasattr(self, 'batch_encoder') else None),
            (self.config.use_species_labels, species_labels, self.species_encoder if hasattr(self, 'species_encoder') else None),
            (self.config.use_tissue_labels, tissue_labels, self.tissue_encoder if hasattr(self, 'tissue_encoder') else None),
            (self.config.use_seqmethod_labels, seqmethod_labels, self.seqmethod_encoder if hasattr(self, 'seqmethod_encoder') else None),
            (self.config.use_disease_labels, disease_labels, self.disease_encoder if hasattr(self, 'disease_encoder') else None),
            (self.config.use_sex_labels, sex_labels, self.sex_encoder if hasattr(self, 'sex_encoder') else None),
            (self.config.use_age_labels, age_labels, self.age_encoder if hasattr(self, 'age_encoder') else None),
        ]

        for use_label, labels, encoder in cell_attributes:
            if use_label and labels is not None:
                cell_token_embs.append(encoder(labels))

        if cell_token_embs:
            cell_token_embs = torch.stack(cell_token_embs, dim=0).sum(dim=0)
            x = torch.cat([sequence_output, cell_token_embs.unsqueeze(1).repeat(1, sequence_output.shape[1], 1)], dim=2)
        else:
            x = sequence_output

        cell_emb = self._get_cell_emb_from_layer(sequence_output, values)
        cell_emb = torch.cat([cell_emb, cell_token_embs], dim = 1) if isinstance(cell_token_embs, torch.Tensor) else cell_emb

        return x, cell_emb, bert_hidden_states, bert_attentions  # (batch, seq_len, embsize), (batch, embsize)

    def forward(
            self,
            src: torch.Tensor,
            values: torch.Tensor,
            embeddings: torch.Tensor,
            batch_labels: Optional[torch.Tensor] = None,
            species_labels: Optional[torch.Tensor] = None,
            tissue_labels: Optional[torch.Tensor] = None,
            seqmethod_labels: Optional[torch.Tensor] = None,
            disease_labels: Optional[torch.Tensor] = None,
            sex_labels: Optional[torch.Tensor] = None,
            age_labels: Optional[torch.Tensor] = None,
            CLS: bool = False,
            MVC: bool = False,
            targets: torch.Tensor = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
        ):
        x, cell_emb, bert_hidden_states, bert_attentions = self._encode(
            src=src,
            values= values, 
            embeddings=embeddings,
            batch_labels=batch_labels,
            species_labels=species_labels,
            tissue_labels=tissue_labels,
            seqmethod_labels=seqmethod_labels,
            disease_labels=disease_labels,
            sex_labels=sex_labels,
            age_labels=age_labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        outputs = {}
        model_output = self.decoder(x)

        outputs["model_output"] = model_output["pred"]
        if self.config.explicit_zero_prob:
            outputs["model_zero_prob"] = model_output["zero_probs"]

        if MVC:
            mvc_output = self.mvc_decoder(
                cell_emb,
                self.bert.cur_gene_token_embs,
            )
            outputs["mvc_output"] = mvc_output["pred"]
            if self.config.explicit_zero_prob:
                outputs["mvc_zero_probs"] = mvc_output["zero_probs"]

        if CLS:
            outputs["cls_output"] = self.cls_decoder(cell_emb)

        # Hidden states and attentions are tuple structure
        if output_hidden_states:
            outputs["hidden_states"] = bert_hidden_states
        if output_attentions:
            outputs["attentions"] = bert_attentions
        return outputs


    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the BERT pretrained model checkpoint (block size 512)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.max_position_embeddings
        self.config.max_position_embeddings = block_size
        self.position_encoder.pe = nn.Parameter(self.position_encoder.pe[:block_size])

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'bert-base-uncased', 'bert-large-uncased'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import BertModel
        print("loading weights from pretrained bert: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'bert-base-cased':  dict(num_hidden_layers=12, num_attention_heads=10, hidden_size=1280),  # 245M params
            'bert-large-cased': dict(num_hidden_layers=24, num_attention_heads=16, hidden_size=1536), # 340M params
        }[model_type]
        print("forcing vocab_size=1196, max_position_embeddings=1196, type_vocab_size=2")
        config_args['vocab_size'] = 1196 # always 1196 for BERT model checkpoints
        config_args['max_position_embeddings'] = 1196 # always 512 for BERT model checkpoints
        config_args['type_vocab_size'] = 2 # always 2 for BERT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['hidden_dropout_prob'] = override_args['dropout']
        # create a from-scratch initialized BERT model
        config = BERTConfig(**config_args)
        model = BERTForPreTraining(config)
        sd = model.state_dict()
        sd_keys = sd.keys()

        # init a huggingface/transformers model
        model_hf = BertModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        transposed = ['attention.self.query.weight', 'attention.self.key.weight', 'attention.self.value.weight', 'attention.output.dense.weight', 'intermediate.dense.weight', 'output.dense.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        # use_fused = False
        extra_args = dict(fused=True) if use_fused else dict()
        # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        # 使用NPU优化器
        optimizer = torch_npu.optim.NpuFusedAdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        
        # 使用APEX优化器
        # import apex
        # optimizer = apex.optimizers.NPUFusedAdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size//cfg.num_attention_heads, cfg.max_position_embeddings
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 256e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS, 910b NPU bfloat16 peak flops is 256 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

