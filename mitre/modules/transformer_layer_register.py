from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq.models.transformer import TransformerConfig
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

from .mutihead_attention_register import MultiheadAttentionRegister


class TransformerDecoderLayerRegisterBase(nn.Module):

    def __init__(
        self, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.normalize_before = cfg.decoder.normalize_before
        self.embed_dim = cfg.decoder.embed_dim
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.self_attn = self.build_self_attention(
            self.embed_dim,
            cfg,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.fc1 = self.build_w(
            self.embed_dim,
            cfg.decoder.ffn_embed_dim,
        )
        self.fc2 = self.build_w(
            cfg.decoder.ffn_embed_dim,
            self.embed_dim,
        )

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export) 
        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export) 
        self.need_attn = True
        self.onnx_trace = False

    def build_w(self, input_dim, output_dim):
        return quant_noise(nn.Linear(input_dim, output_dim), self.quant_noise, self.quant_noise_block_size)

    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttentionRegister(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not cfg.cross_self_attention,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=cfg.decoder.xformers_att_config,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.fc2(nn.functional.relu(self.fc1(x)))

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


# backward compatible with the legacy argparse format
class TransformerDecoderLayerRegister(TransformerDecoderLayerRegisterBase):
    def __init__(
        self, args, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            TransformerConfig.from_namespace(args),
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.args = args

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return super().build_self_attention(
            embed_dim,
            TransformerConfig.from_namespace(args),
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )