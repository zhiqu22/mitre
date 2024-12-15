from typing import Dict, List, Optional

import torch
from fairseq.modules import (
    PositionalEmbedding,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
from fairseq.models.transformer import (
    TransformerConfig,
    TransformerEncoderBase,
)

def module_name_fordropout(module_name: str) -> str:
    if module_name == 'TransformerEncoderRegisterBase':
        return 'TransformerEncoderRegister'
    else:
        return module_name


class TransformerEncoderRegisterBase(TransformerEncoderBase):
    def __init__(self, cfg, dictionary, embed_tokens, return_fc=False):
        super().__init__(
            cfg,
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )
        self.register_factor = cfg.register_factor
        self.register_embed_position = (PositionalEmbedding(cfg.max_source_positions, embed_tokens.embedding_dim, self.padding_idx,learned=cfg.encoder.learned_pos,))

    def forward(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings,
        )

    def forward_scriptable(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        real_lengths = (~src_tokens.eq(self.padding_idx)).sum(dim=1)
        register_nums = torch.ceil(real_lengths.float() / self.register_factor).int()
        max_register_num = register_nums.max().item()
        total_token_nums = src_tokens.size(1) + max_register_num
        batch_size = register_nums.size(0)

        # create registers
        registers = src_tokens[range(src_tokens.size(0)), torch.argmax(src_tokens, dim=-1)].unsqueeze(1).repeat(1, max_register_num)
        # padding
        pads = torch.full_like(registers, self.padding_idx)
        expanded_src_tokens = torch.cat((pads, src_tokens, registers), dim=1)

        indices = torch.arange(total_token_nums).expand(src_tokens.size(0), -1).to(src_tokens.device)
        indices = indices + register_nums.unsqueeze(1)
        
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, indices.size(1)).contiguous()
        src_tokens = expanded_src_tokens[batch_indices, indices]
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = (
            torch.tensor(src_tokens.device.type == "xla") or encoder_padding_mask.any()
        )
        # Torchscript doesn't handle bool Tensor correctly, so we need to work around.
        if torch.jit.is_scripting():
            has_pads = torch.tensor(1) if has_pads else torch.tensor(0)
        
        # embedding
        x = self.embed_scale * self.embed_tokens(expanded_src_tokens)
        x_1 = x[:,:total_token_nums,:] + self.embed_positions(expanded_src_tokens[:,:total_token_nums])
        x_2 = x[:,total_token_nums:,:] + self.register_embed_position(expanded_src_tokens[:,total_token_nums:])
        if self.segment_flag:
            x_1 = x_1 + self.segment_embed(torch.zeros_like(expanded_src_tokens[:,:total_token_nums]))
            x_2 = x_2 + self.segment_embed(torch.ones_like(expanded_src_tokens[:,total_token_nums:]))

        x = torch.cat((x_1, x_2), dim=1)
        x = x[batch_indices, indices]

        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        
        # account for padding while computing the representation
        x = x * (
            1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x)
        )
        # B x T x C -> T x B x C
        x = x.transpose(0, 1).contiguous()
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": src_tokens,
            "src_lengths": [],
            "register_nums": [register_nums],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]

        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = (encoder_out["src_tokens"]).index_select(0, new_order)
        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]
        
        if len(encoder_out["register_nums"]) == 0:
            register_nums = []
        else:
            register_nums = [(encoder_out["register_nums"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
            "register_nums": register_nums,
        }

class TransformerEncoderRegister(TransformerEncoderRegisterBase):
    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
        )

    def build_encoder_layer(self, args):
        return super().build_encoder_layer(
            TransformerConfig.from_namespace(args),
        )