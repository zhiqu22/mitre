from typing import Any, Dict, List, Optional
from torch import Tensor
import torch
from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer import TransformerDecoderBase
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.models.transformer import TransformerConfig

from fairseq import utils

from mitre.modules import TransformerDecoderLayerRegister

def module_name_fordropout(module_name: str) -> str:
    if module_name == 'TransformerDecoderRegisterBase':
        return 'TransformerDecoderRegister'
    else:
        return module_name
    
class TransformerDecoderRegisterBase(TransformerDecoderBase):

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)
        self.heads_num = cfg.decoder_attention_heads
    
    def build_decoder_layer(self, cfg, no_encoder_attn=False,):
        # no cross-attention
        layer = TransformerDecoderLayerRegister(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
    ):
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        src_tokens: Optional[Tensor] = None
        register_nums: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None:
            src_tokens = encoder_out["src_tokens"]
            register_nums = encoder_out["register_nums"][0]
        
        # replace the inference trigger with langtok
        # namely, enc-tgt-dec-tgt strategy
        if prev_output_tokens[0][0].item() != src_tokens[0][-1].item():
            prev_output_tokens[:, 0] = src_tokens[:, -1]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1).contiguous()
        
        # cat source and target
        cat_flag = False
        # incremental_state is None means it is in training.
        # (incremental_state is not None and incremental_state == {}) means this is the first turn of incremental decoding.
        if incremental_state is None or (incremental_state is not None and incremental_state == {}):
            x = torch.cat([enc, x], dim=0)
            cat_flag = True
            tokens = torch.cat([src_tokens, prev_output_tokens], dim=1)

        self_attn_padding_mask: Optional[Tensor] = None
        if cat_flag:
            if tokens.eq(self.padding_idx).any():
                self_attn_padding_mask = tokens.eq(self.padding_idx)
        
        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        src_length = enc.shape[0]
        self_attn_mask = self.build_future_mask(x, src_length, register_nums=register_nums, incremental_state=incremental_state)
        for idx, layer in enumerate(self.layers):
            x, layer_attn, _ = layer(
                x,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)
        
        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if cat_flag:
            x = x[src_length:,:,:]

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        # since the flow is concatenated, we have to remove the source part from flow.
        
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        return x, {"attn": [attn],
                   "inner_states": inner_states,}
        
    def build_future_mask(self, tensor, src_length, register_nums, incremental_state):
        b = register_nums.size(0)
        ns = src_length - register_nums
        if not incremental_state:
            dim = tensor.size(0)
            if (
                self._future_mask.size(0) == 0
                or self._future_mask.size(0) < dim
                ):
                self._future_mask = torch.triu(utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1)
            if self._future_mask.device == tensor.device:
                mask = self._future_mask[:dim, :dim].clone()
            else:
                mask = self._future_mask[:dim, :dim].to(tensor, copy=True)

            mask[:src_length, :src_length] = 0.

            # masking
            batch_mask = mask.unsqueeze(0).expand(b, -1, -1).clone()

            row_indices = torch.arange(dim, device=mask.device).view(1, dim, 1)
            col_indices = torch.arange(dim, device=mask.device).view(1, 1, dim)

            register_start = ns.view(b, 1, 1)
            register_end = (ns + register_nums).view(b, 1, 1)

            source_to_register_mask = (row_indices < ns.view(b, 1, 1)) & \
                                  (col_indices >= register_start) & (col_indices < register_end)

            target_to_source_mask = (row_indices >= register_end) & (col_indices < ns.view(b, 1, 1))

            batch_mask[source_to_register_mask] = float('-inf')
            batch_mask[target_to_source_mask] = float('-inf')

        else:
            # inference
            prev_length = next(iter(incremental_state.values()))["prev_key"].shape[2]
            total_length = prev_length + 1  # current step

            mask = torch.zeros(total_length, device=tensor.device)
            batch_mask = mask.unsqueeze(0).expand(b, -1).clone()

            token_indices = torch.arange(total_length, device=tensor.device).view(1, -1)
            target_to_source_mask = token_indices < ns.view(b, 1)

            batch_mask[target_to_source_mask] = float('-inf')
            batch_mask = batch_mask.unsqueeze(1)

        batch_mask = batch_mask.unsqueeze(1).expand(-1, self.heads_num, -1, -1)
        batch_mask = batch_mask.reshape(b * self.heads_num, batch_mask.shape[2], batch_mask.shape[3])

        return batch_mask
    
class TransformerDecoderRegister(TransformerDecoderRegisterBase):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        super().build_output_projection(
            TransformerConfig.from_namespace(args), dictionary, embed_tokens
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return super().build_decoder_layer(
            TransformerConfig.from_namespace(args), no_encoder_attn=no_encoder_attn
        )