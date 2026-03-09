import torch
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.triton.batch_invariant.mla import decode_attention_fwd

# isort: off
from vllm_ascend.attention.mla_v1 import (
    AscendMLAImpl,
    AscendMLAMetadata,
)
# isort: on

class TritonMLAImpl(AscendMLAImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        block_size: int,
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        decode_meta = attn_metadata.decode
        assert decode_meta is not None

        q =  torch.cat([q_nope, q_pe], dim=-1)
        kv_c_and_k_pe_cache = torch.cat([k_nope, k_pe], dim=-1)

        B = q.shape[0]
        q_num_heads = q.shape[1]
        attn_output = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )

        # For batch invariance, use only 1 split to ensure deterministic reduction
        num_kv_splits = 1

        # TODO(lucas) Allocate ahead of time
        attn_logits = torch.empty(
            (
                B,
                q_num_heads,
                num_kv_splits,
                # NOTE(lucas) idk why the +1 is here but sglang has it so we
                # just mirror that
                self.kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q.device,
        )

        # Add a head dim of 1
        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        seq_lens = attn_metadata.decode.seq_lens
        seq_lens = seq_lens.to(q.device)

        decode_attention_fwd(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            attn_output,
            attn_metadata.decode.block_table,
            seq_lens,
            attn_logits,
            num_kv_splits,
            self.scale,
            PAGE_SIZE,
        )

        attn_output = attn_output.unsqueeze(2).permute(1, 0, 2, 3)

        return self._v_up_proj(attn_output)

