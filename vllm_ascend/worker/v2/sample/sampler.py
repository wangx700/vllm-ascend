# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/sampler.py.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import numpy as np
import torch
from contextlib import contextmanager
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.sample.min_p import apply_min_p
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample
from vllm_ascend.worker.v2.sample.logprob import compute_token_logprobs as ascend_compute_token_logprobs

@contextmanager
def update_compute_token_logprobs():
    from vllm.v1.worker.gpu.sample import logprob
    original_compute_token_logprobs = logprob.compute_token_logprobs
    try:
        logprob.compute_token_logprobs = ascend_compute_token_logprobs
        yield
    finally:
        logprob.compute_token_logprobs = original_compute_token_logprobs


class AscendSampler(Sampler):
    def __call__(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        cu_num_logits_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ):
        with update_compute_token_logprobs():
            return super().__call__(
                logits,
                idx_mapping,
                idx_mapping_np,
                cu_num_logits_np,
                pos,
                input_ids,
                expanded_local_pos,
            )

    def sample(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Override sample method because we need to override triton operators
        called in the method.
        """
        # Copy logits to a new FP32 tensor.
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

        # Apply logit bias (e.g., allowed_token_ids, min_tokens) in place.
        self.logit_bias_state.apply_logit_bias(logits, idx_mapping, idx_mapping_np, pos)

        # Apply penalties in place.
        self.penalties_state.apply_penalties(
            logits,
            idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
            self.num_speculative_tokens,
        )

        # Apply temperature in place.
        apply_temperature(logits, idx_mapping, self.sampling_states.temperature.gpu)

        # Apply min_p in place if any request has a non-zero min_p.
        do_min_p = self.sampling_states.do_min_p(idx_mapping_np)
        if do_min_p:
            apply_min_p(logits, idx_mapping, self.sampling_states.min_p.gpu)

        # Apply top_k and/or top_p. This might return a new tensor.
        do_top_k = self.sampling_states.do_top_k(idx_mapping_np)
        top_k = self.sampling_states.top_k.gpu[idx_mapping] if do_top_k else None
        do_top_p = self.sampling_states.do_top_p(idx_mapping_np)
        top_p = self.sampling_states.top_p.gpu[idx_mapping] if do_top_p else None
        if do_top_k or do_top_p:
            logits = apply_top_k_top_p(logits, top_k, top_p)

        # Sample the next token.
        sampled = gumbel_sample(
            logits,
            idx_mapping,
            self.sampling_states.temperature.gpu,
            self.sampling_states.seeds.gpu,
            pos,
            apply_temperature=False,
        )
        return sampled, logits
