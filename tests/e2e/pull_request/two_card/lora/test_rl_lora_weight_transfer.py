#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
"""End-to-end test for RL LoRA weight transfer over HCCL.

The test first generates with the base model. The trainer then merges the
Alice LoRA adapter into the base model and transfers the resulting parameters
to vLLM. After verifying that the model identifies itself as Alice, the
trainer repeats the transfer with the Bob adapter and verifies that the live
model now identifies itself as Bob.

Topology (requires 2 NPUs):
- NPU 0: vLLM inference worker (rank 1 in the HCCL group)
- NPU 1: trainer / weight source (rank 0 in the HCCL group)

Refer to ``examples/rl/rlhf_http_hccl.py`` for the end-user workflow.

Run with::

    pytest tests/e2e/multicard/2-cards/test_rl_lora_weight_transfer.py
"""

import json
import os
import re
import threading

import pytest
import requests
import safetensors
import torch
import torch_npu  # noqa: F401  # registers the NPU backend
from transformers import AutoModelForCausalLM
from vllm.utils.network_utils import get_ip, get_open_port

from tests.e2e.conftest import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
ALICE_LORA_PATH = "charent/self_cognition_Alice"
BOB_LORA_PATH = "charent/self_cognition_Bob"

INFERENCE_WORLD_SIZE = 1
TRAINER_DEVICE_INDEX = 1
SELF_COGNITION_PROMPT = "Hi, my name is"
EXPECTED_ALICE_NAME = "alice"
EXPECTED_BOB_NAME = "bob"

SERVER_START_TIMEOUT = 2800
INIT_TIMEOUT = 120
UPDATE_TIMEOUT = 300
CONTROL_TIMEOUT = 60
DEFAULT_PACKED_BUFFER_SIZE_BYTES = 2**30
PACKED_BUFFER_HEADROOM_BYTES = 128 * 2**20

PEFT_PREFIX = "base_model.model."
LORA_A_PATTERN = re.compile(r"\.lora_A\.(?:default\.)?weight$")


def _log(message: str) -> None:
    print(f"[trainer] {message}", flush=True)


def _post(
    server: RemoteOpenAIServer,
    route: str,
    *,
    json_body=None,
    timeout: int = CONTROL_TIMEOUT,
):
    response = requests.post(
        server.url_for(route),
        json=json_body,
        timeout=timeout,
    )
    response.raise_for_status()
    return response


class _BackgroundPost(threading.Thread):
    """Run a blocking server collective concurrently with the trainer."""

    def __init__(
        self,
        server: RemoteOpenAIServer,
        route: str,
        *,
        json_body=None,
        timeout: int = CONTROL_TIMEOUT,
    ) -> None:
        super().__init__(daemon=True)
        self._server = server
        self._route = route
        self._json_body = json_body
        self._timeout = timeout
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            _post(
                self._server,
                self._route,
                json_body=self._json_body,
                timeout=self._timeout,
            )
            _log(f"background POST /{self._route} done")
        except BaseException as exc:  # noqa: BLE001
            self.error = exc
            _log(f"background POST /{self._route} failed: {exc!r}")

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError(
                f"server-side /{self._route} failed"
            ) from self.error


def _generate(client, model: str) -> str:
    response = client.completions.create(
        model=model,
        prompt=SELF_COGNITION_PROMPT,
        max_tokens=2,
        temperature=0,
    )
    return response.choices[0].text


def _build_lora_merged_model(lora_path: str) -> torch.nn.Module:
    """Load the base model and merge one PEFT LoRA adapter on NPU 1."""
    device = f"npu:{TRAINER_DEVICE_INDEX}"
    _log(f"loading base model on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device=device, dtype=torch.bfloat16)

    config_path = os.path.join(lora_path, "adapter_config.json")
    with open(config_path, encoding="utf-8") as config_file:
        lora_config = json.load(config_file)
    scaling = lora_config["lora_alpha"] / lora_config["r"]

    weights_path = os.path.join(lora_path, "adapter_model.safetensors")
    with safetensors.safe_open(weights_path, framework="pt") as adapter_file:
        adapter_weights = {
            key: adapter_file.get_tensor(key) for key in adapter_file.keys()
        }

    model_parameters = dict(model.named_parameters())
    lora_a_keys = sorted(
        key for key in adapter_weights if LORA_A_PATTERN.search(key)
    )
    if not lora_a_keys:
        raise ValueError(f"no LoRA A weights found in {weights_path}")

    for lora_a_key in lora_a_keys:
        lora_b_key = LORA_A_PATTERN.sub(
            lambda match: match.group(0).replace("lora_A", "lora_B"),
            lora_a_key,
        )
        if lora_b_key not in adapter_weights:
            raise KeyError(
                f"missing LoRA B weight for {lora_a_key}: {lora_b_key}"
            )

        parameter_name = lora_a_key
        if parameter_name.startswith(PEFT_PREFIX):
            parameter_name = parameter_name[len(PEFT_PREFIX) :]
        parameter_name = LORA_A_PATTERN.sub("", parameter_name) + ".weight"
        if parameter_name not in model_parameters:
            raise KeyError(
                f"base parameter {parameter_name!r} derived from "
                f"{lora_a_key!r} was not found"
            )

        lora_a = adapter_weights[lora_a_key].to(
            device=device,
            dtype=torch.float32,
        )
        lora_b = adapter_weights[lora_b_key].to(
            device=device,
            dtype=torch.float32,
        )
        delta = (lora_b @ lora_a) * scaling
        parameter = model_parameters[parameter_name]
        parameter.data = (parameter.data.float() + delta).to(parameter.dtype)

    _log(f"merged {len(lora_a_keys)} LoRA parameter pairs from {lora_path}")
    return model


def _collect_weight_metadata(train_model: torch.nn.Module):
    names: list[str] = []
    dtype_names: list[str] = []
    shapes: list[list[int]] = []
    max_tensor_bytes = 0

    for name, parameter in train_model.named_parameters():
        names.append(name)
        dtype_names.append(str(parameter.dtype).split(".")[-1])
        shapes.append(list(parameter.shape))
        max_tensor_bytes = max(
            max_tensor_bytes,
            parameter.numel() * parameter.element_size(),
        )

    packed_buffer_size_bytes = max(
        DEFAULT_PACKED_BUFFER_SIZE_BYTES,
        max_tensor_bytes + PACKED_BUFFER_HEADROOM_BYTES,
    )
    return names, dtype_names, shapes, packed_buffer_size_bytes


def _init_hccl_group(server: RemoteOpenAIServer):
    from vllm_ascend.distributed.weight_transfer.hccl_engine import (
        HCCLWeightTransferEngine,
    )

    master_address = get_ip()
    master_port = get_open_port()
    world_size = INFERENCE_WORLD_SIZE + 1
    init_info = {
        "master_address": master_address,
        "master_port": master_port,
        "rank_offset": 1,
        "world_size": world_size,
    }

    _log(
        f"initializing HCCL at {master_address}:{master_port}, "
        f"world_size={world_size}"
    )
    init_thread = _BackgroundPost(
        server,
        "init_weight_transfer_engine",
        json_body={"init_info": init_info},
        timeout=INIT_TIMEOUT,
    )
    init_thread.start()

    torch.npu.set_device(TRAINER_DEVICE_INDEX)
    trainer_group = HCCLWeightTransferEngine.trainer_init(
        {
            "master_address": master_address,
            "master_port": master_port,
            "world_size": world_size,
        }
    )
    init_thread.join()
    init_thread.raise_if_failed()
    _log("HCCL process group established")
    return trainer_group


def _start_weight_update(server: RemoteOpenAIServer) -> bool:
    response = requests.post(
        server.url_for("start_weight_update"),
        json={"is_checkpoint_format": True},
        timeout=CONTROL_TIMEOUT,
    )
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True


def _transfer_weights(
    server: RemoteOpenAIServer,
    train_model: torch.nn.Module,
    trainer_group,
) -> None:
    from vllm_ascend.distributed.weight_transfer.hccl_engine import (
        HCCLTrainerSendWeightsArgs,
        HCCLWeightTransferEngine,
    )

    _post(server, "pause")
    use_lifecycle_endpoints = _start_weight_update(server)

    names, dtype_names, shapes, packed_buffer_size_bytes = (
        _collect_weight_metadata(train_model)
    )
    update_info = {
        "names": names,
        "dtype_names": dtype_names,
        "shapes": shapes,
        "packed": True,
        "packed_buffer_size_bytes": packed_buffer_size_bytes,
    }
    if not use_lifecycle_endpoints:
        update_info["is_checkpoint_format"] = True

    _log(f"broadcasting {len(names)} model parameters")
    update_thread = _BackgroundPost(
        server,
        "update_weights",
        json_body={"update_info": update_info},
        timeout=UPDATE_TIMEOUT,
    )
    update_thread.start()

    trainer_args = HCCLTrainerSendWeightsArgs(
        group=trainer_group,
        packed=True,
        packed_buffer_size_bytes=packed_buffer_size_bytes,
    )
    HCCLWeightTransferEngine.trainer_send_weights(
        iterator=train_model.named_parameters(),
        trainer_args=trainer_args,
    )
    update_thread.join()
    update_thread.raise_if_failed()

    if use_lifecycle_endpoints:
        _post(server, "finish_weight_update")
    _post(server, "resume")
    _log("weight transfer complete; generation resumed")


def _merge_and_transfer(
    server: RemoteOpenAIServer,
    lora_path: str,
    trainer_group,
) -> None:
    train_model = _build_lora_merged_model(lora_path)
    try:
        _transfer_weights(server, train_model, trainer_group)
    finally:
        del train_model
        torch.npu.empty_cache()


@pytest.mark.skipif(
    torch.npu.device_count() < 2,
    reason="RL LoRA weight transfer test requires at least 2 NPUs.",
)
def test_rl_lora_weight_transfer():
    port = get_open_port()
    server_args = [
        "--enforce-eager",
        "--weight-transfer-config",
        '{"backend": "nccl"}',
        "--tensor-parallel-size",
        str(INFERENCE_WORLD_SIZE),
        "--max-model-len",
        "1024",
        "--gpu-memory-utilization",
        "0.6",
        "--port",
        str(port),
        "--trust-remote-code",
    ]
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": "0",
        "VLLM_ASCEND_ENABLE_NZ": "0",
    }

    _log(f"starting vLLM inference on NPU 0, port {port}")
    with RemoteOpenAIServer(
        MODEL_NAME,
        vllm_serve_args=server_args,
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
        auto_port=False,
        max_wait_seconds=SERVER_START_TIMEOUT,
    ) as server:
        client = server.get_client()

        _log("step 1: generating with the base model")
        baseline_output = _generate(client, MODEL_NAME)
        _log(f"base model output: {baseline_output!r}")

        trainer_group = _init_hccl_group(server)

        _log("step 2: transferring Alice LoRA-merged weights")
        _merge_and_transfer(server, ALICE_LORA_PATH, trainer_group)
        alice_output = _generate(client, MODEL_NAME)
        _log(f"Alice model output: {alice_output!r}")

        _log("step 3: transferring Bob LoRA-merged weights")
        _merge_and_transfer(server, BOB_LORA_PATH, trainer_group)
        bob_output = _generate(client, MODEL_NAME)
        _log(f"Bob model output: {bob_output!r}")

    assert EXPECTED_ALICE_NAME in alice_output.lower(), (
        "Alice LoRA transfer did not make the model identify as Alice: "
        f"{alice_output!r}"
    )
    assert EXPECTED_BOB_NAME in bob_output.lower(), (
        "Bob LoRA transfer did not make the model identify as Bob: "
        f"{bob_output!r}"
    )
    assert alice_output != bob_output, (
        "Alice and Bob produced identical outputs; the second weight transfer "
        "may not have taken effect."
    )
    assert baseline_output != alice_output, (
        "The Alice weight transfer did not change the base model output."
    )
