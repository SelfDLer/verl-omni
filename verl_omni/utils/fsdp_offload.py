# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Order NPU host transfers before FSDP2 reads CPU shards to rebuild padding."""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn import Module


def install_fsdp2_cpu_transfer_guard(model: Module, device_type: str) -> bool:
    """Wait for NPU-to-CPU copies inside this model's recursive ``_apply``.

    FSDP2 repads uneven shards inside ``_apply``, before ``Module.to`` returns.
    Synchronizing after ``model.to('cpu', non_blocking=True)`` is too late:
    the CPU may already have copied unfinished DMA data into a new allocation.
    Wrapping the conversion function waits before child FSDP modules repad,
    including offload calls in inherited engine and checkpoint methods.

    Install on the root after FSDP2 wrapping and before loading state. The
    change is local to this model; each NPU-to-CPU tensor transfer blocks.
    CPU conversions, device-bound transfers and non-NPU models are unchanged.

    Args:
        model: Root module after FSDP2 wrapping.
        device_type: Device type of the FSDP mesh.

    Returns:
        Whether a new guard was installed.

    Raises:
        TypeError: An NPU model is not an FSDP2 module.
    """
    if device_type != "npu" or getattr(model, "_verl_omni_cpu_transfer_guard", False):
        return False

    from torch.distributed.fsdp import FSDPModule
    from verl.utils.device import get_torch_device

    if not isinstance(model, FSDPModule):
        raise TypeError("CPU transfer guard requires an FSDP2 root module")
    original_apply = model._apply
    device_module = get_torch_device()

    def guarded_apply(self, fn, *args, **kwargs):
        def convert(tensor):
            source = tensor.device
            result = fn(tensor)
            if source.type == "npu" and result.device.type == "cpu":
                # This must finish before nn.Module._apply returns to a child
                # FSDPModule._apply, which immediately calls reset_sharded_param.
                device_module.synchronize(source)
            return result

        return original_apply(convert, *args, **kwargs)

    model._apply = MethodType(guarded_apply, model)
    model._verl_omni_cpu_transfer_guard = True
    return True
