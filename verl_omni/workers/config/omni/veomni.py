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

"""VeOmni configuration for the omni actor and reference policy."""

from dataclasses import dataclass, field

from verl.workers.config import VeOmniActorConfig, VeOmniEngineConfig

from .actor import OmniLossConfig


@dataclass
class OmniVeOmniEngineConfig(VeOmniEngineConfig):
    """Configure encoder trainability before VeOmni wraps the model."""

    freeze_vision_tower: bool = False
    freeze_audio_tower: bool = False


@dataclass
class OmniVeOmniActorConfig(VeOmniActorConfig):
    """Preserve omni loss settings while selecting the VeOmni engine."""

    veomni: OmniVeOmniEngineConfig = field(default_factory=OmniVeOmniEngineConfig)
    trainer_type: str = "policy_gradient"
    omni_loss: OmniLossConfig = field(default_factory=OmniLossConfig)

    def __post_init__(self):
        super().__post_init__()
        if self.trainer_type not in ("policy_gradient", "direct_preference"):
            raise ValueError(f"Invalid omni trainer_type={self.trainer_type!r}.")
        if self.trainer_type == "direct_preference" and self.omni_loss is None:
            raise ValueError("omni_loss is required for direct_preference training.")
