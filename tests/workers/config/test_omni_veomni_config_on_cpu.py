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

"""Compose the actual VeOmni actor/ref groups with the omni trainer."""

import importlib.util
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_veomni_actor_reference_composition():
    root = Path(__file__).resolve().parents[3]
    reference = os.environ.get("VERL_REFERENCE_DIR")
    if reference:
        verl_dir = Path(reference) / "verl"
    else:
        verl_dir = Path(importlib.util.find_spec("verl").origin).parent
    with initialize_config_dir(config_dir=str(root / "verl_omni/trainer/config"), version_base=None):
        cfg = compose(
            config_name="omni_trainer",
            overrides=[
                f"hydra.searchpath=[file://{(verl_dir / 'trainer/config').as_posix()}]",
                "actor@actor_rollout_ref.actor=omni_veomni_actor",
                "ref@actor_rollout_ref.ref=omni_veomni_ref",
                "actor_rollout_ref.actor._target_=verl_omni.workers.config.omni.OmniVeOmniActorConfig",
                "actor_rollout_ref.actor.freeze_vision_tower=true",
                "actor_rollout_ref.actor.veomni.attn_implementation=sdpa",
            ],
        )
    actor = OmegaConf.to_container(cfg.actor_rollout_ref.actor, resolve=True)
    ref = OmegaConf.to_container(cfg.actor_rollout_ref.ref, resolve=True)
    assert actor["_target_"].endswith("OmniVeOmniActorConfig")
    assert ref["_target_"] == actor["_target_"]
    assert actor["strategy"] == ref["strategy"] == "veomni"
    assert actor["veomni"]["_target_"].endswith("OmniVeOmniEngineConfig")
    assert actor["veomni"]["freeze_vision_tower"] is True
    assert ref["veomni"]["forward_only"] is True
    assert ref["veomni"]["attn_implementation"] == "sdpa"
    assert actor["optim"]["_target_"].endswith("VeOmniOptimizerConfig")
    assert "fsdp_config" not in actor
    assert actor["trainer_type"] == "policy_gradient"
