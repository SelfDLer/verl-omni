# Qwen3-Omni Thinker with VeOmni

The VeOmni engine trains the Thinker and uses vLLM-Omni for text-token rollout.
Image, video and audio preprocessing stays in the existing Qwen3-Omni processor.
The engine converts HF audio features to VeOmni's flat frame representation and
supplies the global modality masks needed by the native VeOmni model.
Compatibility handling preserves verl's four-axis position IDs and bridges the
new Transformers causal-mask API. SDPA audio attention runs each encoder window
independently so clips do not cross-attend.

## Versions

- verl: `.github/verl_pin.txt` (`fefb080262e1c015a0ea05f958822a6a512dc795`).
- VeOmni: `.github/veomni_pin.txt` (`f90b3dc6fbb0ce693745223cc7a94064123dbf4d`, v0.1.11).
- Ascend A3: `docker/Dockerfile.a3.npu`, Torch 2.10.0, torch-npu 2.10.0.post4,
  vLLM 0.28.0 and the repository's vLLM-Ascend/vLLM-Omni pins.
- Transformers: the repository's `pyproject.toml` requirement (5.13.0–5.14.1).

The A3 image installs VeOmni without its extras, since its NPU extra pins Torch
2.7.1. Do not install `veomni[npu]` over this image.

## Run on Ascend NPU

Use the pinned A3 image and source the CANN/ATB environment before launching.

```bash
NUM_NPUS=8 ROLLOUT_TP=2 \
MODEL_PATH=/models/Qwen3-Omni-30B-A3B-Instruct \
TRAIN_FILE=/data/train.parquet VAL_FILE=/data/test.parquet \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_veomni.sh
```

The launcher selects `trainer.device=npu`, disables Ascend NZ weight layout in
both the environment and rollout config, and uses eager rollout execution with
Torch compilation disabled for actor/reference. `NUM_NPUS` is the number of NPUs
per node, `NNODES` defaults to one, and `ROLLOUT_TP` must divide `NUM_NPUS`.
verl retains the configuration names `n_gpus_per_node`, `*_per_gpu` and
`gpu_memory_utilization` for NPU workers too.

The recipe uses SDPA, padded micro-batches of one sample, full-parameter Thinker
text training, frozen vision/audio towers, and parameter/optimizer offload.
The card count and token limits are starting settings, not measured capacity
guarantees. Change the data and reward configuration for your task.

Keep the SDPA/padded defaults for the initial NPU experiment. Packed attention
and sequence parallelism require a separately validated Ascend attention backend.

The actor and reference must select the same VeOmni configuration groups. The
launcher explicitly overrides the actor `_target_`, because `omni_trainer`'s
`_self_` otherwise selects the FSDP actor dataclass after group composition.
`actor.veomni.freeze_audio_tower` controls audio trainability;
`actor.freeze_vision_tower` feeds `actor.veomni.freeze_vision_tower`.

VeOmni owns model loading, FSDP2/EP parallelization, optimizer/offload and
checkpoint state. verl owns RL loss calculation and conversion of stacked MoE
weights into per-expert HF names for rollout. Exported floating weights use bf16.
Packed sequence boundaries come from sample offsets, not mRoPE resets.

LoRA, Talker training, fused logprob kernels and router replay are rejected
explicitly. Ulysses requires packed inputs and a compatible VeOmni attention
backend; multi-device EP/SP and real-model NPU convergence require accelerator
validation before production use.

## AVQA NPU experiment

The AVQA launcher reuses the NPU VeOmni recipe with `QwenOmniRLHFDataset`,
16 kHz audio, choice rewards and the existing AVQA GSPO clipping settings:

```bash
NUM_NPUS=16 ROLLOUT_TP=2 \
MODEL_PATH=/models/Qwen3-Omni-30B-A3B-Instruct \
TRAIN_FILE=$HOME/data/avqa_r1_6k/train.parquet \
VAL_FILE=$HOME/data/avqa_r1_6k/validation.parquet \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_veomni_npu_avqa_v1.sh
```

Source CANN/ATB before running. Defaults are two updates, two responses per prompt,
4096 prompt tokens, 1024 response tokens and four concurrent rollout sequences.
The global prompt batch and PPO mini-batch default to the total NPU count.
Vision/audio towers stay frozen. Checkpoints are saved each step; validation and
automatic resume are disabled for this initial run. Set `LOG_FILE` to choose the
log path, or find it under `logs/gspo_avqa_veomni_npu_<timestamp>.log`.
Trailing Hydra arguments override defaults, for example
`trainer.total_training_steps=100 trainer.save_freq=20 trainer.test_freq=10`.

## Validation

```bash
pytest -q tests/workers/test_omni_veomni_engine_on_cpu.py \
  tests/workers/test_omni_veomni_native_on_cpu.py \
  tests/workers/config/test_omni_veomni_config_on_cpu.py

# Requires Ascend runtime and local GSM8K parquet files; builds tiny weights.
NUM_NPUS=2 TRAIN_FILE=/data/gsm8k/train.parquet VAL_FILE=/data/gsm8k/test.parquet \
bash tests/npu_smoke/run_npu_smoke_omni_veomni.sh
```

The NPU smoke uses TP=1 by default, runs two updates with the reference policy
enabled and saves every step. Logs go to `logs/npu_smoke/veomni_<timestamp>`;
set `LOG_DIR` to override the location. Extra arguments pass through as Hydra
overrides. Separately run resume with `trainer.resume_mode=auto` and a
larger step limit, using the same output directory. Validate real multimodal
data, selected-token logprobs before/after weight reload, and numerical agreement
with the FSDP baseline on the target hardware. CPU tests do not establish NPU
kernel correctness or end-to-end training convergence.
