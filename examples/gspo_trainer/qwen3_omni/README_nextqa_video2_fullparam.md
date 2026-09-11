# Full-parameter NExT-QA training on Ascend

Use the environment and data from [video2](README_nextqa_video2.md).
This entry point loads the original full-depth checkpoint; it never invokes
the debug exporter. Text, vision and audio Thinker parameters are trainable.
The output-only Talker/codec are removed by the existing adapter.

```bash
export MODEL_PATH=/models/Qwen3-Omni-30B-A3B-Instruct
export TRAIN_FILE=/data/nextqa/train.parquet
export VAL_FILE=/data/nextqa/validation.parquet
export OUTPUT_DIR=/checkpoints/nextqa_video2_fullparam_run1
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_nextqa_video2_fullparam_npu.sh
```

Source CANN/ATB beforehand and expose ffmpeg on every Ray worker. Use the same
model/data paths on all nodes, and persistent shared storage for OUTPUT_DIR
on multi-node runs. Do not point MODEL_PATH to the reduced debug export.

## Training defaults

| Setting | Value |
| --- | --- |
| Devices | 16 NPUs, rollout TP=2 |
| Prompt batch | 64 questions (4 per NPU) |
| Responses per question | 8; 512 responses per rollout batch on 16 NPUs |
| Sampling | temperature=1, top-p=1, top-k=-1 |
| Update | One PPO epoch, one optimizer update per rollout batch |
| Optimizer | Full-parameter AdamW, LR=1e-6, 5% warmup, then constant |
| Gradient clipping / weight decay | 1.0 / 0.01 |
| Objective | GSPO with GRPO group advantages; binary answer accuracy |
| Input budget | 8192 prompt tokens, 1024 response tokens |
| Actor / logprob microbatch | 1, padding retained, dynamic batching disabled |
| Rollout graphs | Enabled, capture sizes [1,2,4,8] |
| Validation | Full retained validation set before training and every 25 steps; greedy |
| Checkpoints | Every 25 steps, model + optimizer + extra state; automatic resume |
| Run budget | Up to 500 rollout steps / 10 data epochs, whichever is reached first |

Each question has several sampled responses so that correct and incorrect
answers can provide a group-relative learning signal. More responses cannot
guarantee mixed rewards: all-correct and all-wrong groups still have zero
advantage. We retain the existing narrow sequence clipping and avoid repeated
epochs on the same rollout. These are starting hyperparameters, not measured
optimal settings or a promise of reward improvement. See the
[GSPO description](https://qwenlm.github.io/blog/gspo/) for the algorithm.

Video and its extracted soundtrack remain separate input segments
(`use_audio_in_video=false`). Reward is the existing exact answer-tag match;
the converter supplies nonempty ground truths such as `<answer>A</answer>`.
There is no extra reward for verbosity or formatting alone. Raw rollout
logprobs and independent actor recomputation remain enabled. Importance and
rejection corrections are disabled; they must not hide input/parity defects.

## Judge progress and resume

Compare validation accuracy against the pretraining baseline, not training
sampling reward alone. Check actor gradient norm and learning rate, response
length/truncation, and decoded validation answers under OUTPUT_DIR/validation.
If rewards remain flat, inspect whether answers are truncated/missing tags or
groups have identical rewards before increasing learning rate. A zero-gradient
run does not establish successful weight updates. A rising training reward
with falling validation accuracy indicates the run is not meeting the goal.

For sampled-token parity dumps, optionally append
`++trainer.consistency_debug_dir=/path/to/parity`; this writes every step and
can consume substantial storage. Use the full-depth model when assessing the
actual training configuration. NPU memory fit, parity and reward gains still
require validation on the target machine.

Repeat the same command with the same OUTPUT_DIR to resume. Checkpoints are
not automatically deleted, so a high-scoring earlier validation checkpoint
remains available. Full optimizer checkpoints are large: provision storage or
explicitly set `trainer.max_actor_ckpt_to_keep=3` after deciding retention;
that retains recent checkpoints, not the best-scoring checkpoint. Best-model
selection is manual using the validation results. Start an independent run
with a new OUTPUT_DIR and EXPERIMENT_NAME.

Environment overrides: N_GPUS_PER_NODE, NNODES, ROLLOUT_TP, TRAIN_BATCH_SIZE,
ROLLOUT_N, LEARNING_RATE, TOTAL_TRAINING_STEPS, EXPERIMENT_NAME, OUTPUT_DIR.
Final Hydra overrides are forwarded, for example:

```bash
LEARNING_RATE=5e-7 TOTAL_TRAINING_STEPS=1000 \
  bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_nextqa_video2_fullparam_npu.sh \
  trainer.test_freq=50 trainer.save_freq=50
```

Changing graph capture or concurrency can change memory consumption; reducing
TRAIN_BATCH_SIZE alone does not reduce a single sample's activation memory.
Inspect effective configuration without loading weights with `--cfg job`.
