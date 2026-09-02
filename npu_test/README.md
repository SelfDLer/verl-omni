# Qwen3-Omni rollout/actor Pearson diagnostics

This directory contains NPU-only, one-step ablations for
`run_qwen3_omni_thinker_gspo_npu_nextqa_v1.sh`. They are intended to locate
the source of a low `training/rollout_actor_probs_pearson_corr` without
changing the training algorithm.

The metric compares `exp(actor_old_log_probs)` with
`exp(rollout_log_probs)` over valid response tokens. The actor log
probabilities are recomputed after rollout and before the actor update, so a
low value measures inference-path disagreement rather than learning progress.

## Run

From the repository root on the NPU machine:

```bash
git switch video-dev
git pull
bash npu_test/run_nextqa_pearson_matrix.sh
```

The default matrix runs four independent one-step jobs with the same seed:

| Case | Change from the current recipe | What an improvement isolates |
| --- | --- | --- |
| `baseline` | None | Reproduces the current value on the diagnostic batch |
| `no_audio` | `use_audio_in_video=false` | Audio expansion, audio/video interleaving, or multimodal RoPE |
| `eager` | `rollout.enforce_eager=true` | vLLM graph execution or an NPU kernel numerical difference |
| `no_rmpad` | `model.use_remove_padding=false` | Actor-side sequence packing, masks, or position IDs |

Two optional cases change more than one execution detail:

```bash
bash npu_test/run_nextqa_pearson_matrix.sh tp4 no_audio_eager
```

- `tp4` tests whether vLLM tensor-parallel partitioning changes the result.
- `no_audio_eager` checks for an interaction between audio inputs and graph
  execution.

Useful environment overrides are:

```bash
DIAG_TRAIN_BATCH_SIZE=8 \
DIAG_MAX_RESPONSE_LENGTH=1024 \
DIAG_TOTAL_STEPS=1 \
RESULT_DIR=/absolute/path/to/results \
bash npu_test/run_nextqa_pearson_matrix.sh baseline no_audio eager no_rmpad
```

All ordinary launcher variables (`MODEL_PATH`, `TRAIN_FILE`, `VAL_FILE`,
`N_GPUS_PER_NODE`, `NNODES`, and `ASCEND_HOME_PATH`) are preserved. Use
absolute paths when overriding them.

Each case writes `run.log` and `exit_code.txt`. The runner also creates
`environment.txt` and `summary.csv` under the result directory. To summarize
copied logs again:

```bash
python3 npu_test/summarize_pearson.py /path/to/results --csv /path/to/summary.csv
```

Please return `summary.csv`, `environment.txt`, and any failed case's final
200 log lines.

## Reading the matrix

- `no_audio` reaches `>0.99`: inspect the audio hop padding, vLLM-Omni
  multimodal preprocessing, and audio-aware RoPE indices.
- `eager` reaches `>0.99`: inspect graph-mode kernels and fused MoE execution.
- `no_rmpad` reaches `>0.99`: inspect actor packing, response masks, and
  multimodal position IDs.
- `tp4` moves substantially while the other cases do not: inspect the
  tensor-parallel/fused-MoE path.
- Every case remains near `0.95`: the leading hypothesis is systematic
  HF/FSDP2 versus vLLM-Omni MoE expert-routing or kernel disagreement; the
  next test should compare route IDs and per-token log probabilities.

Qwen3-Omni-30B-A3B is an MoE model. The pinned stack can record vLLM expert
routes, but actor-side routing replay is currently wired for Megatron and
VeOmni rather than this recipe's FSDP2 actor. Enabling rollout route capture
alone therefore cannot make FSDP2 reproduce the rollout route; it is useful
as a follow-up diagnostic, not a complete fix.
