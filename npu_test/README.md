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

The default matrix runs six independent one-step jobs with the same seed. To
avoid decoding the complete NExT-QA train and validation sets before every
case, it selects 64 train samples and 8 validation samples before applying the
exact multimodal length filter:

| Case | Change from the current recipe | What an improvement isolates |
| --- | --- | --- |
| `baseline` | None | Reproduces the current value on the diagnostic batch |
| `no_audio` | `use_audio_in_video=false` | Audio expansion, audio/video interleaving, or multimodal RoPE |
| `no_init_sync` | Keep rollout in its initial checkpoint-loaded state | Initial sleep/reload lifecycle |
| `tp4` | Change rollout tensor parallelism from 2 to 4 | Tensor-parallel or fused-MoE partitioning |
| `low_concurrency` | Reduce vLLM `max_num_seqs` from 128 to 8 | Batch-shape-dependent vLLM/MoE execution |
| `batch_invariant` | Enable vLLM batch-invariant deterministic execution | Dynamic batching or scheduling sensitivity |

Previously tested or optional cases remain available:

```bash
bash npu_test/run_nextqa_pearson_matrix.sh eager no_rmpad no_audio_eager
```

- `eager` disables graph execution.
- `no_rmpad` disables actor-side remove-padding.
- `no_audio_eager` checks for an interaction between audio inputs and graph
  execution.

`no_init_sync` is diagnostic-only. Both actor and rollout initially load the
same checkpoint. This case skips both the first rollout sleep and the first
actor-to-rollout reload, because vLLM-Ascend cannot safely execute after
sleeping and waking the checkpoint-loaded state without a weight reload. It
still marks the untouched rollout checkpoint as model version 0; this metadata
is required for trajectory staleness metrics but does not transfer or reload
any tensors. Normal training and every other case retain the standard initial
and per-step sleep/reload lifecycle.

Useful environment overrides are:

```bash
DIAG_TRAIN_BATCH_SIZE=8 \
DIAG_TRAIN_MAX_SAMPLES=64 \
DIAG_VAL_MAX_SAMPLES=8 \
DIAG_MAX_RESPONSE_LENGTH=1024 \
DIAG_TOTAL_STEPS=1 \
RESULT_DIR=/absolute/path/to/results \
bash npu_test/run_nextqa_pearson_matrix.sh
```

All ordinary launcher variables (`MODEL_PATH`, `TRAIN_FILE`, `VAL_FILE`,
`N_GPUS_PER_NODE`, `NNODES`, and `ASCEND_HOME_PATH`) are preserved. Use
absolute paths when overriding them. The sample caps apply only to this
diagnostic runner; the normal training launcher still uses the full datasets.

Each case writes `run.log` and `exit_code.txt`. The matrix runner captures the
outermost command output itself, so these logs do not depend on the training
launcher's `LOG_FILE` surviving environment setup. The runner also creates
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
- `no_init_sync` reaches `>0.99`: inspect the NPU initial sleep, layerwise
  reload, weight-name mapping, and post-load processing.
- `low_concurrency` or `batch_invariant` reaches `>0.99`: inspect
  batch-shape-dependent fused kernels or scheduling.
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

## Reduced-layer checkpoints

`reduce_qwen3_omni_layers.py` creates a smaller diagnostic checkpoint by
keeping only the first N Thinker text-decoder layers. It deliberately keeps
the complete audio tower, vision tower, Talker, code2wav, embeddings, and
output head. This makes the resulting model suitable for the same multimodal
actor/rollout path while reducing the part most useful for layer-by-layer and
operator-level comparisons.

First validate the source config and weight index without writing data:

```bash
python3 npu_test/reduce_qwen3_omni_layers.py \
  --source /mnt/share/z00988734/src/weight/Qwen3-Omni-30B-A3B-Instruct \
  --output /mnt/share/z00988734/src/weight/Qwen3-Omni-30B-A3B-Instruct-4L \
  --num-layers 4 \
  --dry-run
```

Then create the checkpoint by removing `--dry-run`:

```bash
python3 npu_test/reduce_qwen3_omni_layers.py \
  --source /mnt/share/z00988734/src/weight/Qwen3-Omni-30B-A3B-Instruct \
  --output /mnt/share/z00988734/src/weight/Qwen3-Omni-30B-A3B-Instruct-4L \
  --num-layers 4
```

Unchanged files and complete shards are hard-linked when the source and output
are on the same filesystem; mixed shards are rewritten and fully removed
shards are omitted. Add `--copy-unchanged` if the output must not share hard
links with the source. The command refuses to overwrite an existing output
directory and builds through a temporary sibling directory, so an interrupted
run cannot leave a checkpoint that looks complete.

To prepare a prefix sweep for locating the first divergent layer:

```bash
for layers in 1 2 4 8 16; do
  python3 npu_test/reduce_qwen3_omni_layers.py \
    --source /mnt/share/z00988734/src/weight/Qwen3-Omni-30B-A3B-Instruct \
    --output "/mnt/share/z00988734/src/weight/Qwen3-Omni-30B-A3B-Instruct-${layers}L" \
    --num-layers "${layers}"
done
```

Every output contains `layer_reduction_manifest.json`, recording the exact
kept layers, key counts, tensor size, and whether each shard was linked,
copied, or rewritten. Run a reduced checkpoint with the existing diagnostic
matrix by overriding only the model path:

```bash
MODEL_PATH=/mnt/share/z00988734/src/weight/Qwen3-Omni-30B-A3B-Instruct-4L \
  bash npu_test/run_nextqa_pearson_matrix.sh baseline
```
