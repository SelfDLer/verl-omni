# Joint AV NExT-QA on NPU

These experimental `video-dev-4` recipes preserve the joint audio-video path
from `video-dev-3`. The existing
[`run_qwen3_omni_thinker_gspo_npu_nextqa_v1.sh`](run_qwen3_omni_thinker_gspo_npu_nextqa_v1.sh)
remains the upstream full-depth recipe with independently decoded soundtracks.
See the [main NExT-QA guide](../README.md#training-with-next-qa) for data preparation.

## Environment and gradient synchronization

Use the repository installation guide and dependency pins, then apply the same
recipe-specific verl update required by the upstream NExT-QA launcher:

```bash
python -m pip install --no-deps --force-reinstall \
    "verl @ git+https://github.com/verl-project/verl.git@a0feb78fe8229fde644aec3bbec20b5dc4583509"
```

Restart Ray workers and training processes afterwards. Installing `.[train]`
again may restore the older repository pin; reapply this update afterwards.
The repository pin `fefb080` does not provide the required option.

Both joint AV launchers explicitly set:

```text
actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation=false
```

This disables deferred gradient synchronization during accumulation: gradients
are reduced and sharded after each backward. Gradient accumulation and the
optimizer-step boundary remain unchanged. It lowers gradient memory usage at
the cost of more communication; it does not disable gradient synchronization.
The scripts use the upstream verl option, with no local FSDP override.

## Launchers

- [Full-parameter joint AV](run_qwen3_omni_thinker_gspo_npu_nextqa_joint_av_v1.sh):
  LoRA disabled, vision tower frozen, batch size 32, response limit 1024.
- [LoRA joint AV](run_qwen3_omni_thinker_gspo_lora_nextqa_joint_av_v1_npu.sh):
  rank 32, vision tower frozen, batch size 128, response limit 12288.

These retain the old experimental defaults: remove padding enabled, rollout TP
2, maximum concurrent sequences 128, and 16 NPUs. They differ from the upstream
full-depth recipe in more than the input representation. Match training and
rollout settings explicitly when comparing AV paths; these defaults are not an
isolated processor ablation or a guarantee that the workload fits in NPU memory.

```bash
MODEL_PATH=/models/Qwen3-Omni-30B-A3B-Instruct \
TRAIN_FILE=/datasets/NextQA/train.parquet \
VAL_FILE=/datasets/NextQA/validation.parquet \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu_nextqa_joint_av_v1.sh
```

Use the LoRA launcher above for LoRA training. Both accept additional Hydra
arguments through `"$@"`. `USE_AUDIO_IN_VIDEO=false` selects a visual-only
ablation; it does not switch to the upstream independent-soundtrack dataset.

Both launchers default to TorchCodec and use 8 prompt-filter workers and 8
OMP/TQ/OpenBLAS/MKL threads. Install TorchCodec compatible with your PyTorch
version and verify TorchCodec and FFmpeg on every worker. Joint AV requires
a decodable soundtrack and uses 16 kHz audio.

## Processing and validation

The joint path uses `QwenOmniRLHFDataset`, preserves sampled frame metadata
(including duration), pads audio to the feature extractor hop length, and
restores complete media boundaries before rollout expansion. The actor uses
the upstream `Qwen3OmniVideoProcessor`. Its adapter forwards modality arguments
to Transformers RoPE and retains integer position IDs.

The old global processor timing monkey patch, custom vLLM Thinker/RoPE,
NPU weight-reload override and FSDP no-sync override are removed.
Train/inference parity still needs validation on the target NPU environment;
compare prompt token IDs, position IDs and log-probability correlation.

## NPU environment diagnostics

Run the tracked [environment audit script](../../../scripts/audit_npu_environment.py)
on each training server, from the repository root, using the same Python
environment and CANN setup as the training job:

```bash
mkdir -p outputs/npu_diagnostics
python3 scripts/audit_npu_environment.py > outputs/npu_diagnostics/environment.json
python3 -m pip check > outputs/npu_diagnostics/pip_check.txt 2>&1
```

If the launcher sources CANN inside the script, run these commands after its
`source` lines and before `python3 -m verl_omni.trainer.main_omni`, or run the
same `source` lines in the diagnostic shell first. Keep each server's reports
separate when comparing a multi-node environment.

The audit uses only the Python standard library. It reports installed package
versions, installation commits when available, resolved source paths, source
checkout revisions, selected Ascend environment variables, CANN version files,
and the installed verl agent-loop source fingerprint. It does not import
PyTorch or vLLM, initialize an NPU, install packages, or require locally
downloaded dependency repositories. `pip check` returns a nonzero status when
it finds dependency conflicts; inspect its saved output in that case.

Commit the script with the repository; generated reports belong in the
ignored `outputs/` directory. These reports identify the local interpreter's
environment and do not inspect already-running Ray workers.
