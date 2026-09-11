# NExT-QA on Ascend (video2)

This recipe starts from upstream main `c25701c`, using its synchronous omni
trainer and Qwen3-Omni adapters. It trains the Thinker's text attention LoRA
parameters with GSPO. Vision/audio encoders and the base language model remain
frozen by PEFT; Talker and codec are removed by the existing adapter.

The default topology is one node with 16 Ascend devices and rollout TP=2.
Actual NPU training and convergence still require validation on the target
machine. The local CPU checks do not load model weights or execute NPU kernels.

## Environment and data

Follow [the NPU installation guide](../../../docs/start/install_npu.md) at this
revision, including the pinned verl, vLLM-Omni and vLLM-Ascend revisions. Use a
compatible torch/torch-npu pair. Source your installed CANN and ATB setup scripts
before launch. Install the extra media dependencies in that same environment:

```bash
uv pip install -e '.[audio]' torchcodec tensorboard
# Install ffmpeg and ffprobe using your system package manager.
```

Choose a torchcodec build compatible with the installed PyTorch version, or
use a working torchvision video backend. Ensure ffmpeg/ffprobe and the selected
decoder are available on all Ray nodes.

Keep the official dataset layout:

```text
/data/NextQA/
  repo/train.csv
  repo/val.csv
  repo/map_vid_vidorID.json
  NExTVideo/<VidOR directories>/<video>.mp4
```

The data converter is unchanged from `video-dev-3`:

```bash
python examples/gspo_trainer/data_process/nextqa.py \
  --input_dir /data/NextQA --output_dir "$HOME/data/nextqa"
```

It writes `train.parquet` and `validation.parquet`, maps answer indices 0–4 to
A–E, and drops missing/invalid media and videos without a decodable audio track.
Its output statistics describe the resulting filtered dataset, so validation
accuracy is measured on that retained subset. Absolute video paths must be
mounted identically on every worker.

## Input contract

`NextQARLHFDataset` expands each video into a video item followed by an explicit
audio item referencing its soundtrack. Both reach the model. Audio is resampled
to 16 kHz and padded to a 160-sample boundary on the actor side, matching rollout.
This recipe uses **separate video and audio segments**, not the model's joint
audio/video interleaving mode; set `use_audio_in_video=false` as in the launcher.
It does not promise the same time-aligned joint representation as interleaving.

Sampled frame indices and source timing travel with the decoded video to both
backends. The actor processor computes FPS from sampled frames / duration, and
passes seconds per temporal grid into RoPE. The pinned vLLM-Omni implements the
same metadata rule. NExT-QA has one video per request; multiple videos with
different sampled FPS are rejected explicitly by the actor processor.

## Run and validate

First run two updates plus validation and a checkpoint using real data:

```bash
export MODEL_PATH=/models/Qwen3-Omni-30B-A3B-Instruct
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_nextqa_video2_npu.sh \
  data.train_max_samples=32 data.val_max_samples=16 \
  trainer.total_training_steps=2 trainer.test_freq=1 trainer.save_freq=1 \
  trainer.experiment_name=video2_smoke
```

Then run the full recipe:

```bash
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_nextqa_video2_npu.sh
```

Override `MODEL_PATH`, `TRAIN_FILE`, `VAL_FILE`, `N_GPUS_PER_NODE`, `NNODES`, and
`ROLLOUT_TP` through environment variables. For multiple nodes, start a Ray
cluster first as described in the NPU guide, and launch on the head node.
All final CLI arguments are forwarded to Hydra, including `--cfg job` to
inspect the composed configuration without training.

The starting settings use eager rollout, padded SDPA, microbatch size 1,
8,192 prompt tokens, 1,024 response tokens, four samples per GRPO group, and
greedy validation. This avoids packing independent examples into an unmasked
SDPA sequence. These are correctness-oriented defaults, not tuned throughput
or convergence settings. Check prompt filtering, truncated responses,
`<answer>A</answer>` formatting, finite losses, rewards, and checkpoint resume
on the NPU smoke run before increasing length or concurrency.

The existing `choice_reward.py` gives 1 for an exact tagged answer match and 0
otherwise. Responses use the converter's unchanged reasoning/answer prompt.

CPU regression checks:

```bash
python -m pytest -q tests/utils/test_nextqa_data_process_on_cpu.py \
  tests/utils/test_nextqa_video_inputs_on_cpu.py
```
