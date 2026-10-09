# #8039：两台 A3 的真实 NExT-QA validation

适用环境：两台物理机，每台 16×64 GB NPU，完整 Qwen3-Omni-30B-A3B-Instruct
权重、NExT-QA parquet/原始视频及可运行环境已准备好。

之前的 `repro_vllm_omni_8039.py` 只测试了共享前端和 P1 cache actor，省略完整引擎，
没有复现用户的原问题。这里直接使用现有 NExT-QA V1 配方，执行真实权重加载、
colocate rollout、AsyncOmni.generate 和模型推理；不使用该探针。
本地没有双机 NPU，本文件及启动包装脚本尚未经过硬件实测。

## 1. 保持版本与输入一致

[原 issue](https://github.com/vllm-project/vllm-omni/issues/8039) 报告的是：
vLLM-Omni `ded8934626aaad1a3e816c3a1d9d742efc012d93`、vLLM 0.28.0、
Ray 2.58.0、Transformers 5.13.1、CANN 9.0。
vllm-ascend 使用 `fd81546` 加报告者的 5 个本地兼容补丁，公开 issue 未给出这些补丁内容。
因此不能宣称只按公开版本号安装就能完全重建报告者环境。
关联 [verl-omni #660](https://github.com/verl-project/verl-omni/issues/660)
还记录了 verl `fefb080`、verl-omni `4633c7a`；当前复现分支不是该历史 checkout。
保留你的可运行环境，先记录实际版本及本地 diff，不要为排障直接升级依赖。

两台机器使用相同代码、依赖、权重、tokenizer/processor 配置和 parquet。
parquet 中引用的原始视频必须在两边相同绝对路径可读，`ffmpeg` 必须在两边 PATH 中。
运行开始前记录 `git rev-parse HEAD`、`git diff`、`python -m pip freeze`，
并比较 `sha256sum "$VAL_FILE"`、模型 `config.json` 和 processor 配置。
源码安装的 vLLM-Omni/vllm-ascend 还要各自记录 commit 和本地修改；版本号不足以表示这些差异。

## 2. 启动真实的双机 Ray 集群

在两台机器相同的 Python/CANN/ATB 环境、仓库目录中，先配置：

```bash
# 示例 IP/网卡；换成实际的训练网地址和当前机器的网卡。
export HEAD_IP=10.0.0.10
export LOCAL_IP=10.0.0.10       # worker 上改为 10.0.0.11
export HCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export RAY_ADDRESS="$HEAD_IP:6379"
```

保留原先已经验证可用的 CANN/HCCL 其他配置。NPU 用 `NPU` 自定义资源注册，
不是 `--num-gpus=16`，参见 [Ray Ascend 文档](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html)
和 [verl 多机指南](https://verl.readthedocs.io/en/latest/ascend_tutorial/zh/model_support/examples/multi-machine_task_startup_practice.html)。

Head 执行一次：

```bash
ray start --head --port=6379 --node-ip-address="$LOCAL_IP" --resources='{"NPU":16}'
```

Worker 执行一次：

```bash
ray start --address="$RAY_ADDRESS" --node-ip-address="$LOCAL_IP" --resources='{"NPU":16}'
```

如已有本实验的旧 Ray 集群，先结束作业，再在对应机器 `ray stop`；不要叠加启动。
不要清理其他作业的 Ray/Python 进程。
环境变量必须在启动 Ray 前配置，否则 driver 的新变量未必传到已经启动的远端进程。

## 3. 只在 Head 启动 validation

```bash
export MODEL_PATH=/models/Qwen3-Omni-30B-A3B-Instruct
export TRAIN_FILE=/datasets/NextQA/train.parquet
export VAL_FILE=/datasets/NextQA/validation.parquet

NNODES=2 AGENT_NUM_WORKERS=8 VAL_MAX_SAMPLES=32 \
  bash tests/special_e2e/run_8039_nextqa_two_hosts.sh
```

包装脚本会拒绝节点数不符、节点 IP 重复或每节点 NPU 资源不是 16 的集群，
显式连接 `RAY_ADDRESS`，避免意外新建本地集群。
它使用 `trainer.val_only=true`，不会执行训练更新；但 V1 trainer 仍会初始化
actor/FSDP 和真实 rollout 引擎，因此并不是轻量 CPU 测试。
底层配方仍需要 train parquet 构建数据集，即使本轮只做验证。

关键设置：每节点 16 卡、TP=4，总计 8 个 rollout 引擎副本，TP 应限制在本节点；
8 个 agent workers、32 个 validation 样本组成一个 batch、每样本生成 1 次。
`max_num_seqs=4` 沿用 NPU 配方。32 个样本不等于测得的瞬时 32 并发，
必须以引擎日志中的 in-flight 请求为准。
载荷使用 `NextQARLHFDataset` 的 presampled `(frames, metadata)` 与独立音频，
`use_audio_in_video=false`、`sampling_rate=16000`。
不在首轮关闭多模态缓存、改 eager、使用随机小模型或缩成只有一个引擎副本。

每次运行输出到独立的 `outputs/debug/8039-nextqa-.../`，包含 `preflight.log`、
`driver.log` 和 `validation/`。启动后核实日志中确实有 8 个 rollout 副本，
每台 4 个，且 TP=4 的设备组不跨节点、同一台机器的不同 rollout 副本不重叠。
保存两台机器本次 `/tmp/ray/session_latest/logs/` 中的 worker 日志。

## 4. 单机对照与加压顺序

依次测 A 单机、B 单机、A+B 双机，保持相同版本、模型、validation 数据与采样参数。
单机对照时，结束双机作业并停止本实验两边的 Ray，再只在目标机器启动 head，
将 `RAY_ADDRESS` 指向它，执行：

```bash
NNODES=1 AGENT_NUM_WORKERS=8 VAL_MAX_SAMPLES=32 \
  bash tests/special_e2e/run_8039_nextqa_two_hosts.sh
```

不能仅在仍连接两台机器的集群里改 `trainer.nnodes=1`，否则未证明所有副本确实落在同一台。
`#8039` 写 agent workers=8，而 `#660` 写 16；先测 8，再用相同配置测试
`AGENT_NUM_WORKERS=16`。两种设置应分别与单机结果比较。

先用 32 条定位明显异常，未复现时重新启动独立 validation 作业重复 3 次。
需稳定比较准确率时，三种拓扑都改 `VAL_MAX_SAMPLES=256`，使用完全相同的数据子集。
不要期待 32 条的准确率恰好等于 issue 的 0.73 或 0.41；样本数量和随机采样都会影响它。

脚本允许追加 Hydra override。例如单独验证是否仅在关闭多模态缓存后恢复：

```bash
NNODES=2 bash tests/special_e2e/run_8039_nextqa_two_hosts.sh \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.mm_processor_cache_gb=0
```

这是后续定位对照，不应与首轮拓扑变化同时引入。不要混淆 prefix cache 与多模态 processor cache。

## 5. 判定复现需要什么证据

准确率明显下降、只引用音频的回答、`SampledLogprobContractError` 是症状；
单独出现 NaN、OOM、连接失败或低准确率，不能证明发生了跨请求视频串扰。
确认串扰要以同一个 request ID 对齐以下数据：

1. server.generate 入口和 ARStrategy 准备完成的 prompt：原始 frames 的全量 SHA256、shape、metadata。
2. 提交给 AsyncOmni 的 prompt 和 `_build_add_request_message`：同样的原始视频摘要及 UUID。
3. `_process_tokens`：原始视频摘要；processor fresh 输出：`pixel_values_videos` 的全量 SHA256。

每条日志带 hostname、PID、引擎副本和 request ID。对同一请求在不同阶段比较，
不要只比较全局 mean/std，也不要把来自相同视频的多个问题误判为串扰。
如果入口已经相同，先查样本选择/载荷构造；如果入口各异而同一批 fresh 特征坍缩成相同数据，
才接近 issue 描述。正常 `data=None` 缓存命中本身不是失败。

## 6. 开启视频观测点

两台机器使用包含本次观测代码的同一版本、同一路径；确保 Python 导入的是该 checkout
（例如已在各自环境执行 `pip install -e . --no-deps`）。保留原有模型、缓存和并发配置，
在 Head 执行原来的启动命令，加上以下环境变量即可：

```bash
export VERL_OMNI_VIDEO_TRACE_DIR=./tmp/8039-video-trace/run01
export VERL_OMNI_VIDEO_TRACE_MAX_REQUESTS=0
NNODES=2 AGENT_NUM_WORKERS=8 VAL_MAX_SAMPLES=32 \
  bash tests/special_e2e/run_8039_nextqa_two_hosts.sh
```

包装脚本通过 Ray `runtime_env.env_vars` 将这两个变量传到两台机器的新 actor，
可用于已经启动的 Ray 集群；需要重新启动 validation 作业。默认不开启观测。
`MAX_REQUESTS` 默认每进程记录前 32 次入口调用，`0` 为不限量；首次排查用 32 个样本配合 `0`，
避免各进程独立限额造成链路缺失。再次实验请换 `run02` 等目录。

直接运行原始 `run_qwen3_omni_thinker_gspo_npu_nextqa_v1.sh` 时，需手动追加相同的
Ray override（只在 driver 中 `export` 不保证现有集群的远端 actor 收到）：

```bash
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu_nextqa_v1.sh \
  "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VIDEO_TRACE_DIR='/tmp/8039-video-trace/run01'" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VIDEO_TRACE_MAX_REQUESTS='0'"
```

日志写入**每台机器本地**该目录下的 `video-<hostname>-<pid>-<session>.jsonl`，
请收集两台的文件，或改用两台均可写的共享目录。每行包含事件、主机、PID、进程序号 `seq`、
时间、请求 ID 和调用 `trace_id`；server 范围还带 `replica_rank`、`node_rank`。
不要依靠跨主机时间戳排序，优先按请求映射和进程内 `seq` 判断顺序。

| 事件 | 用途 |
| --- | --- |
| `hook.install` | 检查 agent 和五个 frontend hook 的安装状态；`unavailable` 表示版本接口不匹配/缺依赖，不能当作该层没有异常。 |
| `agent.send` → `agent.request_id` | 发送前视频摘要，以及 agent ID 到实际发给 server 的 `engine_request_id` 的映射。 |
| `server.receive` → `strategy.engine_submit` | 接收的视频和完成 adapter 处理、提交给引擎的 prompt。 |
| `engine.build.before/after` | 构建引擎请求前后的视频、metadata 和 UUID；保留外部 `request_id`，另记本层 `engine_request_id`。 |
| `engine.scope_uuids.before/after` | stage/replica UUID scoping 前后，包含 `stage_id`、`replica_id`；只有引擎实际调用该路径才有记录。 |
| `frontend.process_tokens.before/after` | token 预处理看到的原始视频；`frontend.processed` 记录返回的 video kwargs 和 mm hashes。 |
| `frontend.hf_fresh` | `from_hf_inputs` 返回的本次新算 video 特征，包括 `pixel_values_videos` 和 grid。 |
| `frontend.cache_merge.before/after` | 缓存命中 mask、mm hashes、缺失项的新特征，以及合并后的 video kwargs。 |
| `engine.submission` | 送出前的 video feature 数据/标识，另记 `core_request_id`、`external_req_id` 供后续关联。 |

先确认真实请求的 `server.receive`、`engine.build.before`、`frontend.process_tokens.before`
均有记录；仅有 `hook.install=installed` 不证明该运行路径执行过。agent hook 针对当前
verl 的 `LLMServerClient`；其他客户端实现需补对应观测。若缺少 agent 事件，仍可从 server 开始定位，
但不能据此排除上游传输问题。

按 `agent.request_id.engine_request_id == server.receive.request_id` 连接 agent/server，
再以 server 的 `request_id` 连接各层；`trace_id` 只标识本进程内的一次调用，不跨 RPC 传递。
视频数组记录完整字节 SHA256、dtype、shape，metadata 按原结构记录，不保存视频像素和完整 prompt。
`unsupported` 表示该类型没有可比较摘要，需要补适配，不能当成两个视频相同。

判读时找**同一请求第一次发生变化的位置**：

- agent 与 server 的 raw frame hash 不同：变化发生在这两个观测点之间。
- server 与 strategy 不同：查看 adapter 的正常变换，确认是否意外替换视频。
- strategy、engine build、UUID scoping、process_tokens 的 raw hash 应逐层核对；
  如果某请求的视频变成另一请求的内容，变化前后的相邻观测点就是下一步排查范围。
- raw 各异但 fresh 特征一致：结合 metadata、采样方式、grid 判断 processor 是否处理成相同视频。
  原始 frames 与归一化/切块后的 pixels 是不同表示，不能直接比较两者的 hash。
- fresh 与 merged 异常：按 `mm_is_cached` 和 `mm_hashes` 对齐；fresh/missing 列表仅包含未命中项，
  其索引不一定等于完整视频列表索引。`None` 可能是缓存协议正常省略，不表示视频丢失。

这些观测覆盖 frontend 至 engine submission，尚未检查模型 worker 收到的数据。
若上述各层始终正确，下一步应在实际 worker 的缓存接收/视觉编码入口记录同样摘要。
观测会同步读取完整数组；设备 tensor 会复制到 CPU，因此会增加耗时并可能改变竞争时序。
开启观测后没出现异常不等于排除了串扰，也不代表本 CPU 测试复现了 #8039。

## 本地检查

```bash
bash -n tests/special_e2e/run_8039_nextqa_two_hosts.sh
DRY_RUN=1 RAY_ADDRESS=10.0.0.10:6379 MODEL_PATH=/models/qwen3 \
  TRAIN_FILE=/datasets/train.parquet VAL_FILE=/datasets/val.parquet \
  bash tests/special_e2e/run_8039_nextqa_two_hosts.sh
```

dry run 仅打印命令，不连接 Ray、不加载模型，也不能算成功复现。
