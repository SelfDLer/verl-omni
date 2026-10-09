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

`data.val_batch_size=1` 不能单独作为“只降低并发”的对照。本地参考版 verl 的
`GlobalRequestLoadBalancer.acquire_server` 在 `full_determinism=false` 且负载相同时
选择 `candidates[0]`；若每次新请求到达前上一请求已释放，串行请求会集中到首个副本。
实际运行需核对所安装 verl 的实现，并用新版 `server.receive` 的 `host`、`replica_rank`
和 `request_id` 统计路由；`AGENT_NUM_WORKERS=8` 不保证 8 个推理副本均收到请求。
即使串行时仍异常，也不能排除跨请求缓存或状态残留。此时先检查已有
`validation/0.jsonl` 的 `output`、`gts`、`score`：当前 choice reward 精确比较首个
`<answer>...</answer>` 的内容，格式不符也会得零分。结合实际路由区分副本差异、
输出格式异常与视频内容变化，再选择下一轮对照。文件按随机 uid 排序，行号不是请求时序。

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

## 6. 新版观测：先验证低干扰边界，再开启内容摘要

旧版全局 monkey patch、generate 装饰器、同步文件写入及自动设备到 CPU 拷贝已移除。
新版并未在真实双机 NPU 上验证，因此先确认开启 `boundary + metadata` 后仍能完成验证，
且空输出数量与关闭观测时相符，再开启下一层。此前旧版开启后报错的日志不能直接当作原问题证据。

两台机器同步代码并重启本次 validation 作业。保持模型、缓存、采样及并发设置不变，在 Head 执行：

```bash
export VERL_OMNI_VIDEO_TRACE_DIR=./tmp/8039-video-trace/v2-boundary
export VERL_OMNI_VIDEO_TRACE_STAGE=boundary
export VERL_OMNI_VIDEO_TRACE_MODE=metadata
export VERL_OMNI_VIDEO_TRACE_MAX_REQUESTS=32
NNODES=2 AGENT_NUM_WORKERS=8 VAL_MAX_SAMPLES=32 \
  bash tests/special_e2e/run_8039_nextqa_two_hosts.sh
```

包装脚本将变量传到远端新 actor，并显式选择 `video_trace_single_turn_agent`。
这个诊断类继承原来的 `single_turn_agent`，仅代理本实例的客户端调用和记录合并边界，
不修改上游客户端类。它用于本配方的 thinker 单轮路径，不替代 Talker 或其他自定义 agent。
原始配方脚本直接启动时，需额外传入相同的 Ray env overrides 和
`actor_rollout_ref.rollout.agent.default_agent_loop=video_trace_single_turn_agent`。
只 export driver 环境变量不保证远端 actor 收到。两端必须使用新版 server，
其新增的可选 `video_trace_context` 参数仅用于观测关联，不传给引擎。

| 阶段/模式 | 行为 |
| --- | --- |
| `STAGE=boundary`（默认） | 显式记录 agent/server/strategy 输入和输出，不包装 vLLM 内部方法。 |
| `STAGE=frontend` | 在真实引擎初始化后，仅包装当前 engine/input processor/preprocessor 实例；不修改它们的类、HF 转换器或全局缓存类。 |
| `MODE=metadata`（默认） | 视频/特征只读取 shape、dtype、device、stride 等；文本及 CPU token 快照另行记录。不能用此模式证明视频内容相同。 |
| `MODE=sample` | 仅从 CPU 数组按逻辑 C 顺序等间隔取至多 256 个元素，后台计算 `sample_sha256`。不同可证明采样内容不同，相同不能证明全部内容相同。 |
| `MODE=full` | 仅复制 CPU 数组的完整字节，在后台计算 `sha256`。默认单数组上限 8 MiB，超过时记录 `skipped_byte_limit`，可通过 `VERL_OMNI_VIDEO_TRACE_MAX_BYTES` 调整。 |

所有模式都不读取 NPU/GPU 内容，不调用 `.cpu()`、设备同步或设备归约。
设备数组只记录 metadata；需要内容摘要时标记 `skipped_non_cpu`。
`full` 仍有调用线程上的 CPU 拷贝成本，仅在需要确认具体少量样本时使用。
`capture_ns` 记录每次捕获开销，不能代表观测对整个并发调度的总影响。

格式观测在上述三种模式中都启用：只复制已有 Python 整数列表，不调用 tokenizer、processor
或设备计算。`VERL_OMNI_VIDEO_TRACE_MAX_TOKENS` 默认 32768，限制每个 token 快照；
超限记录 `skipped_token_limit`，设为 0 则关闭 token 内容快照。完整快照记录小端 int64 字节的
SHA256、token 数及全部 IDs，供离线使用同一模型的 tokenizer 解码。文本最多检查前 65536
字符，记录摘要和首尾各 512 字符；超限明确标记 `complete=false`，不能据此判定没有标签。
token 和文本快照也计入后台队列的内存预算。运行中的观测不执行解码。

建议逐步运行，每轮换目录：

1. `boundary + metadata`：先定位空响应在哪一段产生，并与关闭观测的运行比较。
2. `boundary + sample`：比较 agent → server → strategy 的原始视频采样摘要。
3. `frontend + sample`：细分 strategy → engine build → input processor → token preprocessor。
4. 仅在疑似变化区间对少量请求使用 `full` 确认；不要把原始 frames 与处理后 pixels 的 hash 直接比较。

`MAX_REQUESTS` 默认每进程前 32 个根调用，`0` 表示不限。诊断 agent 将是否采样的决定一并传给
server，避免同一请求在不同进程被独立限额截断。应保留相同采样参数，不因诊断改变模型输入。

日志名为 `video-v2-<hostname>-<pid>-<随机ID>.jsonl`。推荐先写两台机器各自的本地绝对路径，
结束后合并收集；共享目录也支持，各进程使用独立文件。所有文件操作及 SHA256 计算由后台线程完成。
队列最多 128 条记录、64 MiB 待处理指纹字节；写盘慢或队列满时丢弃观测，不等待写盘。
普通记录中的 `dropped_events` 和 `trace.health` 会报告丢弃数量；
写入失败由后台输出 `Video trace writer failed`。有丢弃、写入错误或强制终止时日志不完整，
不能根据缺失事件推断调用未发生。正常退出最多等待一秒排空，不保证硬杀进程时尾部日志完整。

| 事件 | 观测位置 |
| --- | --- |
| `agent.begin` / `agent.output` | 数据集 uid、session、sample index；最终 response_ids 和 response_mask 长度。 |
| `agent.source.message` / `agent.template.message` | 原始消息与实际传入模板构建器的角色、文本摘要和格式标记；不遍历媒体数据。 |
| `agent.template.config` / `agent.prompt` | tokenizer 名称/类型、模板摘要、EOS/PAD ID、长度配置，以及模板构建后真正的 prompt token。 |
| `agent.dispatch` / `agent.result` | 调用真实客户端前的原始视频，以及客户端返回的 token 数/停止原因。 |
| `server.receive` / `server.result` | 实际 engine request ID、replica/node rank、接收视频、返回 token 数。 |
| `strategy.submit` / `strategy.result` | adapter 后的 prompt、实际生成参数（含 stop/EOS/长度），原始 completion token、文本摘要和 finish_reason。 |
| `agent.merge.before/after` | Continuous Token 合并前的 assistant token 数、合并后 mask 长度；结合 agent.output 判断最终截断。 |
| `frontend.build.before/after` | 构建引擎请求前后原始视频。 |
| `frontend.uuids.before/after` | 当前 stage/replica 的 UUID scoping。单 replica 不走此分支时可能没有事件。 |
| `frontend.input.before/after` / `frontend.tokens.before/after` | 引擎 input processor / token preprocessor 入口、返回。 |
| `frontend.*.features` | 返回的 video kwargs/hash/feature 标识；可能包含缓存协议正常省略的 None。 |
| `frontend.*.result` | 方法实际返回的 prompt token；与 `.after` 对入参的观察分开。 |
| `trace.install` | frontend 实例方法的 installed/unavailable；仅 installed 不能证明该请求经过此方法。 |

按 `trace_id` 关联一个 agent 调用，`call_id` 关联其中一次客户端 generate；
它们通过显式 RPC 元数据传递，因此不会受客户端重写 request ID 影响。
同一客户端调用的 resume 请求共享 call_id，server 的 request_id 区分实际引擎请求。
`uid` 可对上 `_run_prompt` 的报错，进程内顺序看 `seq`，不要只靠跨主机时间排序。
`object_id` 只在本进程内有意义，不是内容标识。
`extra_info.problem_id`、video_id 和 qid 可用时也随请求传播，用于跨运行定位同一数据样本。

对于 `rm_scores[-1]` 越界，依次看 `strategy.result.token_ids_count`、
`server.result.token_ids_count`、`agent.result.token_ids_count`、
`agent.merge.after.response_mask_count`、`agent.output.response_mask_count`，
找第一处变为零的位置。埋点不会填充 token、丢弃业务样本或抑制业务异常。

格式异常报告由 `analyze_8039_trace.py` 生成，不读取奖励分数、不重新计分：

```bash
python tests/special_e2e/analyze_8039_trace.py /collected/head-trace /collected/worker-trace \
  --tokenizer "$MODEL_PATH" \
  --validation "$OUTPUT_DIR/validation/0.jsonl" > "$OUTPUT_DIR/format-trace-report.json"
```

脚本只加载本地 tokenizer 文件，不加载模型权重、不连接 Ray。`--validation` 可省略；
提供时按本次运行的 uid/session 关联最终导出文本，不按行号或问题文本猜测对应关系。
报告包含逐请求的阶段摘要、token 一致性、标签变化区间、实际停止原因与每个副本的异常计数：

- `prompt_answer_markers_lost`：源系统消息或前一 prompt 含答案标签标记，后续已解码 prompt 不再包含。
- `engine_missing_answer_tags`：引擎原始 completion 解码后已经没有完整答案标签。
- `answer_tags_lost`：输出链路中上一已观测阶段有标签，后续阶段没有，报告具体区间。
- `empty_engine_tokens` / `engine_visible_empty`：区分真正的零 token 与移除特殊 token 后的空文本。
- `engine_length_stop`：保留实际 token 数、max_tokens 与引擎的 length 停止原因。
- `special_token_filter_removed_answer_tags` / `validation_text_changed`：区分特殊 token 过滤及最终导出文本变化。

提示词的 token 变化可能来自正常多模态占位展开，不能仅凭 hash 变化判定损坏；
标签标记检查也不等于完整的提示词语义检查，报告保留首尾文本、模板摘要和原始快照供核对。
一次请求有多个 resume 时，分别记录引擎尝试，不把单次部分输出与合并完成的答案直接比较。
没有 tokenizer、快照超限、摘要不符、缺失事件或日志损坏时报告不完整；有丢弃事件时
`coverage_complete=false`。局部收集不能证明其他机器或未采样请求正常。
该报告给出最早可观察到异常的位置，不将“引擎原始输出缺标签”直接定性为视频串扰或模型内部根因。

本版不观察模型 worker 的 P1 缓存和视觉编码器，也不包装 fresh HF/缓存 merge 的全局静态方法。
若 CPU 输入/处理后特征均正确，才需要再针对实际 worker 的接收/视觉编码入口补观测。
当前测试验证观测的隔离、异常传递和写盘阻塞时的行为，不证明已经修复双机回归或复现 #8039。

## 本地检查

```bash
bash -n tests/special_e2e/run_8039_nextqa_two_hosts.sh
DRY_RUN=1 RAY_ADDRESS=10.0.0.10:6379 MODEL_PATH=/models/qwen3 \
  TRAIN_FILE=/datasets/train.parquet VAL_FILE=/datasets/val.parquet \
  bash tests/special_e2e/run_8039_nextqa_two_hosts.sh
```

dry run 仅打印命令，不连接 Ray、不加载模型，也不能算成功复现。
