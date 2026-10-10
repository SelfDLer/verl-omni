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

默认所有模式都不读取 NPU/GPU 内容，不调用 `.cpu()`、设备同步或设备归约。
第 8 节的 `worker + DEVICE_SAMPLE=1` 是需要显式开启的例外。
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

## 7. 对齐单独一道题在单机和双机中的流动

保留能复现问题的完整 validation 负载、batch、并发和生成参数，仅筛选需要记录的观测。
不要为了追踪一道题把验证集缩成一道题，这会改变路由和并发条件。

从已有报告的 `requests[].sample_key` 找到目标题目，它对应 parquet 中的
`extra_info.problem_id`（通常是 `video_id_qid`），不是随机 uid/request_id。
单机与双机均使用相同数据、抽样 seed 和过滤配置，分别设置：

```bash
export VERL_OMNI_VIDEO_TRACE_SAMPLE_KEY='实际的problem_id'
export VERL_OMNI_VIDEO_TRACE_STAGE=frontend
export VERL_OMNI_VIDEO_TRACE_MODE=sample
export VERL_OMNI_VIDEO_TRACE_DIR="/tmp/8039-request-single-$(date +%Y%m%d-%H%M%S)"
# 单机：原样运行单机对照命令，保留完整验证集。
# 双机：目录改为 /tmp/8039-request-multi-...，原样运行双机复现命令。
```

启动脚本会转发 `SAMPLE_KEY` 到两台机器。指定该变量时，它替代 `MAX_REQUESTS` 的前 N 次选择规则，
因此目标题目即使出现在后面也能被记录；其他题目仍正常生成和计分。
重复出现同一道题会记录多个 trace，不擅自选择其中一次。取消筛选可 `unset VERL_OMNI_VIDEO_TRACE_SAMPLE_KEY`。
若目标题目未被抽到或被过滤，日志里不会有该题，比较工具会报错。

已有 `frontend + sample/full` 的原始日志也可直接比较，无须为了这个工具重新运行。
`boundary + metadata` 旧日志可比较 token，视频内容会明确显示 `unknown`。
结束后收齐两次运行的日志；双机必须包括两台机器的文件。每个目录只放一次运行的数据：

```bash
python tests/special_e2e/compare_8039_request.py \
  --single /collected/single \
  --multi /collected/multi-head /collected/multi-worker \
  --sample-key '实际的problem_id' \
  --tokenizer "$MODEL_PATH" \
  --output-dir ./outputs/debug/8039-request-comparison
```

`--tokenizer` 可省略；仅用于离线解码第一个不同 token 附近的文本，不加载权重、不连接 Ray。
工具生成四个文件：

- `comparison.md`：沿逻辑调用路径排列的阶段表、两边真实路由和日志健康信息。
- `comparison.json`：按 prompt、video、features、sampling 等分别报告第一次观测差异，
  保留前一匹配位置、未能比较的阶段、不同字段和原始文件/行号。
- `single.jsonl` / `multi.jsonl`：该题选定 trace 的原始事件，保留视频指纹、缓存标识和 token IDs。

`equal` 表示该项已记录的值相同；`sample_equal` 仅表示数组采样值相同；
`different` 表示已观测值不同；`unknown` 表示快照缺失、仅 metadata、被截断、缓存省略或存在歧义。
例如 video 的前一匹配位置为 `frontend.input.before`、首次差异为 `frontend.tokens.before`，
则先查这两个位置之间；采样匹配仍可能漏掉未采样像素的更早变化。
工具比较的是两次运行的同一阶段，不会直接比较原始 frames 和处理后的 pixels。
不同进程的 object_id、设备地址、stride 和副本作用域缓存标识不作为内容差异，原始日志仍保留它们。

存在多个同题 trace 时，工具列出候选 ID，需用 `--single-trace-id` / `--multi-trace-id` 显式选择。
一次 trace 有多个 engine resume 请求时，当前工具不猜测它们的对应关系：引擎内部阶段标为 unknown，
仍比较可唯一识别的 agent 输入及最终输出。相同阶段重复出现也不会按文件顺序强行配对。
缺失事件或日志丢弃不能推断为业务未执行；两机时钟不用于确定先后。

首次观测差异不是根因结论。随机采样本身就可能让 response 不同；只有在上游内容已核实一致后，
才能将调查范围继续缩向后续阶段。仅有 boundary/frontend 记录时，尚未覆盖 worker 和模型输入。
第 8 节补充视频/音频特征及 worker 观测；权重和 logits 仍未覆盖。

## 8. 前端一致之后：worker、音频/视频占位与模型输入

针对 `6806999702_8` 这类“前端抽样一致，但双机回答似乎只依据音频”的情况，使用新增 `worker` 模式。
它包含 frontend 观测，并在 worker 启动后仅包装该 runner 实例的方法，不包装模型 forward，不改全局类。
接口按本仓库固定的 vLLM 0.28.0、vLLM-Omni 和 vLLM-Ascend 版本核对；实际安装版本不支持的接口会记录
`trace.install: unavailable`，不能当成该阶段通过。本地 CPU 回归不能替代真实 NPU 验证。

保持两边相同的固定 256 条验证文件和原有 batch/并发设置，只筛选观测对象，不把验证集缩成一道题：

```bash
export VERL_OMNI_VIDEO_TRACE_SAMPLE_KEY=6806999702_8
export VERL_OMNI_VIDEO_TRACE_STAGE=worker
export VERL_OMNI_VIDEO_TRACE_MODE=sample
export VERL_OMNI_VIDEO_TRACE_DEVICE_SAMPLE=0

# 双机：使用原有 MODEL_PATH/TRAIN_FILE/VAL_FILE/RAY_ADDRESS。
VERL_OMNI_VIDEO_TRACE_DIR=/mnt/share/8039-worker-multi \
NNODES=2 VAL_MAX_SAMPLES=256 \
  bash tests/special_e2e/run_8039_nextqa_two_hosts.sh
```

单机用对应的单节点 Ray 集群，`NNODES=1`，目录改成 `/mnt/share/8039-worker-single`；其余实验条件保持一致。
每次用新目录。共享目录可以使用，文件名包含 host/pid/随机标识；非共享目录则收齐两台机器所有进程的文件。
仍使用第 7 节的 `compare_8039_request.py` 命令，无需新的分析工具。

| 事件 | 新增证据 | 不能据此声称什么 |
| --- | --- | --- |
| `frontend.*.features` | `multimodal_features` 中每个视频/音频的索引、占位 offset/length/is_embed、内容及缓存标识 | 前端数据存在不代表 worker 已使用 |
| `worker.receive` | 调度器送到 worker 的恢复后特征、真实 prompt tokens、采样参数、有效 model seed、软件版本及并行配置 | 不直接观测 EngineCore 内部缓存恢复动作；路径/版本一致不代表运行时权重一致 |
| `worker.encoder.write` | 实际写入编码缓存的输出、对应 modality/feature_index | 缓存命中时没有 write 是正常情况；不是视觉编码器内部逐层 dump |
| `worker.encoder.read` | 合并前实际读取的缓存项、是否存在及 embedding | 只记录 cache-present 或 shape 不能证明内容正确 |
| `worker.gather` | 该请求当前 prefill 区间的实际多模态 mask、batch offset | batch offset 不同是正常调度差异；mask 本身不区分音频/视频，要结合 feature_index/占位区间 |
| `worker.model_input` | runner 预处理返回、进入模型执行前的该请求 embedding 与 positions 切片 | 不证明模型注意力实际利用了视频，也不覆盖 logits |

先检查 `trace.worker_registration` 和 `trace.install`。仅对选定请求额外发送一次诊断 RPC，传递标量上下文及
六个诊断设置（DIR/STAGE/MODE/DEVICE_SAMPLE/MAX_BYTES/MAX_TOKENS），不依赖子 worker 是否继承服务进程的环境。
不修改生成参数或请求 ID。RPC 回传每个 worker 的启用状态、实际诊断配置、hook 安装结果和日志文件路径；
`status=registered` 表示已登记观测上下文，不等于已匹配并采集请求。登记发生在实际 engine 的 `add_request_async`
入口，晚于 `AsyncOmni.generate()` 的随机 ID 改写、早于入队。不要在 strategy 调用 generate 前登记原始 ID：
`A` 会先变成 `A-<omni后缀>`，InputProcessor 还可能继续追加后缀；worker 的 `global_request_id` 对应前者，
并非原始 `A`。回执中的 `registered_request_id` 显示真实登记值。
worker 用 Omni 自带的 `global_request_id` 精确关联，保留实际 EngineCore ID；
找不到该字段时不会猜测 UUID 后缀。`trace.worker_unmatched`、注册失败或缺失 worker 事件都表示覆盖不足。
worker 日志仍带同一 sample_key/trace_id，所以比较器会一起收集，不按两机墙钟或日志行号配对。

`DEVICE_SAMPLE=0` 时设备 embedding/positions 只有元信息，比较应为 `unknown`。要进一步比较内容，
两边均改成 `VERL_OMNI_VIDEO_TRACE_DEVICE_SAMPLE=1`，其余条件不变再跑一次。
该选项只在选定样本的 worker 观测位置对每个设备张量抽取最多 256 个逻辑元素并复制到 CPU，
**会增加设备读取、同步及诊断 RPC 开销，可能改变调度时序**；不宣称零扰动。
它不复制整幅视频/整个模型，不读取模型权重，不更改 RNG、采样参数、缓存或奖励逻辑。
每请求每 worker 最多记录 32 个 prefill 区间、256 个观测事件；达到上限会记录 `trace.worker_*limit`，
不能将缺少后续事件解释为“没有视频”。后台队列满、摘要跳过和版本接口缺失同样不能当成相等。

比较表新增 worker 阶段。模型输入和 mask 按 worker rank、精确的 chunk_start/chunk_tokens 对齐；
编码缓存方法返回整个特征，因此按 rank/feature_index 对齐，不再要求它们的 prefill chunk 相同。
重复缓存读取仅在已记录内容摘要一致时合并比较，保留全部来源；缓存值变化时不擅自选择其中一次。
模型输入 chunk 切分不同、无法对齐的重复调用或 worker rank 缺失会标 `unknown`，不强行配对。
`worker_runtime` 在 comparison.json 中保留实际 seed 和版本，Markdown 也列出有效 seed；
这些配置差异是线索，不自动认定为内容损坏。特别是 request seed=null 时，replica 的有效 seed 仍可能不同。

判断顺序：恢复后特征是否先变化 → 编码缓存读写是否变化 → mask/占位是否异常 → 最终 embedding/positions 是否变化。
如果这些可观测内容都一致，调查范围才继续移向权重、模型计算和采样；`sample_equal` 仍只是抽样一致。
只在另做的控制实验中同时设置两边 `actor_rollout_ref.rollout.val_kwargs.temperature=0.0`，
保留原始随机采样实验作为对照；不要让排查脚本偷偷改变原实验。

### worker 全部 unknown、sources 为空

这表示比较器没有找到匹配的 worker 事件，不是 `DEVICE_SAMPLE=0` 的正常表现。
关闭设备抽样只会使设备内容未知；如果 `worker.receive` 被正常采集，其 CPU prompt 和采样参数仍可比较。
不要通过把 `unknown` 改成 `equal`，或盲目开启设备抽样来掩盖缺失。

更新比较器后，可以先用原命令重新分析**现有原始日志**，不必先跑模型。
Markdown 会直接显示每一项 unknown 的具体原因和两边的 worker coverage；JSON 的 `worker_coverage` 中还包括：

- `event_counts`：目标 trace 实际匹配的 worker 事件数。
- `worker_file_inventory`：传入文件里是否存在其他 worker 事件或安装记录。
- `unmatched_target_requests`：前端的精确 core_request_id 是否出现在 worker 未关联记录中。
- `worker_acknowledgements`：新版本注册 RPC 返回的实际启用状态与配置。
- `expected_worker_files_not_supplied`：worker 回报的日志文件不在分析输入内；需要检查收集范围或写入错误。

旧版 `trace.worker_registration.status=sent` 只说明 RPC 返回，不能证明 worker 启用了观测。
旧日志没有回执时，新比较器会明确注明，不推断其环境设置。若实际 worker 载荷根本没被采集，离线比较无法补回它；
需要用修正后的注册链路重新采集。若只是遗漏文件，收齐文件后重新比较即可。

`registered` 且五个 hook 都 `installed`，仍不能证明这些 hook 实际执行。
新日志使用绝对文件路径，并在回执记录 `worker_cwd`；旧版相对路径应按对应 worker 的工作目录解释。
`trace.install` 中的 `frontend.worker_admission` 表示登记入口已包装；接口不兼容会标记 unavailable，报告中仍显示 worker 覆盖缺口。
优先检查注册回执中精确的 host/pid/文件名，而不是其他副本的安装记录总数。
`trace.worker_hook_enter` 记录注册后每个 hook 的首次调用；`trace.worker_unmatched` 同时记录非空但不匹配的
global ID 和等待关联的请求 ID（每次注册最多 64 条，超过后记录 limit）。未匹配的候选可能是同批其他请求，
不能仅凭该事件判定目标请求异常，仍需用前端精确 core_request_id 核对。

### 日志过大时提取 worker 摘要

对 compare 已生成的输出目录执行（目录内应有 `comparison.json`、`single.jsonl`、`multi.jsonl`）：

```bash
python tests/special_e2e/summarize_8039_worker.py /path/to/compare-output
```

只需 Python 标准库，无需模型、设备或重新采集。脚本打印摘要，并写入同目录的 `worker_summary.txt`，
最多 64 KiB；超过上限会显式标记 `TRUNCATED`。它提取实际 feature 模态、缓存读写摘要、rank 间摘要分组、
运行配置和跨运行的具体差异路径，不导出视频、完整 token 列表或回答正文。

`D1`、`D2` 等标签在两次运行间共用，代表 shape、dtype、摘要方案及完整哈希的组合；相同标签只表示
对应的完整/抽样字节摘要相同。`write_vs_read` 检查同一次运行、同一 rank、同一特征的缓存观测；
缺少事件或摘要、缓存标识不一致、摘要方案不兼容时标 `unknown`，重复读写变化时单独标记。
rank 分组是记录值的描述，不预设不同 rank 的张量必须相同。哈希无法计算浮点误差大小或证明准确率下降原因。

`worker.observation.begin/end/error` 等作用域事件没有 `worker_rank`，只计数，不参与数据事件的身份校验。
旧版摘要脚本会因此误报 `worker rank/request identity is missing or ambiguous`；更新脚本后直接重新分析
原 compare 输出即可，无需重跑模型。实际数据事件缺少身份或同一 rank 出现多个请求时，仍会报错并列出事件、行号和具体字段。

### 一轮实验采集视觉编码的多阶段证据

当差异已到达 `worker.encoder.write`，使用以下观测配置运行原来的单机、双机命令。
保持原有数据、采样和并发设置；两轮使用不同的 TRACE_DIR，继续观察同一 sample key：

```bash
export VERL_OMNI_VIDEO_TRACE_SAMPLE_KEY=6806999702_8
export VERL_OMNI_VIDEO_TRACE_STAGE=worker
export VERL_OMNI_VIDEO_TRACE_MODE=sample
export VERL_OMNI_VIDEO_TRACE_DEVICE_SAMPLE=1
export VERL_OMNI_VIDEO_TRACE_VISION=1
export VERL_OMNI_VIDEO_TRACE_NUMERIC_SAMPLES=1024
export VERL_OMNI_VIDEO_TRACE_MAX_BYTES=67108864

# 沿用已验证的 Ray、模型、数据和其他 Hydra 参数。
VERL_OMNI_VIDEO_TRACE_DIR=/path/to/trace-single NNODES=1 \
  bash tests/special_e2e/run_8039_nextqa_two_hosts.sh
VERL_OMNI_VIDEO_TRACE_DIR=/path/to/trace-multi NNODES=2 \
  bash tests/special_e2e/run_8039_nextqa_two_hosts.sh
```

这组开关只改变观测，不过滤实际推理样本，不重放请求，不改变采样、模型权重、计算精度或奖励。
启动脚本检查关键开关，避免开启 VISION 却没有开启 worker/device 采集。**设备读取会同步并改变时序**；
不能宣称无扰动，也不能用 CPU 测试代替真实 Ascend 验证。

新增证据覆盖如下路径：

| 位置 | 记录内容 |
| --- | --- |
| `source` | 调度器实际交给编码器的目标视频数据、完整 CPU 哈希（在字节上限内）和数值抽样 |
| `video.input` / `visual.input` | 组批后的目标视频切片、实际 grid、同批成员和偏移、转换 dtype 前后输入 |
| patch / position | patch embedding 和位置插值输出 |
| 每个视觉 block | block 输入/输出、norm1/norm2、attention、MLP；接口存在时还记录 QKV/proj、attention kernel 的 Q/K/V 与输出、MLP 两个线性层 |
| attention 元信息 | 各层实际 rotary 输入、cu_seqlens、sequence_lengths、max_seqlen；批次边界作为上下文保留，不冒充单样本相等 |
| merger / video output | 主 merger、各 DeepStack merger、拼接后输出、按视频切分后的输出 |
| encoder cache | 原有写入/读取观测及按运行时配置拆分的主视觉/DeepStack 分量 |
| merge / DeepStack | 实际 merge token IDs 与期望 token IDs、mask、视觉 token 位置的最终主 embedding、DeepStack 设置和取用 |
| 实际权重和代码 | 当前视觉模块全部参数/缓冲区的元信息及每张量 64 个数值抽样、实际类/方法源码哈希、视觉配置和 worker 版本/并行配置 |

数值以原 dtype 的有界字节样本保存，离线解码支持 BF16，不在模型运行时转换整块 embedding。
普通观测每张量默认 1024 个元素，最多 4096 个；权重每张量 64 个。
CPU 完整哈希受 MAX_BYTES 限制，不会将整块 NPU 张量搬到 CPU；CPU 复制和哈希本身仍有成本。
每次目标请求每 worker 最多 1024 个数据事件，每特征/观测点最多 4 次，每类权重最多 768 个张量、视觉 block 最多 64 层。
写入队列最多 512 项，原始字节预算仍为 64 MiB。超限、缺失接口或未知布局必须视为观测缺口。

目标视频按调度器成员顺序及实际 grid 对齐，不按张量形状猜测请求身份；不支持的重排/分片布局记录 gap。
此外离线检查同次运行内 source → video input → visual input、visual output → 视频切分 → cache 写入/读取 → 主 embedding，
可发现同样 grid 下的内容错配。视频到主 embedding 的检查还要求实际 merge token IDs 完整验证；
输入分块不同、重复记录或 token 抽样不足时保留 unknown。

**无需等实验结束才检查采集是否工作。** 目标样本执行后，可在运行中检查原始日志（不依赖 comparison 文件）：

```bash
python tests/special_e2e/summarize_8039_worker.py \
  --raw /path/to/trace-multi --sample-key 6806999702_8
```

两机非共享目录时可分别执行；共享目录自动递归读取。输出每个 trace/host/pid/rank 的实际观测、权重数量、
缺失阶段和接口失败。目标样本尚未执行完时 missing 可能是暂时的；只有 installed 不代表采集成功。
如果模型走预编译/图重放路径而跳过 Python 方法，缺失层会显示出来；脚本不自动关闭编译或改变原实验。

两轮结束后，按原命令运行 `compare_8039_request.py`，再运行 `summarize_8039_worker.py /path/to/compare-output`。
`comparison.json` 新增 `vision_stages`、`vision_coverage` 和 `vision_flow`。Markdown 和小摘要按 block 汇总，
展开第一个出现差异的 block；完整逐层、逐 rank 明细保留在 JSON。
数值指标包括不同元素数、最大/平均绝对误差、相对 L2、cosine、NaN/Inf 和绝对误差超过 0.001/0.01/0.1 的计数。
这些阈值只描述误差，不是正确性判据。哈希不同仍标 different，微小浮点差异不会被自动判为数据串扰；
抽样一致也不能证明完整权重/张量相等。尚未覆盖语言模型内部层、最终 logits 和 NPU kernel 内部运算。

### ACLGraphWrapper 下安装成功但视觉内部检查点缺失

如果 manifest 的 `video`、`deepstack.set/consume` 显示安装对象是 `ACLGraphWrapper`，
同时只有 source/cache/merge 检查点和 `visual grid differs from video batch`，
应先检查埋点是否误装到属性转发包装器。包装器上的方法赋值不会拦截其内部模型的方法调用；
埋点没有建立 video 上下文时，即使模型 grid 正常，也会触发该 gap。
修复后的安装逻辑沿方法的 `__self__` 找到真实模型对象，并拒绝把转发方法覆盖在错误对象上。
包装器及其图执行策略保持原样；真正的图重放若绕过 Python，仍会报告缺失而非宣称采集成功。

旧 multi 的权重样本、source、cache 各分支和 merge 观测仍可用于比较。
可以先用修复版采 single，与旧 multi 比较共有检查点；旧 multi 缺失的内部层会继续是 unknown，
无法通过离线脚本补出。如果共有证据不能定位，再决定是否补采 multi 内部层。
比较前保持样本、模型、采样参数、设备采样数量等原实验设置一致。

### 权重比较显示 no_array_snapshot

这表示没有采集到可比较的数组，不表示权重相等。旧版埋点按类的模块名识别 Torch 张量，
会漏掉定义在 `vllm.model_executor.parameter` 等模块里的参数子类。
修复版按 `torch.Tensor` 的继承关系识别；设备读取仍要求 `DEVICE_SAMPLE=1`，权重仍只采样每张量 64 个元素。
离线比较也保留原始 `unsupported` 类型说明，避免统一覆盖成 `no_array_snapshot`。

已有日志无法补回漏采的权重值，但已记录的参数差异仍可分析。无需重跑模型，先重新生成摘要：

```bash
python tests/special_e2e/summarize_8039_worker.py /path/to/compare-output
```

目录中应包含原来的 `comparison.json`、`single.jsonl` 和 `multi.jsonl`。
生成的 `worker_summary.txt` 优先展示最多 64 项参数差异，再列最多 8 项 unknown 的原因；
其余数量明确标注，完整明细仍在 `comparison.json`，文件总上限仍为 64 KiB。
若旧 comparison 已丢失具体的 unsupported 类型说明，需要用原始日志重新运行 compare 才能恢复该说明；
这也不需要重新运行模型。

### 已有 fc1 bias 差异：离线检查坐标与 checkpoint

当参数差异集中在 `blocks.*.mlp.linear_fc1.bias` 时，可以继续使用已采集的数值，
无需启动 Ray、加载模型或重新执行单机/双机实验：

```bash
python tests/special_e2e/audit_8039_vision_bias.py /path/to/compare-output \
  --checkpoint /path/to/Qwen3-Omni-30B-A3B-Instruct
```

输入目录需包含同一轮 compare 生成的 `comparison.json`、`single.jsonl`、`multi.jsonl`。
脚本只依赖 Python 标准库，只从本地 safetensors 文件读取头部与指定的一维 bias；不读取视频或大权重矩阵，
不访问 NPU。省略 `--checkpoint` 时仍能分析两次观测之间的差异位置。

输出 `vision_bias_audit.txt`（最多 64 KiB，适合上传）和带完整来源引用的 `vision_bias_audit.json`，包含：

- 每层差异的采样序号 `slots`、分片内下标 `local_indices` 及双方实际数值。
- `pattern`：多少层的差异落在同一组已记录下标；`sampled_suffix` 只表示采样序列的尾部，不能推断未采样元素。
- `single_vs_checkpoint` / `multi_vs_checkpoint`：分别与 checkpoint 比较，避免把单机默认视为正确结果。
- 对齐时的全局下标、checkpoint 中的参数名与文件，以及具体不一致值。

checkpoint 对齐明确假定普通 column-parallel 连续分片：`global = rank * local_length + local_index`，
并要求记录的 PP=DP=1、TP rank 有效、checkpoint 长度等于 `TP * local_length`、dtype 相同。
padding、重排、缺失 runtime、重复请求、摘要缺失或 dtype 转换不明确时保留 unknown，不推断布局。
这里的相等仅指已采样数值相等，不是完整张量或位模式相等。
若已经训练或恢复过 checkpoint，应指定对应版本；与原始 checkpoint 不同本身不能判定加载错误。
这项检查用于定位参数内容及区域，不能单独区分初始加载、actor 同步和 sleep/wake 哪一步出了问题。

### bias 尾部差异与 FSDP2 不等长分片：小规模独立检查

完整 fc1 bias 长度为 4304。16 路分片时每片 269；32 路按向上取整分片时，
前 31 片各 135，最后一片从全局下标 4185 开始，实际长度 119、补齐长度 135。
若旧观测最后一个正常采样点为 4183、第一个异常点为 4200，这与最后一片的边界吻合，
但还不能证明整片损坏，更不能直接断定加载、卸载或聚合中哪一步有问题。

`probe_8039_fsdp_bias.py` 在独立 torchrun 作业中检查此路径。默认读取 27 个真实 bias，
同时构造长度 4096、4320 的整除对照；每个进程的模型参数总量不足 0.3 MiB（BF16）。
通信库、运行时与分配器还会占用额外内存。省略 `--checkpoint` 可直接使用确定性的合成值。
无需启动 Ray、读取视频或实例化 Qwen 模型。使用原实验的 Python/torch-npu/verl 环境；
先让占用这些卡的训练及 rollout 进程退出。

两台机器分别执行相同命令，仅 `NODE_RANK` 不同（第一台为 0，第二台为 1）：

```bash
HEAD_IP=172.27.3.118       # 第一台机器、另一台能访问的 IP，不含端口
NODE_RANK=0              # 第二台改成 1
torchrun --nnodes=2 --nproc_per_node=16 --node_rank="$NODE_RANK" \
  --master_addr="$HEAD_IP" --master_port=29639 \
  tests/special_e2e/probe_8039_fsdp_bias.py \
  --checkpoint /mnt/share/z00988734/src/weight/Qwen3-Omni-30B-A3B-Instruct \
  --output outputs/debug/8039-bias-fsdp32
```

这里 `master_port` 是独立 torchrun 的空闲端口，不是 Ray 端口。沿用原环境实际使用的 HCCL
网络配置，不假定网卡名为 eth0。两边代码、参数、权重路径必须一致；输出可放共享目录，
每 rank 文件独立命名。不共享时摘要和聚合 JSON 写在 rank 0 所在机器，详细日志在各自机器。
**每次调用更换输出目录**；发现同名 rank 日志时脚本拒绝覆盖。

一次运行会自动检查：

- `export_unobserved.result`：加载 → CPU 卸载 → 载回 → 保存 state_dict 引用 → 再卸载 →
  导出 full_tensor，中间不读取诊断张量，避免观测同步掩盖时序问题。
- `load.local/full_tensor`：新建另一份小模型，区分加载后的本地 shard 和聚合结果。
- `initial_offload.local`、`reload.local`：检查 CPU/NPU 往返后的内容。
- `export.offloaded_local/retained_local/retained_full_tensor/fresh_full_tensor`：
  分别检查卸载后的模型、保存的 state_dict 引用及二者的聚合，寻找引用/补齐存储的问题。
- `forward.result/resharded_local`：执行小模型前向，覆盖 FSDP 自身的 padded all-gather 路径。
- `control.dtensor_full_tensor/padded_all_gather`：用已知正确的局部分片，独立对照 DTensor 聚合和显式补齐后的通信。

所有检查都比较完整向量，摘要给出坏元素总数、首尾全局下标和少量实际值。
默认 `--loader verl` 调用当前环境真正安装的 `fsdp2_load_full_state_dict`、CPU 卸载与载回函数，
并记录路径和源码哈希。`--loader torch` 可作加载器对照，但不替代默认路径。
这是默认手动 `param_offload` 路径的缩小实验，不模拟 CPUOffloadPolicy、优化器更新、
完整 actor 层次、权重传输到 vLLM 或 worker sleep/wake。

两边都加 `--fsdp-size 16` 并换输出目录，可以在相同双机 32 进程作业中构建两个 16 卡分片组作对照；
单机对照则改成 `--nnodes=1 --node_rank=0 --nproc_per_node=16`。不需要先重跑完整 NextQA。

首先查看 rank 0 输出目录中的 `bias_probe_summary.txt`（最多 64 KiB），可以直接发送。
`load.local` 就失败时优先调查加载；local 正确而 full_tensor 错时优先调查聚合；
卸载后首次失败时调查迁移；保存引用与新 state_dict 的导出结果不同则调查引用与补齐生命周期。
这些是定位方向，不是自动的根因判定。同步观测会改变时序，因此保留第一段无中间读数的导出对照。
全部通过只表示这个小模型未复现，不能证明实际 actor 和 rollout 权重链路正常。
发生异常时 rank 日志的 `begin/error` 保留已到达位置，不把未完成步骤计为通过。

### 已复现首次 CPU 卸载后损坏：对照拷贝与补齐顺序

若 `load.local`、`load.full_tensor` 均全部正确，第一次 `initial_offload.local` 就出现错误，
并且独立 `control.dtensor_full_tensor`、`control.padded_all_gather` 也全部正确，
则这个小实验已把首次可见损坏定位到 CPU 卸载步骤。`different=864` 表示 32 rank 各自检查的
27 个参数有差异，而不是 864 个不同参数；整除对照的 64 次检查可独立通过。

一个与这种现象吻合的机制是：`model.to("cpu", non_blocking=True)` 返回 CPU 张量后，
FSDP2 的 `_apply` 内部马上调用 `reset_sharded_param`，在 CPU 上为最后的不等长分片重新分配
补齐存储并复制数据。如果 NPU→CPU 拷贝尚未完成，CPU 就会把旧内容复制进新的补齐存储。
在整个 `model.to` 返回后才同步，无法修复已经完成的错误 CPU 拷贝。
这是需要对照确认的机制；初次卸载失败本身还不能单独证明具体内部操作的责任。
相关源码：[FSDPModule._apply](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/fsdp/_fully_shard/_fully_shard.py)、
[FSDPParam.reset_sharded_param](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/fsdp/_fully_shard/_fsdp_param.py)。

先可离线重新生成旧实验的小摘要，无需 torchrun 或重跑：

```bash
python tests/special_e2e/probe_8039_fsdp_bias.py \
  --summarize outputs/debug/8039-bias-fsdp32
```

读取原 `bias_probe_summary.json`，生成 `bias_probe_summary_v2.txt`，原 JSON 保留不变。
新版优先展示本地分片错误，在每个阶段列出 `failing_ranks`；相同的已记录聚合错误只展开代表项，
避免 rank 0–9 的重复结果挤掉 rank 31。去重只基于已记录的有限错误示例，不能证明完整错误向量相同。

随后在两台执行一次以下小作业；沿用原环境，第一台 `NODE_RANK=0`，第二台改为 `1`：

```bash
HEAD_IP=172.27.3.117
NODE_RANK=0
torchrun --nnodes=2 --nproc_per_node=16 --node_rank="$NODE_RANK" \
  --master_addr="$HEAD_IP" --master_port=29639 \
  tests/special_e2e/probe_8039_fsdp_bias.py \
  --checkpoint /mnt/share/z00988734/src/weight/Qwen3-Omni-30B-A3B-Instruct \
  --suite offload --repeats 2 \
  --output outputs/debug/8039-bias-offload-controls
```

每种对照都重新创建并加载模型，避免沿用已经损坏的参数。默认每种重复两次，自动执行五组：

| 名称 | 改动位置 |
| --- | --- |
| `original` | 原环境安装的卸载函数 |
| `sync_before` | 调用原卸载函数前同步设备 |
| `sync_after` | 调用原卸载函数后同步设备 |
| `blocking` | 使用 `model.to("cpu", non_blocking=False)`，随后清空设备缓存 |
| `sync_before_repad` | 仅在本测试进程内临时包装实际 FSDP 参数类，在 CPU 不等长分片 reset 前同步；结束即恢复 |

每组先检查没有中间张量读数的导出，再用新模型检查加载、本地 CPU 分片、载回、第二次卸载、
保存引用/新 state_dict 的导出与前向。参数依然只有小 bias 向量，不运行完整模型或 NextQA。
`offload_guard` 原始记录包含补齐前同步的调用次数；不支持当前 FSDP 内部接口、或不等长分片
未覆盖两次卸载时记录 GAP，不能当作修复通过。

如果 `original` 失败，而 `blocking` 和已实际调用的 `sync_before_repad` 均通过，
且前后同步对照仍失败，就会显著加强“异步拷贝未完成时 CPU 提前补齐”的判断。
前后同步也可能改变时序而偶尔通过，需结合两次重复与完整阶段结果分析。
当前修改仅用于诊断，对照通过不等于训练主流程已修复。最终训练修复还需覆盖初始化、权重导出、
训练上下文退出和 checkpoint 等实际调用卸载的入口，并用相同样本验证 worker 参数与输出。
新结果仍发送 `bias_probe_summary.txt` 即可，不需要上传完整 JSON。

### 唤醒 OOM 时的显存时序

设置非空 `VERL_OMNI_VIDEO_TRACE_DIR` 且 `VERL_OMNI_VIDEO_TRACE_STAGE=worker` 时，
启动阶段的 `monkey_patch_model` RPC 会传递诊断配置，并在请求注册前安装 worker 的 sleep/wake 观测。
不依赖目标样本先到达，也不依赖 `VISION=1`。每个 worker 单独写入
`video-memory-<host>-<pid>-<suffix>.jsonl`，共享目录不会共用一个文件。

```bash
python tests/special_e2e/summarize_8039_worker.py \
  --memory /path/to/trace-multi > memory_summary.txt
```

这个命令只需要当前运行的日志，不需要单机结果、样本 ID 或 comparison.json。
若两台机器使用本地目录，应分别执行；共享目录可统一读取。
摘要上限 64 KiB，可在失败后直接发送。原始文件保留每次 sleep/wake、首次设备采样、
选定请求的 preprocess、视觉权重采样和各视觉 block 入口时的显存统计。

- `INCOMPLETE_WAKE` 且 `samples_started=0`：记录覆盖的 worker 生命周期内，尚未执行设备采样；
  结合同一 worker 的 OOM 日志，可排除该进程已执行的采样操作导致这次失败。
  不能据此排除其他进程占用、启动安装行为或安装前的问题。
- `samples_started>0`：唤醒前已执行过采样，重点查看采样后、sleep 后和 wake 前的 free/allocated/reserved 变化。
  这只是时序证据，并不能单独证明采样造成 OOM。
- 没有文件、安装缺失、统计查询失败、运行中的未完成 wake 均不能当成“没有显存问题”。

记录使用已初始化 NPU 的统计查询，不创建设备张量，不同步设备，不清缓存，不重置峰值。
free 是整张卡的空闲量，allocated/reserved 是该进程分配器统计，未必覆盖 CaMem 和通信库全部分配；
峰值也可能包含此前模型初始化。生命周期记录同步写入并关闭文件，关键唤醒记录另写 stderr，
因此 native abort 前的记录不依赖后台队列；这不是断电持久性保证，共享目录写入也会增加延迟。
普通记录最多 4096 条，超过后标记截断，唤醒记录继续保留。
设备采样本身仍会分配索引张量和临时结果；采样数量上限与 `MAX_BYTES` **不是 NPU 显存峰值上限**。

## 本地检查

```bash
bash -n tests/special_e2e/run_8039_nextqa_two_hosts.sh
DRY_RUN=1 RAY_ADDRESS=10.0.0.10:6379 MODEL_PATH=/models/qwen3 \
  TRAIN_FILE=/datasets/train.parquet VAL_FILE=/datasets/val.parquet \
  bash tests/special_e2e/run_8039_nextqa_two_hosts.sh
```

dry run 仅打印命令，不连接 Ray、不加载模型，也不能算成功复现。
