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
`status=registered` 表示已登记观测上下文，不等于已匹配并采集请求。worker 用 Omni 自带的 `global_request_id` 精确关联，保留实际 EngineCore ID；
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

## 本地检查

```bash
bash -n tests/special_e2e/run_8039_nextqa_two_hosts.sh
DRY_RUN=1 RAY_ADDRESS=10.0.0.10:6379 MODEL_PATH=/models/qwen3 \
  TRAIN_FILE=/datasets/train.parquet VAL_FILE=/datasets/val.parquet \
  bash tests/special_e2e/run_8039_nextqa_two_hosts.sh
```

dry run 仅打印命令，不连接 Ray、不加载模型，也不能算成功复现。
