# vLLM-Omni #8039: 单机双节点前端复现探针

更新：该探针未能复现用户遇到的问题。它省略完整引擎，不能替代原始拓扑的复现。
已有两台 A3 时请使用 [真实双机 NExT-QA validation 步骤](repro_8039_two_hosts.md)。

对应 [issue #8039](https://github.com/vllm-project/vllm-omni/issues/8039)。
目标是快速检测不同请求的视频是否在前端处理或 P0/P1 缓存传输中混淆。
这是一个缩小范围的复现探针；尚未在目标 Linux 环境实测，不能保证触发原 issue。

## 环境与启动

使用已能导入 vLLM-Omni 的 Linux GPU/NPU Python 环境。仓库当前依赖固定为
vLLM 0.28.0、vLLM-Omni `ded8934626aaad1a3e816c3a1d9d742efc012d93`；
issue 使用 Ray 2.58.0、Transformers 5.13.1。先用原问题环境运行，再比较修复版本。
脚本不会安装或替换依赖，也不会下载模型权重。

`--model` 可指向现有 Qwen3-Omni-30B-A3B-Instruct 目录，或只包含完整配置、
tokenizer、chat template、音视频 processor 文件的目录。也接受 Hugging Face ID，
此时需要联网下载这些小文件。模型代码的导入仍可能需要平台相关依赖；
“不加载权重”不表示可直接在只有标准 Python 的机器上运行。

在仓库根目录执行（两次使用相同参数，仅改变 Ray 节点数）：

```bash
MODEL=/path/to/Qwen3-Omni-30B-A3B-Instruct
python tests/special_e2e/repro_vllm_omni_8039.py \
  --model "$MODEL" --nodes 1 --replicas 2 \
  --report outputs/8039-one-node.json
python tests/special_e2e/repro_vllm_omni_8039.py \
  --model "$MODEL" --nodes 2 --replicas 2 \
  --report outputs/8039-two-nodes.json
```

脚本通过 `ray.cluster_utils.Cluster` 启动独立的本地 Ray 集群：双节点对应两个
raylet、两个不同的 node ID。用硬性的 `NodeAffinitySchedulingStrategy` 将 P1
接收端固定到相应节点，并校验每轮实际接收节点覆盖全部节点。
两个节点共享物理主机/IP；这里没有独立容器、网络命名空间或跨物理机网络。
不连接现有 Ray 集群，不执行 `ray stop`；退出时只关闭自己创建的集群。

默认 32 个不同视频、每视频 4 帧 112×112、并发 8、3 轮请求。
不实例化模型、不开 TP、不申请 GPU/NPU 资源；张量预处理在 CPU 上完成。
两个节点各声明 2 个 CPU、128 MiB object store，Python/vLLM/processor 进程还需额外内存。
有限内存时可先加 `--requests 8 --concurrency 4`。
NPU 环境保留原有 Ascend 环境变量和已匹配的 vllm-ascend/CANN 配置。

## 测量路径

```text
driver: 不同的 (torch.Tensor frames, metadata) + 显式 16 kHz audio
  -> head 上的 Frontend Ray actor
  -> 真实 AsyncOmniEngine.add_request_async / _build_add_request_message
  -> 真实 StagePool 选副本、UUID scoping
  -> 真实 InputProcessor / OmniInputPreprocessor / Qwen HF processor / P0 cache
  -> Ray RPC（跨 node ID）
  -> 对应 Receiver actor 中的真实 MultiModalReceiverCache（P1）
  -> pixel_values_videos 哈希
```

启动模型和 orchestrator 的部分被省略。StagePool 使用仅提供 `stage_type` 的
client 描述，完成真实的本地 round-robin/binding；实际发送由测试的 Ray RPC 完成。
这是共享前端到多个 P1 的缓存边界测试，不是完整 AsyncOmni.generate、
分布式 coordinator、verl ARStrategy、EngineCore 或模型推理测试。
也不复现原报告中 8 个独立完整引擎、TP=4、NPU collective 的所有行为。

每次运行先关闭 P0 缓存，串行计算每个视频的参考特征，并检查原始输入与参考特征
各自保持唯一。随后分别用串行和并发请求测试启用缓存的前端。每种并发设置重新
创建 P0/P1，轮次之间保留缓存，并轮换视频提交顺序，使同一个视频能够访问不同副本。
所有请求都重新构造 prompt；不同视频使用相同形状和元数据，确保判定依赖内容。
并发使用单个 asyncio loop，符合被测版本中 `add_request_async` 同步执行预处理的语义，
不会额外将 processor 放进线程池来制造不受支持的线程竞争。

JSON 报告包含依赖版本、可获取的安装 commit、node ID、request ID、选定副本、
入口和 `_process_tokens` 的视频哈希、P0 视频特征哈希、P1 视频特征哈希、
multimodal identifier、prompt token 哈希和数据省略次数。
合法的 P0 `data=None` 缓存命中交给真实 P1 解析，不会直接当成串扰。

## 结果与后续定位

| 退出码 | 结果 | 含义 |
| --- | --- | --- |
| 0 | `NOT_REPRODUCED` | 本轮未检测到串扰；不能证明原 issue 已修复 |
| 1 | `MISMATCH` | 输入/特征/prompt 或请求覆盖断言失败，查看 JSON 中具体字段 |
| 2 | `ENVIRONMENT_OR_RUNTIME_ERROR` | 缺依赖、模型配置错误、超时或处理异常；不能当作成功复现 |

若双节点失败而单节点通过，比较同一 sample 的入口、preprocess、P0、P1 哈希以缩小范围。
若只有 `--cache-gb 0` 能通过，优先检查缓存键与副本归属。
P1 cache miss 等运行异常会以退出码 2 保存堆栈，须单独分析，不能等同于视频串扰。

需要更接近原始请求规模时，增加 `--replicas 8 --concurrency 32 --frames 12 --size 224`；
这只增加前端/接收端压力，不会启动 8 份大模型。
如小规模探针不触发问题，应继续在完整引擎上对相同边界插桩，不能据此修改生产缓存逻辑。

本地不具备推理依赖时可验证检测器（不运行 Ray，也不代表复现 issue）：

```bash
python tests/special_e2e/repro_vllm_omni_8039.py --self-test
python -m py_compile tests/special_e2e/repro_vllm_omni_8039.py
```
