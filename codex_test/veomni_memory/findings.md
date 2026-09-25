# 诊断进展

## 2026-09-25 / 首轮 / LOCAL-d71ae4c

证据：当前工作区、固定版本公开依赖源码、用户人工历史摘录。**没有服务器新回传；没有 NPU 实测。** 所有本次采样配置推导均受实际环境/最终配置核对约束。

### 已证实（静态）

- native offload 分支关闭手动 param/optimizer 标志；自动 context 消费标志。没有本地 `to()` override；父类显式 `.to()` 仍是无条件手动入口。
- 在线 GSPO 走 V1 `omni_sync` → V 的 ActorRolloutRefWorker/TrainingWorker → 本地注册的 OmniVeOmniEngine。
- 默认 world16、SP1、全局 PPO mini16 × rollout.n2、micro2、dynamic=false 时，每 rank 每 mini 只有一次 backward；需要 worker 内实际计数确认。
- `all_ranks=true` 优先于 `ranks=[0]`，应收集全部训练 rank。`analysis=false` 可以留下尚未解析的原始采集。
- 分组包括文本 decoder、视觉块、音频块及 root；冻结组不等于产生梯度的通信组。
- PyTorch 参考实现有前组 RS event 依赖和 backward 预取；first-grad pinned D2H 可异步，finalize 等待 offload event；已有 CPU grad 才阻塞拷贝再累加。
- Ascend 参考实现的 HCCL stream 保护与 allocator pending-free 可以解释 Allocated/Active 分离的机制，但 post6 实装路径尚未核实。

### 已排除的推理（不是实测排除全部故障）

- “无本地 to override 必然有整模搬运 bug”。当前自动路径有实际标志保护。
- “本次固定两 micro，所以第二轮 CPU add 必然发生”。当前基线推导相反。
- “FSDP没有record_stream，所以HCCL不会延迟回收”；“普通free等于device synchronize”。两个断言都不成立。
- “2434065.5KB 就是 RS 输入”“Reserved 高就是泄漏”“尾部标量 AllReduce 解释全部 backward”。均没有身份/因果证据。

### 未确定

服务器 torch 版本、post6 build/补丁、实际方法绑定、FSDP memory cache 是否启用；真实组/字节数与 micro 数；profile run 归属、schema、rank 映射、时钟校准、通信组全局序号；首个偏斜发生在计算/拷贝/主机推进、通信等待还是 allocator 回收。历史数字不默认属于同 run。

### 已交付能力

环境小报告；SQLite/CSV/Chrome trace 的只读/流式解析及 schema inventory；严格时钟/序号匹配门槛、完整成员校验、Top-K 和轮换统计；CPU empty 时间窗与短调用反证；显式内存单位/allocator下的差值、Active下降及free动作配对。未知 DB 的 ID 关联和嵌套 communication JSON 暂为明确能力缺口，等待 schema 后补适配。

真实 worker 的默认关闭 probe 已接入。Ray 环境通过脚本显式传播；每 rank 的 installed/train_batch 回执验证尚未在服务器执行。不能宣称 probe 已在 NPU 验证。

### 下一次回传后只推进这些事项

1. 用 `environment.json` 核对版本、路径、pin 和实际 FSDP 绑定，必要时改正 T/N 参考机制前提。
2. 用 `summary.json` schema/文件清单补具体 DB/JSON 适配，优先复用已经解析的 SQLite；如果仅 raw，选一个采集目录用实装 profiler 解析。
3. 确认同 run/同钟/同组后找第一个高价值窗口，同时检验 audit 中三个候选的反证；仍缺实际 worker batch/storage 才补采。

### 本地验证记录

Windows Python 3.13.5（仅诊断工具测试）：`python -m unittest discover -s codex_test/veomni_memory -p test_summary_on_cpu.py -v` **16 项通过**，包括内存单位、free 配对、不同表时钟隔离和 DTensor 本地 storage 别名去重。已执行 Python 语法编译、三个 CLI 的 help、环境采集器的 Windows 缺包容错 smoke、37 个来源文件 SHA256 校验、Git Bash `-n`、LF 检查和 `git diff --check`。没有执行 NPU 基线。系统 `python` 别名不可用，实际使用 `C:/Users/leovzhang/miniconda3/python.exe`。初次测试暴露 Windows tempfile 权限和 SQLite 连接未显式关闭的问题，已修正为工作区测试目录及明确 close。未安装新依赖；本机没有 ruff。

### 服务器同步前验证

用户随后要求提交并推送到 `origin/qwen3-omni-veomni` 以便服务器拉取。已通过现有 pre-commit 缓存运行 Ruff、格式检查和 mypy；修正了诊断脚本格式并补齐许可证头，16 项合成测试再次通过。诊断文件许可证检查及 diff 检查通过。

完整 pre-commit 已尝试：docstring、device API、DataProto 和测试结构检查通过；全仓文档检查因已有 6 篇文档缺 Last updated 失败，许可证检查因已有 `examples/flowgrpo_trainer/minimax_h3/prepare_fl2va_data.py` 失败；配置生成和全仓 shell 检查受 Windows/WSL 路径及 sh 不可用限制。没有修改这些无关文件，也没有声称完整检查全绿。此次仅同步诊断代码与默认关闭的接入，不创建 PR，不上传结果或源码下载缓存。
