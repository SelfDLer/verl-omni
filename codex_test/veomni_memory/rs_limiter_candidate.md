# VeOmni / NPU RS 提交限流候选

这是默认关闭、尚未经 NPU 验证的候选补丁。Windows 已完成控制逻辑测试和静态检查；NPU 对照由用户执行。依据见 [offload=false 分支审计](offload_false_analysis.md)。不改变 mixed precision、CPU offload 配置、梯度缩放、归约操作或集合通信调用顺序。

## 单点行为改动与回退

生产行为位于 `verl_omni/workers/engine/veomni/rs_limiter.py::RSLimiter.wrap`：给**当前 OmniVeOmniEngine 实例**的 FSDP group 包装 `post_backward`，在调用原方法前，对共享 `comm_ctx.reduce_scatter_state.event` 执行一次 host `synchronize()`。无上一事件、未参与本组梯度归约、no_sync 梯度累积阶段不等待。不替换任何全局 FSDP 类、分配器或 collective 方法。

开关是 worker 环境变量 `VEOMNI_LIMIT_RS_INFLIGHT=1`；不设或设为 `0` 完全不安装包装。启动时控制，回退需用 `0` 重启 worker，不支持给现有 worker 改环境变量后热回退。forward-only engine 不安装。

只接受本次审计的 torch 2.10.0、torch-npu 2.10.0.post6、NPU 上普通 1-D FSDP2 / DefaultReduceScatter；校验 `_fsdp_param_group.py`、`_fsdp_collectives.py`、NPU 补丁文件和 verl VeOmni 父类文件的 SHA256。不支持 enhance/cache 补丁、HSDP、自定义归约 hook、compiled autograd。显式开启但条件不符会报错，避免悄悄在另一条实现上运行。

独立开启补丁时，向现有启动命令追加：

```bash
'++ray_kwargs.ray_init.runtime_env.env_vars.VEOMNI_LIMIT_RS_INFLIGHT="1"'
```

此时不需要开启 `VEOMNI_RS_AB_MODE`；后者只是本目录验证入口，默认未启用。

## 安全性和实际约束

1. **等待哪个事件。** 等上一组的 `reduce_scatter_event`，来自固定 PyTorch `_fsdp_collectives.py::foreach_reduce:556`。此前 RS 流提交了 copy-in 的计算流依赖，以及 HCCL work.wait 对通信完成事件的设备侧等待，因此该 RS 事件完成意味着上一组 RS 已完成读取输入。它位于后续输出处理 / native gradient D2H 之前，**不保证 D2H 已完成**。
2. **何时释放引用。** 包装器仅保留事件引用，不跨原方法保留 `ReduceScatterState` 或输入 tensor。等待返回后，仍由原始 `_fsdp_param_group.py::post_backward:533–540` 提交原有 wait_event、清输入引用，再进入 foreach_reduce。没有提前释放或改写这些步骤。等待抛错则不调用原方法。
3. **何时旧块能复用。** 当所有持有者放弃 tensor，分配器按原来的 recordStream / eraseStream / event 逻辑处理。对应事件完成并被 process_events 收割后，块才进入可复用池。本补丁不抹除 HCCL 标记、不手动 free、不调用 empty_cache。即使上一 RS 事件已完成，释放时在 HCCL 流尾部记录的分配器事件仍可能覆盖随后提交的工作；不能宣称 synchronize 返回的一刻旧块已经入池。
4. **实际限制什么。** 对同一 FSDP communication context，下一组 RS 输入申请不再越过上一组未完成的 RS。它限制 CPU 超前提交产生的这部分在途工作，不是“全模型最多一个通信 tensor”的保证，也不承诺全局 Active 的硬上限。AG、预取、D2H、workspace、当前层完整梯度、梯度引用持有的 RS 输出仍各有生命周期。另一 communication context 也有自己的序列。
5. **为何可能减少积压。** 原路径允许 CPU 在 device wait_event 尚未执行时继续分配大输入，旧输入因跨流保护仍不可复用。这里把等待移到下一次输入申请之前，给通信和 allocator 回收事件追上提交速度的机会；若这条链是主要压力来源，预期减少等待复用的旧输入、分配重试和长停顿。不是修复内存正确性，也不是已证明的根因修复。
6. **代价与边界。** 入口包装比原来的精确清引用点更早，当前 group 的 accumulate/reshard 也会推迟，可能延长当前完整参数的存活时间、减少 overlap 或变慢。对后续 AG/HCCL 尾部事件也没有无限强的回收保证。这正是需要开关对照的原因，不能预先承诺吞吐提高。没有加入新的 collective、跨 rank barrier 或设备全局同步到生产补丁。

关闭 `enable_fsdp_offload` 时同一限制仍有效：消失的是 native 参数 H2D / 梯度 D2H，输入的计算流分配和 HCCL 使用关系仍在。两个 offload 分支应各自做开关对照，不能直接跨分支比较，因为工作集和手动 empty_cache 入口不同。

## 固定输入对照

验证入口是 `codex_test/veomni_memory/run_rs_ab.sh`，复用已有真实 Qwen3-Omni AVQA 启动配置：单节点 16 卡、SP=EP=1、micro=2，默认 3 个 update。两个独立的新训练进程先后运行：

- **A / off / capture：** 每个 rank 在 update_actor 内部、训练前保存实际 TensorDict、Python/NumPy/Torch/NPU RNG 状态。包括已产生的 response、old log-prob、advantage、多模态输入和 batch 元数据。A 自己也从同一个保存文件重新载入后训练。
- **B / on / replay：** 每次 update 用对应 rank/update 的固定文件替换新 rollout 送来的 actor 输入并恢复 RNG；重新从相同初始模型开始，随后按相同固定输入序列更新。新 rollout 不作为 B 的 actor 训练输入。

这是 actor 更新对照，不比较 rollout 输出质量或训练收敛。仍运行现有 rollout 流程，不能据此声称两个进程整个 allocator 初态完全一致。每个 rank 校验配置、world size，以及第一个 update 前的本地参数和 buffers 字节哈希；要求优化器为新建、未初始化状态，不允许 resume checkpoint。文件哈希和 backward 次数在汇总阶段再次核对。固定输入序列相同不代表跨 run 强制 bitwise deterministic；汇总检查 loss/grad norm（默认 rtol=1e-5、atol=1e-6），它们是数值哨兵，不等价于验证所有梯度/参数逐元素相同。

在服务器仓库根目录、原训练环境和相同模型/数据环境变量下执行：

```bash
git pull --ff-only
export RS_AB_ROOT=/absolute/server/path/rs_ab_native_true_01
bash codex_test/veomni_memory/run_rs_ab.sh
```

`RS_AB_ROOT` 必须是不存在的新绝对路径。所有 rank 都需要访问；脚本限定单节点，不处理跨节点文件汇集。沿用原环境中的 `ASCEND_HOME_PATH`、`MODEL_PATH`、`TRAIN_FILE`、`VAL_FILE`。如已有启动时额外覆盖，将相同 Hydra 参数追加在上述脚本末尾。验证脚本最终固定步骤数、offload 标志、SP/EP、关闭 profiler 和验证环境变量，避免两臂不一致。

单独检查 native offload=false 时，用另一个新目录：

```bash
RS_AB_ROOT=/absolute/server/path/rs_ab_native_false_01 \
RS_AB_NATIVE_OFFLOAD=false bash codex_test/veomni_memory/run_rs_ab.sh
```

无需为了本补丁同时跑两个 offload 分支；先使用当前要判断的配置。不要把此复现实验生成的 checkpoint 当作训练续跑结果。中断后使用新目录；不覆盖固定样本。固定文件含真实训练内容，全部留服务器，不上传。最终只需查看 `brief.json`（最多 8 KiB），不要求传出固定输入、profile 或整套日志。

## 指标口径与判定

`rs_ab.py::BackwardMetrics` 包装当前 engine 的 forward_step 返回值，只围绕父类原来的 `loss.backward()` 计时；FSDP group 前后只读 memory_stats。当前父类仅对该返回值调用 backward；这是固定源码接口，不是可泛化的 Tensor 替代物。包装在每次 update 后恢复。

| 指标 | 口径 |
|---|---|
| `alloc_retries` | 每次 loss.backward 的 `num_alloc_retries` 结束值减开始值；另有整个 update 的重试增量，防止把停顿移到 backward 之外误判为改善 |
| `active_peak_bytes` / `allocated_peak_bytes` | backward 入口重置 peak 计数后的各自峰值；**两峰之差不是同时刻的 gap**。重置仅限显式验证模式，会影响同进程其他 peak 统计，因此对照关闭 profiler |
| `gap_sampled_max_bytes` | backward 入口/出口及 group 前后，Active.current − Allocated.current 的最大采样值；不是连续时间精确峰值，也不能独占归因于 RS |
| `limiter_wait_ms` / `limiter_wait_calls` | 补丁新增 event.synchronize 的 host 墙钟时间及调用次数；包括已完成事件的调用开销，off 为 0 |
| `backward_host_ms` | 原 loss.backward 调用的 host 墙钟时间，含补丁等待；设备未必已经完成 |
| `backward_device_ms` | 计算流的起止 timing event 跨度，含该流依赖的通信与等待，不是各算子耗时之和；只在 update 完成后读取，不逐 backward 同步 |
| `update_actor_host_ms` / `update_actor_drained_ms` | worker 中实际 train_mini_batch 和 output.cpu 的时间；后者另含末尾设备 drain。包括阶段 offload / optimizer，排除 Ray RPC、装饰器、固定文件 I/O、哈希和采样器安装 |

两臂均在 update 入口和出口做一次全设备同步，以界定计时范围；**生产补丁没有这两个同步**。它们会清除跨 update 的在途工作，所以本对照检验的是 update 内积压，不能证明原运行跨 update 的累积行为。期间不增加每层或每个 backward 的全设备同步，也不增加指标 all-reduce。

保留每个 rank/update 的小 JSON，汇总剔除第一个 warmup update，对 update 耗时取各 rank 最大值再比较；报告 backward 的最坏耗时、最大采样 gap、峰值和重试总量。缺失 rank、输入/初态不匹配、backward 数不一致直接拒绝比较；数值不接近返回非零退出码。

**目标：减少不可复用内存积压、分配重试和长停顿，同时审视 update 总耗时及 overlap 代价。Reserved 或 APP/HBM 不必更低；稳定的可复用缓存可以保留。** gap 下降但 update 变慢不是无条件成功；backward 重试下降而 update 重试未改善也需要保留判断。结果接近时不能凭单次 AB 下结论，可在相同空闲资源条件下重复配对。

## Windows 验证

```powershell
python -m unittest codex_test.veomni_memory.test_rs_limiter_on_cpu -v
python -m compileall -q verl_omni/workers/engine/veomni/rs_limiter.py codex_test/veomni_memory/rs_ab.py codex_test/veomni_memory/compare_rs_ab.py
bash -n codex_test/veomni_memory/run_rs_ab.sh
git diff --check
```

CPU 测试验证等待/引用顺序、失败时不继续提交、实例隔离、默认关闭、采样口径和不完整结果拒绝；它不模拟 HCCL、allocator 或真实模型。测试按用户要求放在 `codex_test`，使用标准库 unittest，不安装 Torch/NPU 来伪造设备验证。

本次 Windows 结果：16 项 unittest 通过；修改文件的 Ruff、format、mypy、Python compileall、Git Bash `-n`、diff whitespace 检查通过。pre-commit 的 docstring、device API、DataProto、test structure 检查通过。完整 hooks 未全绿：全仓库 6 篇旧文档缺日期、`examples/flowgrpo_trainer/minimax_h3/prepare_ref2va_data.py` 缺许可证，另有 Windows 配置生成脚本路径失败及 `sh` 缺失；没有为这些无关问题修改文件。
