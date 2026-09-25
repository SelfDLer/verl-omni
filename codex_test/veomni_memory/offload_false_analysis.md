# enable_fsdp_offload=false：消失和保留的机制

审计基线：本仓库 `d129b787`，依赖版本、固定提交和服务器文件哈希沿用 [post6 静态诊断](static_diagnosis_post6.md)。只做静态分析；以下假定**仅把 enable_fsdp_offload 改为 false，param_offload=true、optimizer_offload=true 及混合精度等其他配置保持不变**。自动阶段搬运还以 `disable_auto_offload=false` 为条件。

**结论：原生逐层参数 H2D、梯度 D2H 及其完成等待消失；RS 输入的跨流延迟复用链仍成立。阶段性手动 offload 和主动 empty_cache 恢复，训练阶段的参数、优化器状态及分片梯度则留在 NPU。不能把这个开关视为“只去掉 D2H，其他内存行为完全相同”的对照。**

## 配置实际进入哪里

| 文件、函数、行号 | false 分支的行为及条件 |
|---|---|
| 本仓库 `verl_omni/workers/engine/veomni/omni_impl.py::_build_model_optimizer:99–123` | 106 传递开关；113 的条件不成立，因此不执行 115–116 的两个手动 offload 标志清零，也不执行 native 分支的单独 buffers 放置 |
| V `verl/workers/engine/veomni/transformer_impl.py::__init__:181–188` | 保留 `_is_offload_param=true`、`_is_offload_optimizer=true`，`_uses_fsdp2_cpu_offload_policy=false` |
| O `veomni/distributed/torch_parallelize.py::parallelize_model_fsdp2:196–211,360–389` | 不添加 CPUOffloadPolicy；初始化物化到计算设备。mesh、mixed precision、reshard 配置不因本开关改变 |
| V `VeOmniEngine.initialize:262–267`；V `verl/workers/engine/base.py::BaseEngineCtx._context_switch:314–335` | 初始化后按标志卸载；train context 进入时加载参数和优化器，退出时卸载。`disable_auto_offload=true` 会跳过 context 的自动搬运 |
| V `VeOmniEngine.to:493–517`；V `veomni/utils.py:57–121` | 模型走 `model.to(device)` / `model.reshard(); model.cpu(); empty_cache()`；优化器逐 tensor 搬运状态。不是旧版复写 to 的推断，而是当前继承入口的实际调用 |

V / O / T / N 分别指 [固定 verl][V]、[固定 VeOmni][O]、[PyTorch 2.10][T]、[torch-npu post6][N]。下面的路径都相对于这些源码仓库。

## 哪些消失，哪些仍然存在

| 可疑机制 | false 分支裁定 | 代码证据、触发条件 |
|---|---|---|
| native CPU 参数分片及每次 unshard 的参数 H2D | **消失**。阶段加载后，参数分片驻留 NPU | T `torch/distributed/fsdp/_fully_shard/_fsdp_param.py::FSDPParam.__init__:239–243`，`all_gather_inputs:781–786`：`offload_to_cpu=false`，跳过该 H2D 分支 |
| RS 后分片梯度的逐层 D2H、CPU 梯度累积 | **消失**。若有多次梯度累积，改在 NPU 累积；没有据此假设本次存在多轮累积 | T `_fsdp_collectives.py::foreach_reduce:617–642`：跳过 CPU 转移，保留 NPU 梯度 DTensor / 原地累加 |
| pinned gradient D2H 的 grad_offload_event 与 host synchronize | **消失**，指本条 native 梯度路径 | T `foreach_reduce:621–631` 不再创建事件；N `torch_npu/distributed/fsdp/_add_fsdp_patch.py::_patched_finalize_backward:70–73` 无此事件可等。不能扩展为所有 finalize 等待都消失：69、74–83 的通信收尾仍在 |
| RS 输入完整 FP32 拼接缓冲区和 HCCL 跨流使用 | **仍在**，大小及分配流不由 offload 开关决定 | T `foreach_reduce:501,516–548`。输入是整个 FSDP group 的 padded gradient，非单 rank 的分片梯度；详见下一节 |
| RS 输出由梯度持有 | **改变为 NPU 常驻引用**，不能与 RS 输入混为一谈 | T `foreach_reduce:610–642,652–655`。初次梯度赋值持有 RS 输出视图；后续累积保留已有梯度存储，未必保留每次新输出。通常到清梯度才释放；V `EngineTrainModeCtx.__exit__:700–702` 默认先 zero_grad 再阶段卸载 |
| FP32 master、BF16 参数通信/计算、FP32 梯度归约 | **保持**，只翻转本开关不会降低精度或 RS 字节数 | O `torch_parallelize.py:198–205,433–434`；原始 dtype 与 reduce dtype 仍为 FP32。若一个 group 的 padded 梯度共 G 个元素、分片度为 s，RS 输入仍约 4G 字节，输出约 4G/s 字节 |
| AG、重新分片、backward 预取、激活及 workspace | **仍在**。AG 输入准备路径有所变化 | N `_patched_get_param_all_gather_inputs:96–137`：无 CPU offload 且有 param_dtype、无自定义 pre-all-gather 时可走 foreach-copy，仍创建 BF16 平铺临时输入，不是零临时内存 |
| 参数/优化器的 CPU↔NPU 搬运、主动清缓存 | **阶段性手动路径恢复**；不是所有 D2H 都消失 | V `veomni/utils.py::offload_veomni_model_to_cpu:72–75`、`offload_veomni_optimizer:93–101`、`load_veomni_optimizer:113–121`；优化器状态在训练阶段搬入 NPU |
| 分配失败→设备同步→workspace/缓存清理→重试；UnmapMem/FreePhysical | **仍在**，allocator 不读取这个 FSDP 开关 | N `NPUCachingAllocator.cpp::malloc:1261–1275`、`emptyCache:1651–1662`、`release_cached_blocks:2815–2823`、`ExpandableSegment::unmapHandles:594–623` |

驻留参数、优化器状态和分片梯度增加，同时 native 搬运及其延迟减少，净峰值/耗时不能仅靠静态代码判定。这里没有把分片梯度的正常引用存活称为“释放失败”。

## RS 输入延迟复用链逐段核对

以下链针对当前普通 FSDP2 collective 分配路径、独立计算/RS/HCCL 流、post6 默认 `MULTI_STREAM_MEMORY_REUSE=1`，且未开启 HCCL blocking wait。源码默认不等于已确认所有 Ray worker 的环境；改变这些条件需要重新判断，但 `enable_fsdp_offload=false` 本身不改变它们。

1. **在计算流分配输入，随后交给 RS/HCCL。** T `_fsdp_param_group.py::FSDPCommContext.lazy_init:61–78` 创建独立流；T `_fsdp_collectives.py::foreach_reduce:501,521–531` 在当前计算流分配、拼接 FP32 输入，再让 RS 流等待计算流；535–548 在 RS 流上下文发起通信。CPU offload 分支直到 617 才出现，影响的是归约后的梯度输出。
2. **同步形式的 collective 不代表 CPU 已等完通信。** T `DefaultReduceScatter.__call__:116–131` 使用 `async_op=false`；T `distributed_c10d.py::reduce_scatter_tensor:4591–4598` 调用 work.wait；N `ProcessGroupHCCL.cpp::WorkHCCL::synchronizeInternal:1055–1060` 将 HCCL 完成事件的等待提交给当前 RS 流。普通非 blocking 路径允许 CPU 继续提交；1095 起另有 blockingWait 分支，构造默认 false 见 1222。
3. **输入上的 HCCL recordStream 未被擦除。** N `ProcessGroupHCCL.cpp::collective:4202–4235` 记录 HCCL 流；`synchronizeInternal:1069–1074` 尝试 erase。N `NPUCachingAllocator.cpp::eraseStream:3463–3468` 要求当前流与该块的分配流相同：此处是 RS 流与计算流，条件不满足，提前返回。这个保护条件与 CPU offload 无关。
4. **下一组清掉 Python 引用，仍不等于 allocator 可复用。** T `_fsdp_param_group.py::post_backward:591–593` 保存输入和 RS 事件；下一次 533–540 向计算流提交 wait_event 后清引用，再进入下一次 foreach_reduce。设备等待保证后续设备访问顺序，**不阻止 host 先执行下一次内存分配**，也不会自行清掉步骤 3 的 HCCL 标记。
5. **旧输入进入等待事件的 Active 内存。** N `NPUCachingAllocator.cpp::free:1492–1513` 先降低 allocated_bytes；若 stream_uses 尚在，则 insert_events。`insert_events:3047–3065`、`process_events:3093–3125` 等事件完成后才调用 `free_block:2396–2444`，后者将块归还池并降低 active_bytes。因此存在“多个旧输入已无人引用、仍暂不能复用，host 又申请新输入”的条件。

**所以成立的是条件性的积压机制，不是“关 offload 后必然积压”的结论。** 通信足够快时，事件可能在下次申请前已经完成，缓冲区可复用；host 领先于通信、事件尚未完成时，旧输入才会暂时累积。该链不需要 D2H 作为必要条件。

其他边界：N `register/OptionsManager.cpp::GetMultiStreamMemoryReuse:73–95` 的 mode=2 改用 stash，不能套用上述 erase 失败链；N `_add_fsdp_patch.py:286–319` 的增强补丁及 `_use_mem_cache` 可以改变输入分配/持有方式。当前已审计框架入口未启用该通信内存缓存，且关闭 native offload 不会自动启用它。没有建议移除 recordStream 或提前复用未完成通信的存储。

## 对四个现象与后续改动的影响

| 观察 | 关闭 native offload 后是否仍可解释 |
|---|---|
| 各 rank 慢层编号不同 | 可以：RS 等待、各 rank 的本地请求/缓存状态及触发压力的位置仍可不同；无需逐层 D2H |
| 慢层开始恰好达到各自 APP/HBM 峰值 | 可以：未完成事件的输入加上仍存活的模型/梯度等工作集，仍可能使下一次申请触发回收；不能据此断言已占满硬件容量 |
| 慢层中大量 UnmapMem/FreePhysical | 可以，但必须再有物理回收入口。分配重试链仍在；恢复的阶段性 empty_cache 也可释放物理内存，但它的调用位置不能直接解释真正 backward 层内的清理 |
| 随后骤降、缓慢上涨并平稳 | 可以：缓存物理页释放后重新建立工作集。阶段卸载也能导致下降，必须与 backward 层内回收分开；趋稳不是源码保证 |

因此，如果 false 分支仍出现同样现象，**可以排除“原生梯度 D2H 是必要条件”的解释，但不能据此证明 RS 输入是唯一积压对象**。如果现象改善，也不能直接排除 RS 输入：该开关同时改变驻留工作集、搬运、host 提交速度和清缓存位置。

这项核对保留了“在下一组 RS 输入分配前限制未完成通信的 host 提交”作为候选单点方向；它针对两条分支共有的链，而不是只修 D2H。本次只完成分支审计，尚未实现该改动，也没有 NPU 验证结果。

[V]: https://github.com/verl-project/verl/tree/fefb080262e1c015a0ea05f958822a6a512dc795/verl
[O]: https://github.com/ByteDance-Seed/VeOmni/blob/f90b3dc6fbb0ce693745223cc7a94064123dbf4d/veomni/distributed/torch_parallelize.py
[T]: https://github.com/pytorch/pytorch/tree/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/torch/distributed/fsdp/_fully_shard
[N]: https://github.com/Ascend/pytorch/tree/38e483ee79ec8a3286517808740a811c0f0ae66c/torch_npu
