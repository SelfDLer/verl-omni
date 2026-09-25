# VeOmni / FSDP2 / torch-npu post6 静态诊断

审计日期：2026-09-26。代码基线：本仓库 `e811650c`。本报告只分析源码，不运行训练、不修改训练实现、不要求补采。它更新旧 `audit.md` 中关于 NPU 参考版本、对照路径和 root reshard 的适用范围。

**结论：四个现象组合更支持“各 rank 在不同推进位置遇到本地内存压力，某次分配或清缓存操作承担设备同步及批量物理回收”的解释。当前 backward 路径中，分配失败后的全量回收重试是最有代码依据的触发候选；`UnmapMem/FreePhysical` 本身不能把它与显式 `empty_cache` 区分开。** 已定位出可产生延迟复用的具体流条件，以及 VeOmni 和 FSDP2 对照路径的实质差异；尚不能把这些机制升级成已验证的唯一根因。

## 版本、配置和证据边界

| 项目 | 当前依据 |
|---|---|
| 当前仓库 | `e811650c`；工作区另有 SD35 示例脚本修改及未跟踪 `.worktrees/`，不进入本路径，也未修改它们 |
| verl 固定提交 V | `.github/verl_pin.txt`：`fefb080262e1c015a0ea05f958822a6a512dc795` |
| VeOmni 固定提交 O | `.github/veomni_pin.txt`：`f90b3dc6fbb0ce693745223cc7a94064123dbf4d` |
| PyTorch T | 已回传环境：`2.10.0+cpu`，源码提交 `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`；在 NPU 软件栈中不能凭 `+cpu` 后缀判定未使用 NPU |
| torch-npu N | 已回传环境：`2.10.0.post6`，git_version `38e483ee79ec8a3286517808740a811c0f0ae66c`；本报告逐行读取这个提交，不再用旧 `85f239d` 分支快照代替 |
| 服务器 verl | 已回传 checkout 是 `76796f0af54808fe524b7b6029f7ea847244cfe3`，有 trainer 本地修改；但 VeOmni engine 文件 SHA256 与 V 相同。不能把“engine 相同”扩大成“整个服务器调用链相同” |
| 已核对文件 | 环境报告中的 V `veomni/transformer_impl.py`、O `torch_parallelize.py`、T `_fsdp_param_group.py` / `_fsdp_collectives.py` 哈希均与下载固定版本一致；N 补丁文件亦与本次 post6 下载一致 |
| 本次配置 | 沿用现有 `run_baseline.sh:35–51` 的 mini=16、rollout.n=2、micro=2 及三个 offload=true；脚本默认 world=16、SP=1、EP=1、非动态 batch；额外覆盖不凭空假设 |
| FSDP2 对照 | 本仓库 `run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh:45–64` + `OmniFSDPEngine` + V 的 FSDP engine；另列显式开启 `offload_policy=true` 时的行为，避免把不同开关混为一谈 |

新读取的公开源码 URL、commit、SHA256 在 `static_source_manifest.json`；此前固定源码在 `source_manifest.json`。没有将用户环境文件、机器路径或原始 profiling 纳入 Git。

下文 L 表示当前仓库文件（`workers/`、`pipelines/` 均位于 `verl_omni/` 下）；V/O/T/N 表示以上固定提交。主要文件简称在末尾链接到固定源码，其余路径可在两份 manifest 中查到 URL；表中行号是源码行号，不是 profiler 的行号。

## 三个 offload 开关的实际消费链

| 文件、函数、行号 | 触发条件 | 实际行为及结论 |
|---|---|---|
| V `workers/config/actor.py::VeOmniActorConfig.__post_init__:364–367`；V `workers/engine_workers.py` actor 构造 `607–629` | strategy=veomni | `actor.engine = actor.veomni`，TrainingWorker 接收 engine config；并注入 micro-batch 等参数 |
| L `workers/engine/veomni/omni_impl.py::OmniVeOmniEngine._build_model_optimizer:97–112` | 当前 omni VeOmni 注册路径 | 将 `enable_fsdp_offload` 传给 O `build_parallelize_model`，而非仅保留配置字段 |
| O [parallelize_model_fsdp2:196–211][O-parallel] | `enable_fsdp_offload=true` | 设置 `CPUOffloadPolicy()`；T `_fsdp_api.py::CPUOffloadPolicy` 默认 pin_memory=true。这是训练中参数 H2D、梯度 D2H 的原生策略 |
| V `veomni/transformer_impl.py::__init__:181–182` → L `omni_impl.py::_build_model_optimizer:113–124` | `param_offload=true`、`optimizer_offload=true`，随后 native offload=true | 父类先把两个手动标志设 true；本地构造随后把两者设为 false，并标记 `_uses_fsdp2_cpu_offload_policy=True`；只额外把 buffers 放到设备 |
| V [BaseEngineCtx._context_switch:314–328][V-base]；V `veomni/transformer_impl.py::initialize:262–267` | 自动初始化及 train/eval context | 消费改写后的运行标志；不会因为原始配置仍为 true 就整模反复搬运 |
| V [VeOmniEngine.to:493–517][V-veomni] | 显式调用且传 `model=True` / `optimizer=True` | 仍可手动搬运；该方法刻意不受配置保护。本地没有 `to()` override。已追踪的自动 backward 内没有这种调用，不能据“方法存在”认定发生重叠 |
| L `omni_impl.py:128–131` → O `offloading.py::build_activation_offloading_context:73–89` | 仅 `enable_activation_offload=true` | 才创建 saved-tensor offload hooks；当前默认 false，返回两个 nullcontext。不是三个参数 offload 开关中的第四次搬运 |

**静态裁定：当前路径没有“三个 true 导致三套逐层 offload 同时运行”的证据。** 原生 CPUOffloadPolicy 是生效的训练内策略；两个手动标志在实际对象上关闭。不能通过恢复旧 `to()` override 来解释或修复本次层内物理回收。

backward 调用链为 V `BaseEngine.train_batch:113–132` 的 zero_grad → `VeOmniEngine.forward_backward_batch:398–439` 的切 micro / forward / `loss.backward()` → T FSDP hooks `pre_backward:476`、`post_backward:495` → `foreach_reduce:446` → HCCL → 梯度 D2H → N `_patched_finalize_backward:68`；optimizer 在 backward 返回之后。

## 与当前 FSDP2 对照的差异

| 比较项 | VeOmni 当前配置 | FSDP2 对照及触发条件 | 文件、函数、行号 |
|---|---|---|---|
| native offload | `enable_fsdp_offload=true`，CPUOffloadPolicy | FSDP2 `param_offload/optimizer_offload=true` 仅设置手动阶段搬运；还需 **`fsdp_config.offload_policy=true`** 或 forward_only 才启用 CPUOffloadPolicy。V `engine/fsdp.yaml:19` 默认 false | L `omni_impl.py:105,113`；V [FSDPEngine._build_fsdp_module:431–450][V-fsdp] |
| 原始参数 dtype / 训练时分片驻留 | mixed_precision=true 时 FP32 master；native offload 下 CPU pinned | 示例指定 `model_dtype=bfloat16`；无 native policy 时训练 context 把 BF16 参数分片放 NPU；若显式 policy=true 则 BF16 分片在 CPU | L VeOmni `omni_impl.py:82–87`；L FSDP `omni_impl.py::_build_module:188–224`；V `BaseEngineCtx:314` |
| 计算参数、RS dtype | BF16 / FP32 | 默认同为 BF16 / FP32；不能声称 VeOmni 的 RS dtype 天生比 FSDP2 大一倍 | O `MixedPrecisionConfig:248–254`、O `torch_parallelize.py:199`；V `FSDPEngine._build_fsdp_module:372–379` |
| RS 后 `.grad` dtype | 转回 orig_dtype=FP32；native policy 再拷到 CPU | 示例 orig_dtype=BF16，FP32 RS 输出转 BF16；无 native policy 留 NPU，有 policy 则 BF16 D2H | T [foreach_reduce:600–642][T-collectives]；T `_fsdp_param.py::init_dtype_attrs:437–449` |
| 分组 | 文本 decoder、vision block、audio encoder 各按类 fully_shard，未接管参数归 root；默认没有单独包装 embedding / lm_head | 本地 HF adapter 仅保留文本 decoder 为 `_no_split_modules`；V 另按 embedding 类型和 `embed_tokens/lm_head` 名称包装，tie=true 时排除相关单独包装；其余归 root | O generated Qwen3 model `2885–2889`；O [parallelize_model_fsdp2:111–123,289–335][O-parallel]；L `thinker_training_adapter.py:78–117`；V [apply_fsdp2 / _select_fsdp2_wrap_targets:549–600][V-utils] |
| shard 维度 / 大小 | 默认 EP=1 时普通 FSDP2 dim0 padding 分片；world16、fsdp_size=-1 对应 shard16 | 默认也是 dim0 FSDP2；不能把“VeOmni”理解成未分片。EP/emb parallel 启用才增加额外 mesh/placement | V VeOmni engine `151–178`；O `torch_parallelize.py:232–277`；T `_fsdp_param.py::_init_sharded_param:332–400` |
| forward / backward 预取 | 本地 forward_prefetch=false；不建立 VeOmni 手动前/反向预取链，但 **T 默认 backward prefetch 仍在** | FSDP2 forward_prefetch=false 也不禁止默认 backward prefetch；两者是共有机制 | O `torch_parallelize.py:337–356`；V `fsdp_utils.py:602–613`；T [pre_backward:476–493 / _backward_prefetch:632–645][T-group] |
| reshard | 子组显式 reshard_after_forward=true，backward 也通常 reshard | 对照默认显式 true；不能照搬“root 永不 reshard”的旧概括 | O `build_parallelize_model:413`、`:196,335`；V `FSDPEngine:450`；T `_fully_shard.py:206–219`、`_fsdp_state.py:184–187` |
| micro / 累积 | micro=2；默认完整 mini 每 rank=16×2/16=2，所以一次 train_batch 是 1 次 backward；VeOmni 不设置 no-sync | 同等 batch 下也只有1次。若将来实际有多个 micro，V FSDP2 在非最后 micro 关闭 gradient sync、保留完整梯度；VeOmni 每 micro 同步，native policy 可能对已有 CPU grad 做阻塞 D2H 后累加 | V `trainer_base.py::_update_actor:1734`、`engine_workers.py:262–269`、`engine/utils.py:122–128`；V VeOmni `432–439`；V FSDP `_gradient_sync_context:672–696`、`:725–743`；T `foreach_reduce:616–637` |
| 模型计算/冻结集合 | native VeOmni forward；当前明确冻结 visual/audio_tower；SDPA、remove_padding=false 的默认脚本 | HF thinker forward；AVQA 对照脚本 remove_padding=true、model_dtype=BF16，不能当成只切 backend 的严格对照。已查的 FSDP2 模型构造/adapter 中也没有 VeOmni 那段双塔 `requires_grad_(False)`，不能仅凭 actor.freeze_vision_tower 配置宣称冻结集合相同 | L VeOmni `omni_impl.py:90–96`；L FSDP `omni_impl.py:205–228`；L `model_base.py::OmniModelBase.configure_model:623–645`；两份 AVQA/base 脚本 |

**root reshard 更正：** T 2.10 只有 `reshard_after_forward=None` 才在 lazy init 自动让 root 不 reshard；当前两条包装路径显式传 bool true，`_auto_reshard_after_forward=False`。V `fsdp_utils.py:600` 的注释“root 不 reshard”不能覆盖这个实际实现。VeOmni root 组更大的参数覆盖范围，可能带来更大的聚合分配及 forward/backward 再次 unshard；不应把它错误解释成 root 必然一直不释放。

## 参数、梯度、通信缓冲区的大小与存活期

定义：某一实际 FSDP 组中，P=全部参与 AG 的参数 padding 后元素总数，G=本次实际产生梯度的参数 padding 后元素总数，s=shard 数（默认16）。冻结参数可贡献 P，不贡献 G；不同分组不能用同一个 P/G 直接比较。以下是标准 BF16 计算、FP32 reduce 路径的字节公式，不是服务器测量；别名 storage 不重复相加。

| 对象 | 大小 / dtype / 位置 | 创建、保留、释放位置和条件 |
|---|---|---|
| 参数原始分片 | VeOmni CPU FP32 约4P/s；BF16 FSDP2 对照约2P/s，训练位置取决于 native policy | T `_fsdp_param.py:362–400`，offload=true 才搬 CPU / pin；模型及 optimizer 长期持有 |
| 参数 H2D / cast 临时量 | VeOmni 先 H2D FP32 约4P/s，再 cast BF16 约2P/s；不是直接只传 BF16 shard | T `_fsdp_param.py::all_gather_inputs:782–786`，原始 shard `.to(device, non_blocking=True)` 在 dtype cast 之前。N `_patched_get_param_all_gather_inputs:96–121` 对 offload 参数不走普通 foreach fast-path |
| AG 拼接结果 | BF16 约2P；本 rank 输入是输出的一段 view | T `foreach_all_gather:236–289`；N `_patched_all_gather_copy_in:142–159`。copy-out 产生的完整参数另有 storage；AG 拼接结果由 `_all_gather_result` / `comm_ctx.all_gather_state` 保留到相应事件依赖建立后清引用（T group `407–427`） |
| 完整计算参数 | BF16 约2P，去掉 padding 的实际值依参数形状 | T `_fsdp_param.py::init_all_gather_outputs:465` / `init_unsharded_param:480`；reshard 的 `free_unsharded_param:694–714` 清 storage。清 Python 引用 / resize(0) 不保证跨流物理回收已经完成 |
| 完整 autograd 梯度 | 常规同步路径 BF16 约2G | T group `post_backward:511–527` 转交 `unsharded_grads`；T `foreach_reduce:527–530` copy-in 后清列表。不应统称“完整梯度一直保留到 RS 结束” |
| RS 输入 | **每 rank 都需要 FP32 4G**，不是4G/s | T `foreach_reduce:517–527`，计算流 allocate/copy-in；T group `591–593` 保存输入；下一组 `533–540` 建流依赖并清引用。后端 stream 使用记录仍可能让 storage 等待复用 |
| RS 输出 | FP32 4G/s；VeOmni 是梯度 D2H 源 | T `foreach_reduce:535–556` 在 RS 流分配；`:602` 转 orig_dtype。BF16 对照另有转换后的约2G/s；是否短时重叠由执行/释放进度决定 |
| CPU 梯度 | VeOmni FP32，本地真实 shard 约4G/s；有 native policy 的 BF16 对照约2G/s | T `foreach_reduce:617–642`：首次 grad 且 pin_memory 才异步 D2H，记录 grad_offload_event；已有 grad 时阻塞拷贝后 CPU +=。N `_patched_finalize_backward:68–73` 主机等待这些事件 |
| optimizer 状态 | native policy 下跟随 CPU 参数分片建立并长期保留 | L VeOmni `omni_impl.py:125–127`；V `_build_optimizer:271–284` 调 O AdamW，O `optimizer.py:261–319`。阶段手动开关 false 不等于 optimizer state 在 NPU；也不等于优化器不存在 |
| 激活 / workspace | 输入、模态、重算及 kernel 决定，无法由 P/G 推出 | 两条路径采用不同模型前向和 padding 默认；checkpoint 会叠加重算临时量。N workspace 分配器是独立缓存，不能把所有 APP/HBM 都算作 FSDP RS |

约 `2434065.5 KB` 的历史块没有本次 shape、单位和归属，不能强行认成 RS 输入。公式可以说明大组为何产生 GB 级整组分配，不能替代实际参数配置。FP32 master 使 CPU 分片/梯度和 H2D 生命周期改变，但**相同 G 的 FP32 RS 输入在两条路径上同样大**。

## 一个具体的延迟复用条件：RS 输入的分配流不同于等待流

| 文件、函数、行号 | 条件与效果 |
|---|---|
| T [FSDPCommContext.lazy_init:70–78][T-group] | 为计算、AG copy-in、AG、RS 提供不同执行流；RS 使用独立 stream |
| T [foreach_reduce:497–539][T-collectives] | `reduce_scatter_input` 在当前计算流分配；进入 `with stream(reduce_scatter_stream)` 后才分配输出并调用 RS |
| T `DefaultReduceScatter.__call__:116–131`；T `distributed_c10d.py::reduce_scatter_tensor:4584–4598` | 默认 async_op=false 在调用 RS 的流中 `work.wait()`；false 不代表强制全设备主机同步 |
| N [ProcessGroupHCCL::collective:4202–4235][N-hccl]、`_reduce_scatter_base:6211–6216` | 默认模式1记录输入/输出的内部 HCCL stream；模式2改用 stash 保持引用。默认值在 `OptionsManager.cpp::GetMultiStreamMemoryReuse:73–95` |
| N [WorkHCCL::synchronizeInternal:1055–1092][N-hccl] | 向当前流添加 HCCL 完成依赖，并尝试 eraseStream；普通路径不是 host barrier（barrier/显式 blockingWait 是另有条件的分支） |
| N [NPUCachingAllocator::eraseStream:3442–3471][N-allocator] | **block 的分配流 != 当前提交等待的流时，3463–3468 直接返回，不移除使用标记。** 常规 RS 输入正是计算流分配、RS 流 wait；这是有明确条件的保留，不是凭空假设“后端可能有引用” |
| T group `post_backward:533–540`；N allocator `free:1476–1513` | 下一组在计算流 wait_event 后清 RS 输入引用，不自动重走一次后端 erase。若仍有 stream_uses，free 先减 allocated，插入事件，暂不 free_block |
| N allocator `process_events:3093–3125`、`free_block:2396–2444` | 事件完成且 event_count=0 才减 active、把块放回缓存；每流遇到尚未完成的头部事件会停止该流的回收扫描 |

由此得到的**静态可行链条**：某 rank 的通信进度落后 → RS 输入失去 Python 引用但 HCCL stream 标记未消 → Allocated 已降而 Active 仍高 → 主机继续提交工作并申请新块 → 更易到达本地分配压力点。这不要求永久泄漏，也不要求整层所有参数都保留到全部卡结束。

这里需保留两条边界：其一，这也是相同 PyTorch/NPU 后端的 FSDP2 共有机制，VeOmni 的区别在请求大小/分组、D2H、激活和推进节奏；其二，默认 eraseStream 的流检查是安全保护，不能建议直接删除。D2H 在 RS/post-reduce 流中又占用后续执行时间，可能延长流水线，不等于每份输出都必须依靠 recordStream 才安全。

N `_add_fsdp_patch.py:277–283` 的 essential patch 与 `:286–319` 的 enhance patch 是两套入口。两条已审计框架路径导入 torch 的 fully_shard，未发现调用 N 包装器或设置 `_use_mem_cache`；不能把 post6 文件中的 FSDPMemCache 当成已启用功能。即使启用，`FSDPMemCache.free:51–58` 也是标记槽空闲、仍持有 tensor，不等于物理释放。

## 物理回收入口：不能仅凭 UnmapMem 判定重试

| 入口与代码证据 | 触发条件 | 物理释放、同步与重试计数 | 对本次层内现象的解释力 |
|---|---|---|---|
| **分配压力后的全量回收**：N [DeviceCachingAllocator::malloc:1231–1275][N-allocator]；`alloc_block:2693–2707` | 缓存查找/回调、新分配、可用缓存块释放后仍失败，且不在 graph capture | `1272` 设备同步 → `1274` 清 workspace → `release_cached_blocks(...,free_physical=true)` → `alloc_block(isRetry=true)`；只有进入后者才在2701增加 num_alloc_retries，可重试成功而没有最终 OOM | **最强候选**：按 rank 本地状态、请求大小、stream pool、可用设备内存触发，天然不绑定同一模型层编号 |
| **主动 empty_cache**：N `memory.py::empty_cache:168–180` → `Module.cpp::THNPModule_emptyCache:1287` → allocator `3380–3382,1651–1662` | 应用/框架显式调用 | 同样设备同步、清 workspace、release_cached_blocks(free_physical=true)；不增加 num_alloc_retries | 同样能产生物理释放及骤降；但当前已查的 VeOmni 层内 backward 无该调用 |
| **释放可用大缓存块后重试**：N allocator `malloc:1255–1258`、`release_available_cached_blocks:2764–2811` | 初次新分配失败，且 max_split_size 不是无限默认值 | 同步设备、release_block；后续 `alloc_block(false)` 不增加 retry 指标。`release_block:2858–2873` 断言非 expandable、调用 aclrtFree | 可造成层内同步/物理下降；该分支本身不是 expandable 的逐页 UnmapMem 路径 |
| **阈值 GC**：N allocator `malloc:1248–1252`、`garbage_collect_cached_blocks:2624–2682` | 已设 allowed maximum、GC threshold>0、超阈值且有可释放块 | **实现2660确实同步设备，尽管2628注释说不强制同步**；再 release_block，无 retry 增量 | 也是压力相关入口；默认 GC threshold=0 未启用。不能仅凭注释说 GC 无同步；也不能直接用它解释 expandable UnmapMem |
| **仅清虚拟映射**：N `emptyVirtAddrCache:3385–3390`、`unmap_block:2899–2903` | 显式 empty_virt_addr_cache 且 expandable=true | free_physical=false，unmapHandles 将 handle 存入 pool；本入口不直接 FreePhysical | 能解释 UnmapMem，单独不足以解释“大量 FreePhysical + HBM 骤降” |
| **workspace 增长 / 清空**：N `NPUWorkspaceAllocator.cpp::malloc:75–139` / `empty_cache:226–268` | 本流需更大 workspace，或上层清空 workspace | 增长时设备同步并 aclrtFree 旧块；清空时遍历释放；其本体用 aclrtMalloc/Free，而非 caching expandable 的 UnmapMem/FreePhysical | 可叠加 APP/HBM 变化，但不能把这两种底层释放 API 全归给 workspace |
| **其他边界入口**：N allocator `releasePool:2198–2214`、`FreeDeviceCachedMemory:3852–3854` | graph pool 最后用户释放且 ForceUncached / 显式内存回收接口 | 可进入同一 release_cached_blocks 或 emptyCache；此外 ExpandableSegment::map `469–476` 失败回滚会 FreePhysical，但没有配套已映射区的 UnmapMem | 条件不同于常规训练层；已查路径中无相应层内调用证据，不作为主解释 |

`NPUAllocatorConfig.h:137–144` 的源码默认：multi_stream_lazy_reclaim=false、memory_fraction=1、max_split_size=无限、GC threshold=0、expandable=false；环境变量/运行时可覆盖。V `engine_workers.py::update_weights:767,820` 切换 expandable；V `utils/device.py::set_expandable_segments:136` 提供实际设备入口。不能以 standalone 环境没列出变量就断言 worker 一定是默认值。

框架中的主动清理调用也已定位：

| 文件、函数、行号 | 调用位置 / 是否能解释 backward 层内 |
|---|---|
| V `engine_workers.py::ActorRolloutRefWorker.init_model:689–690` | 初始化收尾，不是层内 |
| V `engine_workers.py::ActorRolloutRefWorker.update_weights:767–768,809–812` | naive 权重同步前后主动 aggressive_empty_cache；可能影响下一阶段缓存初态，但不能冒充当前 backward 层内调用 |
| V `memory_utils.py::aggressive_empty_cache:31–74` | gc.collect → device.empty_cache → 可选 synchronize；最多重试3轮是此 Python helper 的清理轮数，**不等于 allocator.num_alloc_retries** |
| V `veomni/utils.py::offload_veomni_model_to_cpu:57–75` | model.reshard、model.cpu 后 empty_cache；当前 native policy 下自动标志为 false，不会从 train context 走入；显式 to(model=True) 仍可触发 |
| V `fsdp_utils.py::offload_fsdp2_model_to_cpu:196–211` | FSDP2 手动阶段 offload 后 empty_cache；注释“不是同步点”不能套到 NPU：post6 emptyCache 实现明确同步设备 |
| O `offloading.py::offload_model_to_cpu:124–143` | 存在显式整模 offload helper，但当前 backward 使用的是 activation context，不调用此 helper |

**直接物理回收链**为 N allocator `release_cached_blocks:2815` → `synchronize_and_free_events:2992`（让可回收块进入池）→ `release_blocks:2952` → `unmap_block:2899` → `ExpandableSegment::unmapHandles:594–623`。当 free_physical=true 且确实有可解除映射的完整物理粒度，后者在605再次设备同步，在616/618循环 UnmapMem/FreePhysical；`release_blocks:2975–2980` 还会释放先前缓存的物理 handles。

所以 API 调用数量对应物理段/页粒度，不能等同于 tensor 数或层数。普通 `free_block` 只是回到缓存；仅通信完成 / finalize_backward 等待，也不会凭空走完整的物理回收链。发生大量物理释放，说明还要有请求释放缓存/解除映射的入口。

## 四个新现象逐项裁定

| 观察 | 压力触发同步回收如何同时解释 | 限制 / 竞争解释 |
|---|---|---|
| 各 rank 慢层编号不同 | `malloc:1231–1275` 在各自本地 allocator 中按请求和可用块决定；过去事件完成速度、激活/workspace、不同 stream 缓存可使阈值落在不同层。无需某一层代码固定更慢 | 这是条件推断，不证明具体哪类对象先积压；主动清理若发生在不同阶段或外部回调也可错位，但当前层内调用链不支持固定显式清理 |
| 慢层开始恰好达到各自 APP/HBM 峰值 | 新申请不能满足时，发起该申请的层承担同步与清池；层可能只是触发点，不是主要内存生产者 | “观测峰值”不等于达到硬件总容量；process fraction、块大小/池限制、其他进程/非 caching 内存也会影响失败。APP/HBM、Reserved、Active 是不同口径 |
| 慢层中大量 UnmapMem/FreePhysical | expandable 全量缓存回收执行逐物理粒度 unmap/free；前面设备同步还可能等待其他 rank 的 HCCL | 此特征强于单纯 Active 下降，但重试/显式empty_cache共享此链，不能单独区分；也不能将 profiler 的层包含区间全算成计算耗时 |
| 占用骤降，随后缓慢上涨并平稳 | 回收释放缓存物理页；后续同类请求重新映射/申请并逐步建立可复用的工作集，达到工作集后趋稳 | “趋稳”是依赖后续请求稳定、异步工作不再持续积压的推断，不是 allocator 保证。进入较轻的后续阶段也能下降趋稳；不能据此证明已修复或永久泄漏 |

单纯“通信慢”能解释等待和 Active 延迟下降，但**不能单独解释物理 Unmap/FreePhysical**；需要与回收入口组合。单纯“CPU offload 会释放显存”也不足：D2H、清引用、缓存块变可复用、物理页归还，是四个不同动作。

历史 Active/Allocated 差值与此机制相容，但没有把不同 run 的数据相加。新四项观察来自用户，本报告没有原始 trace，也不宣称从源码还原出了它们的实际触发栈。

**最终排序：** 第一候选是本地内存压力 → 分配器同步回收重试；具体可积压对象以跨流 FP32 RS 输入为有明确源码条件的候选，同时保留 AG、D2H、激活/workspace 的贡献。第二候选是共享相同回收链的主动清缓存，但已发现的框架调用在初始化/权重同步/阶段搬运，不能自然解释各 rank 不同 backward 层内。当前源码不支持把旧 to override 缺失、整层统一释放、固定某张慢卡、必然的梯度双重累积当成根因。

这是一份带触发条件的静态诊断，没有提出经 NPU 验证的修复，也没有增加训练或回传任务。

## 固定源码链接

[V-base]: https://github.com/verl-project/verl/blob/fefb080262e1c015a0ea05f958822a6a512dc795/verl/workers/engine/base.py#L314
[V-veomni]: https://github.com/verl-project/verl/blob/fefb080262e1c015a0ea05f958822a6a512dc795/verl/workers/engine/veomni/transformer_impl.py#L493
[V-fsdp]: https://github.com/verl-project/verl/blob/fefb080262e1c015a0ea05f958822a6a512dc795/verl/workers/engine/fsdp/transformer_impl.py#L366
[V-utils]: https://github.com/verl-project/verl/blob/fefb080262e1c015a0ea05f958822a6a512dc795/verl/utils/fsdp_utils.py#L549
[O-parallel]: https://github.com/ByteDance-Seed/VeOmni/blob/f90b3dc6fbb0ce693745223cc7a94064123dbf4d/veomni/distributed/torch_parallelize.py#L196
[T-group]: https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L70
[T-collectives]: https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L446
[N-allocator]: https://github.com/Ascend/pytorch/blob/38e483ee79ec8a3286517808740a811c0f0ae66c/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp
[N-hccl]: https://github.com/Ascend/pytorch/blob/38e483ee79ec8a3286517808740a811c0f0ae66c/torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
