# 当前快照审计：通信偏斜与 Active Memory

## 证据范围

审计日期 2026-09-25。仓库 HEAD `d71ae4c11935f8dd073644b05943a84ecd1a79b3`，分支 `qwen3-omni-veomni`。已读根目录 `AGENTS.md`；相关路径未发现更深层 AGENTS。开始时已有：

- 修改：`examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1_separate_async.sh`。
- 未跟踪：`.worktrees/`。
- 三个历史 `pytest-cache-files-*` 目录不可读；未触碰。

以上保持原状。此次新增诊断目录，以及 engine 末尾默认关闭的四行 probe 接入；没有切分支、合并远端、提交或推送。所有下面的“源码确认”描述**当前工作区或列出的固定源码**，不自动等于服务器运行事实。

| 标识 | 来源 / revision | 可信范围 |
|---|---|---|
| L | 当前仓库 HEAD + 初始工作区 | 当前本地实现 |
| V | [verl fefb080](https://github.com/verl-project/verl/tree/fefb080262e1c015a0ea05f958822a6a512dc795) | `.github/verl_pin.txt`、`pyproject.toml` 一致；服务器安装待核 |
| O | [VeOmni f90b3dc](https://github.com/ByteDance-Seed/VeOmni/tree/f90b3dc6fbb0ce693745223cc7a94064123dbf4d) | `.github/veomni_pin.txt`；Docker 按 pin 安装、`--no-deps` |
| T | [PyTorch 2.10.0 / 449b176](https://github.com/pytorch/pytorch/tree/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0) | Docker 目标版本的条件性参考；服务器 torch 尚未采集 |
| N | [Ascend/pytorch 85f239d](https://github.com/Ascend/pytorch/tree/85f239d69ebb205ac18d968ffd03fa1d7974dab3) | 获取时 `v2.10.0` **分支**的不可变 snapshot，不能冒充 `2.10.0.post6` wheel 源码 |

依赖文件下载到忽略目录 `_sources/{V→verl,O→veomni,T→torch,N→npu}/`；准确 URL/SHA256 见 `source_manifest.json`。后文依赖行号均相对于该仓库文件。Dockerfile 的 torch-npu 是 **post4**，用户已确认服务器是 **post6**，所以 HCCL/allocator 的逐行机制目前标为“源码确认（N 参考）/服务器待验证”。`collect_env.py` 采集实装版本、导入路径、可得 git_version 和实际 FSDP 方法源码/哈希；不拿 Windows 包版本代替服务器。

目前没有收到可归属本次 run 的 profiling 数据。没有任何 NPU 实测结论。

## 配置与实际路径

**源码确认（L/V）**：基线入口为 `run_baseline.sh` → `examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_veomni_npu_avqa_v1.sh:35` → `run_qwen3_omni_thinker_gspo_veomni.sh:41` → `verl_omni.trainer.main_omni`。基础 YAML → 本地 omni YAML `_self_` → 基础脚本参数 → AVQA 参数 → 用户附件参数 → 本脚本额外尾部参数；同一键后出现者覆盖前者。环境变量仍可决定 `NUM_NPUS/NNODES/ROLLOUT_TP/ATTN_IMPLEMENTATION/USE_REMOVE_PADDING` 等，须记录实际值。

**源码确认（L）**：`main_omni.py:336,344` 的在线 `policy_gradient` 分支设置 `trainer.use_v1=True`，调用 V 的 `run_ppo/TaskRunnerV1`；`trainer/config/omni_trainer.yaml` 选择 `omni_sync`；`trainer/omni/ray_omni_trainer.py:69` 的 `OmniPPOTrainerSync` 继承 V1 `PPOTrainerSync`。不能用本文件下方的 offline DPO trainer 或旧 `verl_omni/workers/engine_workers.py` 推导这次更新。

**源码确认（V）**：`trainer/main_ppo.py:115` 的 TaskRunnerV1 创建注册 trainer；`trainer/ppo/v1/trainer_base.py` 使用 V 的 `ActorRolloutRefWorker`，内部 `TrainingWorker` 在 `workers/engine_workers.py:125` 通过 EngineRegistry 创建 engine。外部模块 `VERL_USE_EXTERNAL_MODULES=verl_omni` 注册 `model_type=omni_model, backend=veomni, device=npu` 对应 L 的 `OmniVeOmniEngine`（`omni_impl.py:48`）。

**源码确认（L/V）**：静态 MRO 为 `OmniVeOmniEngine → VeOmniEngineWithLMHead → VeOmniEngine → FSDPEngineWithLMHead → FSDPEngine → BaseEngine → object`。V `veomni/transformer_impl.py:99,817`，V `fsdp/transformer_impl.py:87,1121`。`forward_step` 来自 FSDPEngineWithLMHead；模型构造、输入适配来自本地 override；`forward_backward_batch/optimizer_step/to/train_mode/eval_mode` 来自 VeOmniEngine。worker probe 记录实际 MRO/方法文件/行号以核对 monkeypatch。

| 项目 | 当前无额外环境/CLI 覆盖时的值 | 来源 |
|---|---|---|
| 训练 world | 16 × 1 节点 | AVQA 脚本 NUM_NPUS/NNODES 默认 |
| rollout TP | 2 | AVQA/base 的 ROLLOUT_TP；与训练 DP 不是同一维度 |
| 训练 SP / EP | 1 / 1 | V `trainer/config/engine/veomni.yaml:13,15` |
| 训练 DP / shard / replicate | 16 / 16 / 1 | V engine `__init__:151`，`fsdp_size=-1` |
| param / reduce dtype | bf16 / fp32；master fp32 | O `arguments_types.py:241`；L `_build_model_optimizer:78` |
| native offload | true | 附件最后覆盖 |
| 手动 offload 配置 / 运行标志 | true,true / false,false | L `omni_impl.py:113` |
| 冻结 | visual 与 audio_tower | base 脚本与 L `omni_impl.py:87` |
| checkpoint | 开；non-reentrant | base 脚本；V engine YAML `enable_reentrant=false` |
| activation offload | false | L `trainer/config/omni/model/omni_model.yaml:50` |
| attention / remove_padding | sdpa / false | base 脚本环境默认；不是 FA 默认 YAML 最终值 |
| dynamic batch / PPO epochs | false / 1 | base 脚本；V actor YAML:119 |
| profiler ranks | 全部训练 rank | V `utils/profiler/profile.py:128` 先判断 all_ranks，ranks=[0] 不缩窄 |
| profiler step / analysis | step 2 / false | 附件；V `mstx_profile.py:158` 传 `analyse_flag=analysis` |
| rollout 权重切换 | 默认 naive、free_cache_engine=true | V `trainer/config/rollout/rollout.yaml:61,272`；若服务器额外覆盖须复核 |

### micro-batch 推导

**源码确认（V）**：`trainer/ppo/v1/trainer_base.py::_update_actor:1734` 将 PPO mini 乘 rollout.n：`16×2=32`；`TrainingWorker.train_mini_batch:262–269` 除训练 DP；`train_batch:350` 注入 engine micro 参数；`engine/utils.py::prepare_micro_batches:122–128` 固定大小切分。`force_group_size` 未另设时为 1。

**推断（以上默认 mesh 和当前完整 mini 成立）**：每 rank mini=`32/16=2`，micro=2，故一次 train_batch 只有 **1 个 micro-batch / 1 次 backward**。默认一个 PPO epoch；32 条 rollout 输出对应一个全局 mini。若 SP=S、world=W，DP=W/S，完整 mini 的 micro 数是 `32/(W/S)/2=16S/W`（还要求整除、实际数据和 force_group_size 一致）。rollout TP=2 不再除一次。V1 的 batch padding/对齐在 `trainer_base.py:1496`，所以须以真实 worker `local_samples/actual_micro_batch_count` 最终确认。旧 micro=1 的“两轮累积”不能套用。

## offload 入口逐项核对

**源码确认（L）**：`omni_impl.py::_build_model_optimizer:113–126` 在 native offload 时置 `_is_offload_param=False`、`_is_offload_optimizer=False`、`_uses_fsdp2_cpu_offload_policy=True`；仅遍历 buffers 放到计算设备，参数由 CPUOffloadPolicy 管理。没有本地 `to()` override。

| 入口 | 确认的行为 | 证据 |
|---|---|---|
| 初始化结束 | `to(cpu, model=False, optimizer=False, grad=False)`，不整模搬运 | V VeOmniEngine.initialize:262；标志在构造时已被 L 改写 |
| train/eval 自动 context | property 返回 `_is_offload_*`；进入时两者均 false 则返回；退出传 false | V FSDPEngine:174；BaseEngineCtx:314；VeOmni ctx:658,684 |
| train context 退出 | 清梯度；不是手动搬整个模型 | V EngineTrainModeCtx:697 |
| save/load checkpoint | whole-model load 有 native policy guard；手动 offload 条件 false | V VeOmniEngine:533,552；本基线 save_freq=-1、resume disable，不执行周期 checkpoint |
| 权重导出 | native policy 跳过整模 load，但每个 DTensor 仍可 H2D/full_tensor；这是独立临时存储 | V VeOmniEngine:576,611,633；L:182 的 bf16 导出适配 |
| naive 权重同步末尾 | offload 由 property 判断，当前不执行；该阶段自带 cache/sync 操作 | V ActorRolloutRefWorker.update_weights:767–820 |
| 显式 RPC `TrainingWorker.to` / `ActorRolloutRefWorker.to` | **不检查标志**；调用者传 model=True 仍会整模 `.to` | V engine_workers:159,534；VeOmniEngine.to:493；utils.py:79 |

**结论（源码确认 + 边界）**：当前已查到的基线自动路径没有“因为缺少 override 就必然整模上卡”的证据。固定 V1 trainer 源码未发现无条件调用上述显式 RPC 的基线路径；RPC 仍然存在，不能宣称所有入口已封死。若服务器增加调用方/补丁或 checkpoint backend 被改，需结合实际方法源和日志核查。本轮不添加防御性 `to()` 修复。

## 构造、分组、计算与同步

**源码确认（L/O）**：本地 engine 用 float32 master 构建原生 VeOmni 模型；`configure_veomni_model` 保留原生 forward 和 split hints（L adapter:97）。先冻结视觉/音频，再 `build_parallelize_model`；O `torch_parallelize.py:433` 在 mixed precision 时 float()，:442 启用 non-reentrant checkpoint，:196 设置 MixedPrecisionPolicy，:211 CPUOffloadPolicy，:289–335 子模块到 root 逐个 fully_shard。

**源码确认（O）**：原生 top-level 模型在 `generated/patched_modeling_qwen3_omni_moe_gpu.py:2885` 固定三个 split 类：文本 decoder、vision block、audio encoder。文件名有 gpu 也不能仅凭文件名排除 NPU：注册器 `qwen3_omni_moe/__init__.py:41` 选择该 generated 类；实际 NPU kernel patch/ops 来自平台 dispatch，需实装核对。当前本地适配不改 split 集合。冻结塔仍可有 FSDP 参数/all-gather，通常不产生参数梯度 RS。root 还管理未被子组接管的参数。因此**不等于“一层一组”，更不等于“每组都有梯度通信”**。EP>1 或 mixed precision 忽略模块还会继续拆组（O:295,316）；默认 EP=1，不能假定专家单独做 EP 通信。

**源码确认（V）**：`BaseEngine.train_batch:126` 每 mini 先 zero_grad，再 `forward_backward_batch`，最后 optimizer_step。VeOmni `forward_backward_batch:398–410` 每 mini 先算 loss_mask token sum，执行标量 DP AllReduce 和 `.item()`，再切 micro；:432–439 每 micro 执行 forward/backward，未设置 no-sync。`FSDPEngineWithLMHead.forward_step:1512` 搬输入、autocast、模型与 loss；:1560 已有 loss `.item()`，是原训练路径同步点。probe 不新增此类操作。

**源码确认（V/O）**：optimizer_step:370–384 包含 clip norm、DTensor norm full_tensor、finite 检查、optimizer.step 和 norm.item；O `optim/optimizer.py:261,317` 默认 AdamW(fused=False, foreach=True)。参数/梯度本地分片在 CPU 时，对应 Adam 状态也在 CPU 延迟初始化；第一次 step 前 state 空不能报作零成本/失踪状态。Adam 尾部时间还可能包含 DTensor 分发、线程调度、内存分配与同步，不能把整个区间解释成 CPU 算术。

**推断 / 待验证**：历史 optimizer 之后的标量 ReduceSum+AllReduce，与下一次 `forward_backward_batch` 的 loss_mask.sum/DP all_reduce 路径相符，但没有 correlation/phase 证据前不能给 `hcom_allReduce__612_268_1` 命名归属；它不是解释所有 backward RS 的依据。

## 每 rank storage 生命周期

以下为 **T 参考源码确认，服务器 torch/NPU patch 待核**。设一个实际通信组 shard 大小为 s；参数 i 在 EP 等前置切分后的完整本地形状元素数为 n_i，dim0 padding 后元素数 p_i，p_i/s 是 padded shard 元素数。`G` 是本次实际产生梯度的参数集合，不包括冻结/未使用参数。必须用实际 storage（`DTensor._local_tensor.untyped_storage().nbytes()`）计量，不使用 DTensor 全局 numel，storage 别名只计一次。

| 对象 | 位置 / dtype / 每 rank 条件字节数 | 创建与持有 | 引用/存储释放及最终复用 |
|---|---|---|---|
| master 参数分片 | CPU pinned，fp32；约 `4Σp_i/s`，含 padding | T `_fsdp_param.py:362–400`；FSDPParam.sharded_param、模型与 optimizer 持有 | 长驻到模型销毁/替换；不是每组 backward 后释放 |
| 参数 H2D 源/目标及 cast | 源 CPU shard；目标 NPU fp32 `4Σp_i/s`，随后 bf16 `2Σp_i/s` | T `all_gather_inputs:782–786` 先 H2D 再 cast；all-gather copy-in 流 | 源 master 长驻；临时目标在 copy-in 消费后失去引用，复用仍受所属流/后端使用约束 |
| AG 拼接输出与输入 view | NPU bf16，组总 `2Σp_i`；输入是其中 `1/s` view，不另计同一 storage | T `_fsdp_collectives.py:175–188,236`；AllGatherResult | forward 可延迟到下一组 copy-out event 后清引用，T param_group:407–427；HCCL 使用另有后端保护 |
| 临时完整参数 | NPU bf16，约 `2Σn_i`，可能带 padding | T param `init_all_gather_outputs:465`、`init_unsharded_param:530`；通常是 per-param AG output 的 view | `reshard → free_unsharded_param:694–714` 缩 storage，不等于 Parameter Python 对象销毁；root/不 reshard 组生命周期更长 |
| 完整本地 autograd 梯度 | NPU bf16，约 `2Σ(G)n_i`；无 no-sync 时通常如此 | unsharded_param.grad；T param_group:514–527 移交 unsharded_grads 并置 grad=None | T collectives:527 copy-in 之后 :530 clear list；计算流按序保障原梯度可复用，**无需全都保留到 RS 完成** |
| RS 拼接输入 | NPU fp32，`4Σ(G)p_i` | T collectives:516–527；当前计算流 allocate/copy-in；comm_ctx.reduce_scatter_state 保存一个显式前组引用 | 下一组 current_stream.wait_event 后清引用（param_group:533–540）；流依赖不等于 HCCL allocator tracking 已消失 |
| RS 输出 / D2H 源 | NPU fp32，`4Σ(G)p_i/s`；每参数是 as_strided view | T collectives:535–556 在 RS 流分配；:610 分出 view；post_reduce 流执行除法/转换/拷贝 | native offload 下不保留 NPU grad 为长期 `.grad`；RS/HCCL/D2H 必须按依赖完成才能安全复用；无 offload 时 `.grad` 会保留输出别名到清梯度 |
| D2H 目标 / CPU grad | CPU fp32，本地各实际 sharded_size；实际 pinned 状态和 storage 由 probe 核对 | T collectives:616–642；CPU grad DTensor/optimizer 持有 | 首次 grad且pin_memory时非阻塞，记录 grad_offload_event；已有grad则阻塞 D2H 后 CPU add；每 mini zero_grad 或 train context 退出释放 |
| Adam exp_avg/exp_avg_sq/step | native offload 下主要 CPU；fp32 master 对应两个 fp32 moment；`约8Σ(trainable)n_local`，step 为小标量 | O AdamW 创建，首次有效梯度 optimizer.step 延迟初始化；optimizer.state 持有 | 跨 step 长驻；native policy 下不是每 mini 反复整份上/下卡。实际 local/padding/view 以 probe 为准 |
| 激活/重算/workspace/buffers | NPU，多种 dtype，随输入长度/模态/算子变化，无法从参数数目算出 | autograd saved tensors、checkpoint 和算子/模型 | 激活在最后消费/重算后释放；workspace/缓存可能有独立 allocator；不能与梯度共享同一生命周期假设 |

实际字节数本地无法取得：没有服务器模型实例、配置权重形状和分组运行状态。probe 会输出全模型本地去重 storage、实际组清单与 FP32 RS 条件上界。不是用上述公式伪造测量值。`2434065.5 KB` 没有 unit 口径、allocation stack、dtype/shape/组成员和地址证据时无法归属 RS；完整参数/AG/拼接/workspace 都须排查。

### D2H、finalize 与累积

**源码确认（T）**：`foreach_reduce:616–637` 仅当 `pin_memory && sharded_param.grad is None` 时 `non_blocking=True`；已有 CPU grad 时 D2H 阻塞，再在 CPU `.+=`。`FSDPParamGroup.finalize_backward:604` 对 post_reduce 加流依赖，并逐个 `grad_offload_event.synchronize()` 等 D2H 完成，再清未消费 AG。N 的 essential patch `_patched_finalize_backward:68` 保留这一区分；它不是无条件全设备同步。

**推断（当前完整 mini）**：只有一个 micro 且 train_batch 先 zero_grad，常规路径不进入“第二 micro CPU grad 累加”；第一轮异步 D2H 和 finalize 的主机等待仍存在。共享参数重复 hook、实际配置不同或外部改动等只能通过运行证据确认，不能预先排除。

## 已有流控为何不能直接排除积压

**源码确认（O/T）**：本地 `forward_prefetch=false` 使 O `torch_parallelize.py:337` 不设置成串的手动前/反向预取列表；不代表禁用 T 默认 backward prefetch。T `param_group::_backward_prefetch:632` 根据逆 forward 顺序预取上一组；:490 当前组 unshard 无操作或等待已预取结果。AG 的 comm_ctx 保存上一组结果以交叠 copy-out；下一组按事件等待/清理。RS comm_ctx 也只显式保存前一组输入，在 :537 对 current stream 建 wait_event。

**重要区分（源码确认/推断）**：这限制流上的执行与 FSDP 层显式引用，不是“整个进程最多一份 RS buffer”的主机阻塞承诺。host 可以已发出多次分配/释放请求；通信后端还可能追踪其内部流，导致早先释放的 block 暂不可复用。默认 backward 预取还可能预取冻结/无 backward 组，finalize 负责等待清理；不能按文本 decoder 数量机械计算 pending 组数。

**源码确认（N 参考）**：`torch_npu/distributed/fsdp/_add_fsdp_patch.py` 的 essential patch（:290 左右）替换 finalize 和 AG copy-in。enhance patch 是另一入口：包装的 `torch_npu.distributed.fsdp.fully_shard` 调用 `_apply_fsdp_enhance_patch`。其 FSDPMemCache 按 dtype 复用 buffer，RS 只缓存首个 allocate（输入），foreach_reduce 后将 cache 槽标为空闲；持有缓存引用仍可使 Allocated/Active 都高。

**待验证**：当前 O 从 `torch.distributed._composable.fsdp` import fully_shard，并没有在已审计路径显式设置 `_use_mem_cache`。N cache 还需 enhance patch 已应用且模块 `_use_mem_cache=True`。所以“torch_npu 有缓存实现”不能推出“这次已启用”；worker 记录实际 fully_shard/allocate 方法源、组 `_use_mem_cache` 和 RS 标志。post6 中对应代码必须以实装为准。

**源码确认（O）**：HCCL premul patch 在 `torch_parallelize.py:279` 要求 NPU 且 extra parallel enabled，默认 EP=1 不满足；不能无条件套到当前基线。

### HCCL 与 allocator 的另一层生命周期

**源码确认（N 参考）**：`ProcessGroupHCCL.cpp:305–328` 区分调用流与内部 HCCL 流，先建 event 依赖；`collective:4661–4699` 根据实际 memory reuse mode 选择 recordStream 或 stash 引用；`_reduce_scatter_base:6651–6677` 还保护输出。`WorkHCCL::synchronizeInternal:1114` 等待语义通常为流阻塞，并按模式 eraseStream/清保留引用；barrier 分支才有显式 npuSynchronizeDevice。不能把普通 Work.wait 全部写成 host/device synchronize，也不能从 FSDP Python 没有 record_stream 推出后端没有记录。

**源码确认（N 参考）**：allocator `free:1524` 先降低 allocated_bytes、发 FREE_REQUESTED；若有 stream uses，则插入事件，否则 free_block。`free_block:2475` 要求 event_count=0，发 FREE_COMPLETED、降低 active_bytes 并入缓存池；Reserved 仍可能不降。`process_events:3241` 查询完成事件；`release_cached_blocks:2918` 先同步未完成事件，再归还可释放段。**普通 free 不等于全设备同步。** `malloc:1309–1323` 的申请失败重试分支则可能清 task queue / 同步设备并回收缓存；实际 post6 是否同样走这条路径，须由实装来源与 CPU 栈/重试证据确认。

因此各阶段应分开：最后上层引用释放 → allocator free request（Allocated下降）→ 后端相关事件完成 → allocator处理事件/块可复用（Active下降）→ 缓存段归还设备（Reserved下降）。事件完成到指标下降还可能隔着一次 allocator 查询。Active−Allocated 高支持“有已申请释放但未可复用的块”等候选，不提供这些块的张量身份。若仍有 HCCL/cache 强引用，则甚至尚未发 free request。多 allocator、单位或时间点不一致时不能相减。

## 历史观测台账与因果边界

全部以下数据来源为用户附件人工摘录，**run/step 未核实，标为历史运行线索**，不是本次基线测量：16 ranks 轮换晚到；计算约3%、未掩盖通信>40%、某通信统计等待约98.5%；Active 55864 / Allocated 8516 / Reserved 56056 MB；CPU aten::empty 24.7635s；约2434065.5KB 分配；RS 两 rank 的 `[24811.149,29077.953]` 与 `[29056.527,29078.273]` ms，sync 29108.231ms；optimizer rank2 `[38925.7524,44489.15803]`、rank8 `[38206.4749,53209.37432]` ms，后续标量起点差约8.721s。

这些区间如确认同组同钟，会支持早到 rank 等晚到及同步后释放的相关性，但不足以证明最初是谁拖慢。通信汇总占比与概览占比不相乘。大 empty 的存活期不替换调用 duration。optimizer 尾部与 backward 大 RS 分开筛选。

最多三个候选，尚无最终根因：

1. **先有计算/H2D/主机提交偏斜，通信等待延长后端相关 block 的不可复用时间，再触发分配重试，反过来放大下一组偏斜。** 机制对应 FSDP事件→HCCL记录流→allocator重试链。缺少：同 run 16 rank 的同组/同钟关联，晚到 rank 在首个异常 collective 前的 CPU/设备任务，free request/completion 或明确重试栈。反证：大偏斜出现之前无上游推进差，而 allocator 已先阻塞。
2. **先有局部分配/回收路径停顿或 native offload 的 H2D/D2H/主机进度差，造成后续 collective 提交迟到。** 不预设碎片或内存泄漏；候选包含实际 D2H等待、allocator跨流回收以及主机调度。缺少：一个长 CPU empty 前后的 allocation/free/address/流、前序任务完成时间、后续主机 collective enqueue；first-grad D2H/finalize 是否占主导。反证：申请停顿只在所有 rank 已进入同一 RS 之后出现，或首次偏斜时无内存压力。
3. **optimizer 阶段独立的 CPU shard/state 更新或调度偏斜，被下一处标量 collective 暴露。** 对应 AdamW/clip/next-mini token count 路径；仅解释尾部。缺少：AdamW 子区间 self/inclusive、主机线程/拷贝/等待事件、标量关联和实际 `.grad/state` placement。不能拿该候选替代 backward 根因。

优先用已有 profiling 找一个完整窗口：长 CPU allocation 开始 → 对应前序设备/通信结束 → 明确 free completion 或同 allocator Active 降低 → allocation 返回 → 后续 **host提交**及 device collective 开始。任何缺环均标缺口；同时检查短 allocation、无 Active 下降、晚到前无 allocation 等反证。首轮解析器仅提供时间关联，尚不能在未知数据库关系表上自动证明完整因果链。

下一步首先是环境与 schema 清单。若已有 profiling 无法区分“一 micro、无 CPU grad 累积”与“实际多 micro/旧配置”，信息量最高的最小补采是默认关闭的 worker metadata probe：不改 offload、batch、精度或同步策略；若每 rank 都只执行一次 forward_step 且进入 mini 前清梯度，就排除常规第二轮累积；若多次，则回到最终配置/数据派发定位差异。对同步、预取或 offload 做性能 A/B 应等待已定位的具体候选和实测显存预算，不在本轮盲试。
