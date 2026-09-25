# VeOmni backward / Active Memory 诊断

本目录用于 `d71ae4c11935f8dd073644b05943a84ecd1a79b3` 的首轮审计。结论见 [audit.md](audit.md)，证据进展见 [findings.md](findings.md)。Windows 没有 NPU；这里的测试不是服务器训练验证。

## 先做这两步，不重跑训练

运行位置：**Linux 服务器当前仓库根目录、训练使用的 Conda 环境，沿用已加载的 CANN/ATB**。`/absolute/path/to/ONE_RUN` 替换为一个已有 profiling run 的真实目录；可以是包含 16 rank 的目录，不要指向全部历史 run 的总目录。每次分析使用新的结果目录。

```bash
python codex_test/veomni_memory/collect_env.py --output-dir codex_test/veomni_memory/results/env_first
python codex_test/veomni_memory/summarize_profile.py --profile-root /absolute/path/to/ONE_RUN --output-dir codex_test/veomni_memory/results/inventory_first --inventory-only --top-k 20
```

上述文件先留在服务器。环境报告不读取训练样本、权重值或完整环境变量，但包含软件实际导入路径、Git 状态和源码片段。**这个独立进程不能证明 Ray worker 的实际对象状态。** 已经回传过环境报告就无需重复采集。

## 回传大小受限：只导出一个最多 2 KiB 的文件

对已经生成的 `summary.json` 执行下面的命令，不重跑训练、不重扫 profiling、不导入 torch，也不联网：

```bash
python codex_test/veomni_memory/export_brief.py --summary codex_test/veomni_memory/results/inventory_first/summary.json --output codex_test/veomni_memory/results/brief.txt --max-bytes 2048
```

只需检查并回传 `brief.txt`，无需回传其他文件。上限按完整文件的 UTF-8 字节数计算（包含换行），默认 2048 字节，远低于 100 KB；命令会打印实际大小。支持 `--max-bytes` 调节，最小 512 字节。输出文件须不存在，重复导出时换一个文件名。工具合并重复 schema，仅保留文件类型统计、表名、列名/类型及出现次数，不输出文件路径、环境源码、事件值或错误原文；schema 名称仍须按公司要求检查。

每行是独立 JSON；`schema` 是本摘要中的编号，`offset` 是该组列的起始位置。空间不足时省略整行，末行 `schema_records_omitted` 明确计数；上游清单本身也可能已截断。该文件仅用于决定下一步 schema 适配，不能据此判断性能或根因。需要更多信息时再定向导出，勿把分片作为绕过总量限制的办法。

已知 schema 或存在 Chrome trace 时，可去掉 `--inventory-only`，使用另一个 `--output-dir`。生成的 `summary.md`、`summary.json`、CSV、`local_events.sqlite` 以及原始 DB/trace 均先留在服务器；不再要求整套回传。结果目录自动写入 `.gitignore`。

## 首轮解析能力和边界

仅需 Python 3.11+ 标准库，不需要 torch、torch_npu、pandas 或 ijson；不会联网安装依赖。

| 输入 | 首轮行为 |
|---|---|
| 任意文件名的 SQLite | 按文件头识别，`mode=ro` + `query_only`；探测表/列后只取映射字段，不执行 `SELECT *` 或全表 COUNT |
| 单 rank / cluster DB | 同一读取机制；单位明确的扁平表可读取。字符串 ID 表关联、版本特定的 HCCL 关系表需根据 schema 补适配 |
| CSV / CSV.gz | 逐行读取；识别单位明确的 `start_ns`、`start_time(us)` 等。裸时间戳不猜单位 |
| Chrome trace JSON / JSON.gz | 增量解析 `traceEvents` 或顶层数组；每个事件最多 4 MiB。保留 X 的 CPU inclusive time、设备候选事件；B/E 暂不重建，计入未支持计数 |
| communication JSON | 清单模式输出有界字段名样本。其嵌套统计不是执行时间轴；尚未提供版本适配时明确报告能力缺口 |
| 只有原始采集目录 | 文件大小清单；不自动启动 torch_npu/msprof 分析，不解析全部历史记录 |

默认摘要扫描时间和输入大小成正比，几十 GB 可能耗时数分钟以上；大部分事件写入服务器本地 `local_events.sqlite`，其大小可达输入的数倍，应选择有足够空间的目录。内存不持有完整 trace。返回清单最多 1000 文件、每 DB 200 表、每表 100 列，超限会报告；请缩小到一个 run。`--top-k` 上限 100；窗口最多 5 个、每窗口保留前 120 个邻近事件（明确截断），另保留前序完成和后续 collective 候选。短 `aten::empty` 也输出，作为反证候选。

`--step 2`、`--ranks 0,1,...,15` 是**严格过滤**：字段未知时排除并计数，不把目录名、device 或 pid 猜成 rank/step。首次清单不要加过滤。没有经核实的公共时钟、run、通信成员、全局序号、step 和数据量，就不报告跨 rank 偏斜。设备 collective 开始时间也不等于主机提交时间。不同文件的相对时间原点不混用。重复 rank 或缺成员的通信组不参与统计；Top-K 外的完整匹配组也用于最晚 rank 轮换统计。

输出保留原始时间及单位，换算纳秒使用 Decimal → 整数，避免 epoch ns 转 float 丢精度。没有 self time 就不推算；没有等待/传输字段就为 null。内存表的存活期不能充当 CPU 调用时长。内存差值仅在显式 allocator、device/process 和单位相符时计算；`MB` 按十进制，`MiB` 按二进制，实际报表若标成 MB 却使用二进制，需要在适配中注明。没有 action/address 时不能证明 FREE_REQUESTED → FREE_COMPLETED 的延迟，更不能把差值全部归因于 RS。

## schema-map：证据足够时启用分析

`--schema-map /server/path/reviewed_map.json` 显式映射数据库表、CSV 列或 trace 字段。字段名以 `summary.json` 的 schema 为准。下例是**合成格式示例，不是对你现有 DB 的假设**：

```json
{
  "sources": [{
    "glob": "rank*/profiler.db",
    "table": "DeviceCollectives",
    "columns": {
      "name": "op_name", "rank": "global_rank", "device": "device_id",
      "pid": "process_id", "stream": "stream_id", "step": "step",
      "micro_batch": "micro_batch", "start": "start_ns", "end": "end_ns",
      "group": "comm_group", "members": "members_json", "collective": "collective_type",
      "collective_seq": "global_sequence", "bytes": "input_bytes"
    },
    "constants": {
      "kind": "collective", "phase": "backward", "time_unit": "ns",
      "run": "ONE_CONFIRMED_RUN", "clock": "VERIFIED_COMMON_ORIGIN",
      "clock_verified": true, "sequence_verified": true
    }
  }]
}
```

只有核实了时钟/序号语义才能置两个 `verified=true`；本地 taskId 或名字后缀不满足要求。若序号会在 micro-batch 重置，必须映射 `micro_batch`，否则不得声明序号已验证。`phase` 也不能把整个表都盲标 backward。只需 mapping，不需上传任何 DB。CSV 不填 `table`；trace 通过 `args.rank` 这样的字段路径映射。计数曲线设 `kind=memory`，映射 `allocated/active/reserved`，显式给 `allocator`、`memory_unit`、`device/pid`。内存动作可映射 `action/address`，只接受实际提供的 `ALLOC/FREE_REQUESTED/FREE_COMPLETED` 语义。

如只有原始 profiling，先从 `environment.json` 确认 torch_npu 版本和实际路径。使用**同一训练环境**中的 `torch_npu.profiler.profiler.analyse`，先查看本机签名：

```bash
python -c 'import inspect, torch_npu; print(torch_npu.__version__); print(inspect.signature(torch_npu.profiler.profiler.analyse))'
```

若该版本支持 `profiler_path`，仅对清单中选择的一个采集目录调用 `torch_npu.profiler.profiler.analyse(profiler_path="/absolute/path/to/SELECTED_CAPTURE")`。先验证该目录的结果再决定是否解析同 run 其余 rank。cluster 分析使用服务器已安装、与 CANN 匹配的工具；具体 CLI 未获得环境证据前不编造。上述步骤会生成解析产物，不是首轮必需步骤。

## 可选 worker 内补采（当前不要求执行）

唯一生产代码接入是 `omni_impl.py::_build_model_optimizer()` 末尾四行，只有 `VEOMNI_MEMORY_PROBE=1` 才 import 并安装探针。默认不改变训练。`worker_probe.py` 在真实 engine 实例上记录：MRO 和方法来源、最终 engine 配置、实际 world/DP/SP/EP、两个 offload 标志、FSDP 组清单、本地 storage 去重统计、初始化前/后的 optimizer state 和每次真实 `forward_step` 数。全局 step 缺失时明确为 null，本地 train call 编号不冒充 step。显式保留冻结模块和 root 组；FP32 RS 字节预算标为条件上界。

诊断参数通过 `run_baseline.sh` 的 `ray_kwargs.ray_init.runtime_env.env_vars` 显式传入。只有每个 worker 的 `installed` 和 `train_batch` 记录，以及验证器无缺失，才能说明 Ray 子进程实际执行。脚本不能事先声称继承成功。

需要补采时，在服务器仓库根目录沿用模型/数据环境变量运行：

```bash
VEOMNI_MEMORY_PROBE=1 VEOMNI_MEMORY_PROBE_DIR="$PWD/codex_test/veomni_memory/results/worker_run1" bash codex_test/veomni_memory/run_baseline.sh
python codex_test/veomni_memory/worker_probe.py --worker-dir codex_test/veomni_memory/results/worker_run1 --expected-ranks 16
```

默认全部 rank，每 rank 前 3 次 train call；可通过 `VEOMNI_MEMORY_PROBE_RANKS`、`VEOMNI_MEMORY_PROBE_MAX_CALLS` 收窄。输出 `worker_rank*_pid*.jsonl` 和 `worker_validation.json`。该探针读取参数 metadata 并写少量 JSON，会有主机遍历/文件 I/O 扰动；不增加 `.item()`、tensor 内容格式化、设备 event、synchronize、barrier、empty_cache 或 collective。不会逐层刷盘。optimizer 计时只表示主机调用耗时。

`run_baseline.sh` 完整复现附件的 micro=2、3 steps、profile step=2 参数；末尾额外参数优先级最高。关闭探针即可回到原基线；移除 engine 末尾对应四行即可撤回接入，不使用 `git checkout` 覆盖其他修改。当前不提供无预算的 no-sync / 关闭 native offload 实验，也不恢复旧 `to()` override。

## Windows 验证与静态来源

在仓库根目录：

```powershell
& C:/Users/leovzhang/miniconda3/python.exe -m unittest discover -s codex_test/veomni_memory -p test_summary_on_cpu.py -v
```

测试仅用合成 SQLite/trace，验证纳秒精度、通信对应、缺列和只读约束。`collect_env.py --help`、`summarize_profile.py --help`、`worker_probe.py --help` 也可在 Windows 运行。不能在 Windows 运行训练基线。

`source_manifest.json` 保存审计依赖的仓库、不可变 revision 和 SHA256。`fetch_sources.py` 是可选的、显式联网下载公开源码的工具；服务器离线使用不需要它，不安装任何包。下载缓存 `_sources/` 被忽略，不替换环境中的依赖。PyTorch 2.10.0 和 Ascend 参考 snapshot 的机制解释必须由服务器版本报告核对，尤其不能把 Ascend 分支 snapshot 当作 post6 发布包。
