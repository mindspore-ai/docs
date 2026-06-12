# 日志

MindSpore Transformers 动态图（PyNative）训练通过 `scripts/msrun_launcher.sh` 拉起，运行时产生的所有日志均通过 `msrun` 的 worker 进程日志统一收集。理解日志的目录结构与关键字段，是定位「训练为什么失败了 / loss 为什么不对 / 哪张卡出了问题」的前提。

本页介绍日志的用途与目录结构、`msrun` 的可配置项、训练日志中关键字段的含义，以及排查问题时的查看顺序。任务启动方式见 [启动任务](./start_tasks.md)。

## 日志总览

PyNative 训练运行时，所有日志统一由 `msrun` worker 进程日志承载：

- **`msrun` worker 进程日志**：由启动脚本 `scripts/msrun_launcher.sh` 通过 `LOG_DIR` 参数（默认 `output/msrun_log`）传递给底层 `msrun` 的 `--log_dir`。`msrun` 把每个 worker（设备）进程的**标准输出（stdout）与标准错误（stderr）整体重定向**到一份文件，命名为 `worker_{i}.log`（另有调度进程 `scheduler.log`）。框架 `logger` 的 `to_std=True`（默认开启）会将所有日志镜像到标准输出，因此**逐 step 的训练指标（loss、lr、grad_norm 等）、运行模式 banner、并行切分、数据集/权重加载等框架日志，以及 Python traceback、C++/HCCL 底层报错、segfault 退出信息等，全部落在 `worker_{i}.log` 中**。进程崩溃、组网失败等问题也主要靠它定位。

> **排查入口**
>
> - **进程没起来 / 中途崩溃 / 报底层错**：看对应 rank 的 `worker_{i}.log`（含完整 traceback 与 HCCL/驱动报错）。
> - **训练能跑，但要看 loss/lr/收敛/单步耗时**：同样看 `worker_{i}.log`，框架按 step 打印的指标都在里面。
> - **怀疑某张卡异常**：对照该 rank 的 `worker_{i}.log` 与 `scheduler.log`。

---

## 目录结构示意

默认配置下（`msrun_launcher.sh` 的 `LOG_DIR` 默认为 `output/msrun_log`），日志落盘结构如下：

```text
output/
└── msrun_log/                  # msrun 的 --log_dir，收集 worker 进程日志
    ├── scheduler.log           # 调度进程日志（组网、进程拉起/回收）
    ├── worker_0.log            # rank 0 进程的 stdout/stderr
    ├── worker_1.log            # rank 1 进程的 stdout/stderr
    └── ...                     # 每个 worker 一份，worker_{i}.log
```

> **文件名与路径来源**
>
> - `worker_{i}.log` / `scheduler.log`：由 `msrun` 在 `--log_dir` 下生成并按 worker/scheduler 命名（属 msrun 行为，非 mindformers 定义；参见 [启动任务](./start_tasks.md)）。
> - 日志目录由 `msrun_launcher.sh` 的 `LOG_DIR` 参数控制，默认 `output/msrun_log`。

---

## msrun 日志

`msrun` 在 `--log_dir` 指定的目录下，为每个 worker 进程生成一份 `worker_{i}.log`，把该进程的标准输出与标准错误整体写入其中；同时生成一份 `scheduler.log` 记录组网与进程管理。推荐通过 `scripts/msrun_launcher.sh` 启动，由 `LOG_DIR` 参数控制日志目录。

### 单机启动

单机 8 卡：

```bash
bash scripts/msrun_launcher.sh \
  "run_mindformer.py --config /path/to/pretrain_xxx.yaml --mode 1" \
  8 8 8118 output/msrun_log False 300
```

启动后，`output/msrun_log/` 下会出现 `worker_0.log` ~ `worker_7.log` 与 `scheduler.log`。

### 多机启动

多机场景下，每个节点各自执行 `msrun_launcher.sh`，各节点的 worker 日志落在本节点的 `LOG_DIR` 下（不会自动汇聚到主节点）。以 2 机 16 卡（每机 8 卡，主节点 IP `192.168.1.1`）为例：

主节点（节点 0）：

```bash
bash scripts/msrun_launcher.sh \
  "run_mindformer.py --config /path/to/pretrain_xxx.yaml --mode 1" \
  16 8 192.168.1.1 8118 0 output/msrun_log False 300
```

从节点（节点 1）：命令与主节点基本相同，只需将节点序号参数从 `0` 改为 `1`（即 `msrun_launcher.sh` 的第 7 个位置参数），其余参数保持一致；该节点的 `worker_{i}.log` 落在节点 1 自己的 `output/msrun_log/` 下。

> **多机排查提示**
>
> 多机问题需要登录到对应节点查看该节点的 `LOG_DIR`。例如怀疑全局 rank 10（节点 1 上的本地 rank 2）异常时，应到节点 1 的 `msrun_log` 目录查看其 `worker_10.log`。

`msrun_launcher.sh` 各参数（`WORKER_NUM`、`LOCAL_WORKER`、`MASTER_ADDR`、`MASTER_PORT`、`NODE_RANK`、`LOG_DIR`、`JOIN`、`CLUSTER_TIME_OUT`）的完整说明见 [启动任务](./start_tasks.md)。

---

## 日志中的关键字段

PyNative 训练循环每隔若干步，由 `LossCallback` 回调（`mindformers/pynative/callback/loss_callback.py` 的 `_print_log`）打印一行训练指标到标准输出，进而被 `msrun` 收集到 `worker_{i}.log` 中。一条典型记录形如：

```text
[INFO] 2026-06-09 10:20:30 [.../loss_callback.py:231] _print_log: { step:[  100/ 1000], loss:   2.345678, per_step_time:    850ms, lr: 1.000000e-04, grad_norm:   1.234000, throughput:  12.34T }
```

逐字段含义如下：

| 字段 | 示例 | 含义 |
|---|---|---|
| `step:[ 100/ 1000]` | 当前 step / 总 step | 训练所处的步数进度（全局步 / `training.steps`）。 |
| `loss: 2.345678` | 当前步 loss | 本步训练 loss（单值），用于判断收敛趋势。 |
| `per_step_time: 850ms` | 单步耗时 | 本步训练耗时，关注性能与抖动。 |
| `lr: 1.000000e-04` | 当前学习率 | 反映 warmup 与衰减调度；若学习率调度器不支持实时获取当前学习率，则该字段不打印。 |
| `grad_norm: 1.234000` | 梯度全局范数 | 全部参数梯度的全局 L2 范数。结合梯度裁剪阈值判断训练稳定性；若该步未取到则打印 `grad_norm: NaN`。 |
| `throughput: 12.34T` | 吞吐 | 本步训练吞吐（单位 `T`）。 |

> **补充说明**
>
> 逐 step 行由动态图的 `LossCallback._print_log` 拼接，字段名是 **`grad_norm`**（不是 `global_norm`），且 **没有** `Epoch`、`loss_scale`、`overflow cond` 等字段，`loss` 为单值（非「本步/滑动均值」两段式）。MoE / MTP 模型在开启分项 loss 时，行内会额外追加 `load_balancing_loss`、`mtp_{i}_loss`（如 `mtp_1_loss`）等字段。
>
> 如果训练任务开启了 PP（流水线并行），loss 将只在最后一个 stage 上显示。如 8 卡任务开启 PP 2 进行训练，则 rank_0~3 为 stage0，对应的日志 `worker_0~3.log` 中无 loss 信息，需要在 stage1（rank_4~7）的日志中查看 loss 信息（`worker_4~7.log`）。
>
> - 需要逐参数 / 逐 micro-step 的 `local_norm`、`local_loss`，或 MoE 的 tokens-per-expert 等更细指标时，通过 `monitor` 配置开启（动态图下这些指标输出到训练日志，暂不写入 TensorBoard）。详见 [训练指标监控与 Profiling](./monitor.md)。
> - 确认是否运行在动态图：启动阶段会打印 banner `Running MindFormers in PYNATIVE_MODE.`（来源 `run_mindformer.py`，对应 `--mode 1`）以确认进入动态图；未出现该 banner 则说明未以 `--mode 1` 启动。

---

## 排查建议

按以下顺序定位问题，通常最省时：

- **先看 `scheduler.log`**：`msrun_log/scheduler.log` 记录了集群组网全过程（worker 注册、拓扑构建、集群初始化、worker 注销），以及各 worker 的超时或异常退出信息（如 `Node X is timed out, please check this node's log`），是排查组网失败和进程异常退出的入口。
- **进程崩溃 / 底层报错 → 看对应 rank 的 `worker_{i}.log`**：含完整 Python traceback、HCCL/驱动等底层报错与退出信息。
- **怀疑单卡异常 → 看对应 rank 的 `worker_{i}.log`**：对比该 rank 的日志与其他正常 rank 的日志，定位差异。
- **通信 / 组网问题 → 看 `scheduler.log` 与各 `worker_{i}.log`**：组网建立、进程拉起与回收的信息在 `scheduler.log`；HCCL 报错通常分散在各 worker 日志中，需跨 rank 对照。
- **多机问题 → 登录对应节点查看本机 `LOG_DIR`**：各节点日志不汇聚，须到出问题的节点上查看其 `msrun_log`。

---

## 相关文档

- 启动任务与 `msrun` 参数：[启动任务](./start_tasks.md)
- 更细粒度指标与 Profiling：[训练指标监控与 Profiling](./monitor.md)
- 配置文件总览：[配置文件说明](./configuration.md)
- 并行维度与 rank 划分：[分布式并行训练](./parallel_training.md)
- 端到端训练流程：[训练](../guide/training.md)
