# Ascend性能调优

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/model_train/optimize/profiler.md)

## 概述

本教程介绍如何在Ascend AI处理器上使用MindSpore Profiler进行性能调优。MindSpore Profiler可以为用户提供算子执行时间分析、内存使用分析、AI Core指标分析、Timeline展示等功能，帮助用户分析性能瓶颈、优化训练效率。

## 操作流程

1. 准备训练脚本；

2. 在训练脚本中调用性能调试接口，如mindspore.Profiler以及mindspore.profiler.DynamicProfilerMonitor接口；

3. 运行训练脚本；

4. 通过[MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/70RC2/msinsightug/msascendinsightug/AscendInsight_0002.html)软件查看性能数据。

## 使用方法

收集训练性能数据有三种方式，用户可以根据不同场景使用Profiler使能方式，以下将介绍不同场景的使用方式。

### 方式一：mindspore.Profiler接口使能

在训练脚本中添加MindSpore Profiler相关接口，Profiler接口详细介绍请参考[MindSpore Profiler参数详解](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Profiler.html)。

**Graph模式采集样例：**

**Graph**模式下，用户可以通过Callback方式来使能Profiler。

```python
import os
import mindspore as ms
from mindspore import Profiler

class StopAtStep(ms.Callback):
    def __init__(self, start_step, stop_step):
        super(StopAtStep, self).__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        self.profiler = Profiler(start_profile=False, output_path='./profiler_data')

    def on_train_step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step:
            self.profiler.start()

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.stop_step:
            self.profiler.stop()
            self.profiler.analyse()
```

完整案例请参考[graph模式采集完整代码样例](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/profiler/graph_start_stop_profiler.py)

**PyNative模式采集样例：**

**PyNative**模式下，用户可以通过设置schedule以及on_trace_ready参数来使能Profiler。

例如用户想要采集前两个step的性能数据，可以使用如下配置的schedule进行采集。

样例如下：

```python
import mindspore
from mindspore import Profiler
from mindspore.profiler import schedule, tensor_board_trace_handler

STEP_NUM = 15
# 定义训练模型网络
net = Net()
with Profiler(schedule=schedule(wait=0, warm_up=0, active=2, repeat=1, skip_first=0),
              on_trace_ready=tensor_board_trace_handler) as prof:
    for _ in range(STEP_NUM):
        train(net)
        # 调用step采集
        prof.step()
```

使能后落盘数据中kernel_details.csv中包含了Step ID一列信息，且Step ID为0,1，表示采集的是第0个step以及第1个step数据。

完整案例参考[PyNative模式采集完整代码样例](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/profiler/py_native_step_profiler.py)

### 方式二：动态profiler使能

用户如果想要在训练过程中不中断训练流程，修改配置文件，完成新配置下的采集任务，可以使用mindspore.profiler.DynamicProfilerMonitor接口使能，
该接口需要配置一个JSON配置文件，如不配置会生成一个默认配置的JSON文件。

JSON配置样例如下：

```json
{
   "start_step": 2,
   "stop_step": 5,
   "aicore_metrics": -1,
   "profiler_level": 0,
   "activities": 0,
   "analyse_mode": -1,
   "parallel_strategy": false,
   "with_stack": false,
   "data_simplification": true
}
```

1. 用户需要在实例化DynamicProfilerMonitor前配置如上的JSON配置文件，详细参数介绍请参考[DynamicProfilerMonitor参数详解](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.profiler.DynamicProfilerMonitor.html)，将配置文件保存在cfg_path中；
2. 在模型训练后调用DynamicProfilerMonitor的step接口采集数据；
3. 用户如果想在训练中变更采集、解析任务，可以去修改JSON配置文件，如变更上述JSON配置中的start_step为8，stop_step为10，保存后，DynamicProfilerMonitor会自动识别出配置文件变更成新的采集、解析任务。

样例如下：

```python
from mindspore.profiler import DynamicProfilerMonitor

# cfg_path中包括上述的json配置文件路径，output_path为输出路径
dp = DynamicProfilerMonitor(cfg_path="./cfg_path", output_path="./output_path")
STEP_NUM = 15
# 定义训练模型网络
net = Net()
for _ in range(STEP_NUM):
    train(net)
    # 调用step采集
    dp.step()
```

此时生成的结果文件包含两个文件夹：rank0_start2_stop5以及rank0_start8_stop10，分别代表采集的step为2-5和8-10。

完整案例请参考[动态Profiler使能方式案例](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/profiler/dynamic_profiler.py)。

### 方式三：环境变量使能

用户如果想最简单地使能Profiler，可以使用环境变量使能方式，该方式只需将参数配置到环境变量中，在模型训练中会自动采集性能数据，但该方式暂不支持
schedule参数方式采集数据，其他参数都可以使用。详细配置项介绍请参考[环境变量使能方式参数详解](https://www.mindspore.cn/docs/zh-CN/master/api_python/env_var_list.html)。
环境变量使能方式相关配置项，样例如下：

```shell
export MS_PROFILER_OPTIONS='
{"start": true,
"output_path": "/XXX",
"activities": ["CPU", "NPU"],
"with_stack": true,
"aicore_metrics": "AicoreNone",
"l2_cache": false,
"profiler_level": "Level0"}'
```

加载完环境变量后，直接拉起训练脚本即可完成采集。需要注意的是该配置中**start**必须为true，才能达到使能效果，否则使能不生效。

## 性能数据

用户通过MindSpore Profiler 采集、解析后的性能数据包括框架侧、CANN侧和device侧的原始性能数据，以及解析后的性能数据。

在使用MindSpore进行模型训练时，为了分析性能瓶颈、优化训练效率，我们需要收集并分析性能数据。MindSpore Profiler提供了完整的性能数据采集和分析能力，本文将详细介绍采集到的性能数据的存储结构和内容含义。

性能数据采集完成后，原始数据会按照以下目录结构进行存储：

> - 以下数据文件用户无需打开查看，可根据[MindStudio Insight用户指南](https://www.hiascend.com/document/detail/zh/mindstudio/70RC2/msinsightug/msascendinsightug/AscendInsight_0002.html)指导进行性能数据的查看和分析。
> - 以下是结果文件全集，实际文件数量和内容根据用户的参数配置以及实际的训练场景来生成，如果用户没有使能相关参数或是训练中没有涉及到相关场景，则不会生成对应的数据文件。  

```sh
└── localhost.localdomain_*_ascend_ms  // 解析结果目录，命名格式：{worker_name}_{时间戳}_ascend_ms，默认情况下{worker_name}为{hostname}_{pid}
    ├── profiler_info.json             // 多卡或集群场景命名规则为profiler_info_{Rank_ID}.json，用于记录Profiler相关的元数据
    ├── profiler_metadata.json
    ├── ASCEND_PROFILER_OUTPUT         // MindSpore Profiler接口采集性能数据
    │   ├── api_statistic.csv          // 配置profiler_level=ProfilerLevel.Level1或profiler_level=ProfilerLevel.Level2生成
    │   ├── communication.json         // 为多卡或集群等存在通信的场景性能分析提供可视化数据基础，配置profiler_level=ProfilerLevel.Level1或profiler_level=ProfilerLevel.Level2生成
    │   ├── communication_matrix.json  // 通信小算子基本信息文件，配置profiler_level=ProfilerLevel.Level1或profiler_level=ProfilerLevel.Level2生成
    │   ├── dataset.csv                // activities中配置ProfilerActivity.CPU生成
    │   ├── data_preprocess.csv        // 配置profiler_level=ProfilerLevel.Level2生成
    │   ├── kernel_details.csv         // activities中配置ProfilerActivity.NPU生成
    │   ├── l2_cache.csv               // 配置l2_cache=True生成
    │   ├── memory_record.csv          // 配置profile_memory=True生成
    │   ├── minddata_pipeline_raw_*.csv
    │   ├── minddata_pipeline_summary_*.csv
    │   ├── minddata_pipeline_summary_*.json
    │   ├── npu_module_mem.csv         // 配置profile_memory=True生成
    │   ├── operator_memory.csv        // 配置profile_memory=True生成
    │   ├── op_statistic.csv           // AI Core和AI CPU算子调用次数及耗时数据
    │   ├── step_trace_time.csv        // 迭代中计算和通信的时间统计
    │   └── trace_view.json
    ├── FRAMEWORK                      // 框架侧的性能原始数据，无需关注，data_simplification=True时删除此目录
    └── PROF_000001_20230628101435646_FKFLNPEPPRRCFCBA  // CANN层的性能数据，命名格式：PROF_{数字}_{时间戳}_{字符串}，data_simplification=True时，仅保留此目录下的原始性能数据，删除其他数据
          ├── analyze                  // 配置profiler_level=ProfilerLevel.Level1或profiler_level=ProfilerLevel.Level2生成
          ├── device_*
          ├── host
          ├── mindstudio_profiler_log
          └── mindstudio_profiler_output
```

MindSpore Profiler接口将框架侧的数据与CANN Profling的数据关联整合，形成trace、kernel以及memory等性能数据文件。各文件详细说明如下文所示。

`FRAMEWORK` 为框架侧的性能原始数据，无需关注；`PROF` 目录下为CANN Profling采集的性能数据，主要保存在 `mindstudio_profiler_output` 目录下。

### communication.json

该性能数据文件信息如下所示：

- hcom\_allGather\_\*@group
    - Communication Time Info
        - Start Timestamp\(μs\)
        - Elapse Time\(ms\)
        - Transit Time\(ms\)
        - Wait Time\(ms\)
        - Synchronization Time\(ms\)
        - Idel Time\(ms\)
        - Wait Time Ratio
        - Synchronization Time Ratio
    - Communication Bandwidth Info
        - RDMA
            - Transit Size\(MB\)
            - Transit Time\(ms\)
            - Bandwidth\(GB/s\)
            - Large Packet Ratio
            - Size Distribution
                - "Package Size\(MB\)": \[count, dur\]
        - HCCS
            - Transit Size\(MB\)
            - Transit Time\(ms\)
            - Bandwidth\(GB/s\)
            - Large Packet Ratio
            - Size Distribution
                - "Package Size\(MB\)": \[count, dur\]
        - PCIE
            - Transit Size\(MB\)
            - Transit Time\(ms\)
            - Bandwidth\(GB/s\)
            - Large Packet Ratio
            - Size Distribution
                - "Package Size\(MB\)": \[count, dur\]
        - SDMA
            - Transit Size\(MB\)
            - Transit Time\(ms\)
            - Bandwidth\(GB/s\)
            - Large Packet Ratio
            - Size Distribution
                - "Package Size\(MB\)": \[count, dur\]
        - SIO
            - Transit Size\(MB\)
            - Transit Time\(ms\)
            - Bandwidth\(GB/s\)
            - Large Packet Ratio
            - Size Distribution
                - "Package Size\(MB\)": \[count, dur\]

### communication_matrix.json

该性能数据文件信息样例如下所示：

- allgather\-top1@\*
    - src\_rank\-dst\_rank
        - Transport Type
        - Transit Size\(MB\)
        - Transit Time\(ms\)
        - Bandwidth\(GB/s\)
        - op_name

### dataset.csv

`dataset.csv` 文件记录dataset算子的信息。

| 字段名 | 字段解释 |
|----------|----------|
| Operation | 对应的数据集操作名称 |
| Stage | 操作所处的阶段 |
| Occurrences | 操作出现次数 |
| Avg. time(us) | 操作平均时间(微秒) |
| Custom Info | 自定义信息 |

### kernel_details.csv

`kernel_details.csv` 文件由 `ProfilerActivity.NPU` 开关控制，文件包含在NPU上执行的所有算子的信息，若用户前端调用了 `schedule` 进行 `step` 打点，则会增加 `Step Id` 字段。

与Ascend PyTorch Profiler接口采集数据结果的不同之处在于当 `with_stack` 开关开启之后，MindSpore Profiler会将堆栈信息拼接到 `Name` 字段中。

### trace_view.json

`trace_view.json` 建议使用 [MindStudio Insight工具](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/mindinsight_install.html) 或 chrome://tracing/ 打开。MindSpore Profiler暂时不支持record_shapes与GC功能。

### 其他性能数据

其他性能数据文件的具体字段与含义可以参考[昇腾官网文档](https://www.hiascend.com/document/detail/zh/mindstudio/70RC3/T&ITools/Profiling/atlasprofiling_16_0035.html)。

## 常见工具问题及解决办法

### 使用step采集性能数据常见问题

#### schedule配置错误问题

schedule配置相关参数有5个，wait、warm_up、active、repeat、skip_first。每个参数大小必须**大于等于0**；其中**active**必须**大于等于1**，否则抛出警告，并设置为默认值1；如果repeat设置为0，表示repeat参数不生效，Profiler会根据模型训练的次数来确定循环次数。

#### schedule与step配置不匹配问题

正常来说schedule的配置应小于模型训练的次数，即repeat*(wait+warm_up+active)+skip_first应小于模型训练的次数。如果schedule的配置大于模型训练的次数，Profiler会抛出异常警告，但这并不会打断模型训练，但可能存在采集解析的数据不全的情况。
