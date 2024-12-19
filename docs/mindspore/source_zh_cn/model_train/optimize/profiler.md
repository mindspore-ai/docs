# Ascend性能调优

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindspore/source_zh_cn/model_train/optimize/profiler.md)

## 概述

本教程介绍如何在Ascend AI处理器上使用MindSpore Profiler进行性能调优。MindSpore Profiler可以为用户提供算子执行时间分析、内存使用分析、AI Core指标分析、Timeline展示等功能，帮助用户分析性能瓶颈、优化训练效率。

## 操作流程

1. 准备训练脚本

2. 在训练脚本中调用性能调试接口，如mindspore.Profiler以及mindspore.profiler.DynamicProfilerMonitor接口

3. 运行训练脚本

4. 通过[MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/70RC2/msinsightug/msascendinsightug/AscendInsight_0002.html)软件查看性能数据

## 使用方法

收集训练性能数据有三种方式，用户可以根据不同场景使用Profiler使能方式，以下将介绍不同场景的使用方式。

### 方式一：修改训练脚本

在训练脚本中添加MindSpore Profiler相关接口，Profiler接口详细介绍请参考[MindSpore Profiler参数详解](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/mindspore/mindspore.Profiler.html?highlight=profiler#mindspore.Profiler)。

- 自定义训练

    MindSpore函数式编程用例使用Profiler进行自定义训练，可以在指定的step区间或者epoch区间开启或者关闭收集Profiler性能数据。

    ```python
    profiler = ms.Profiler(start_profile=False)
    data_loader = ds.create_dict_iterator()

    for i, data in enumerate(data_loader):
        train()
        if i==100:
            profiler.start()
        if i==200:
            profiler.stop()

    profiler.analyse()

    ```

- 自定义Callback

    对于数据非下沉模式，只有在每个step结束后才有机会告知CANN开启和停止，因此需要基于step开启和关闭。

    ```python
    import os
    import mindspore as ms
    from mindspore.communication import get_rank

    def get_real_rank():
        """get rank id"""
        try:
            return get_rank()
        except RuntimeError:
            return int(os.getenv("RANK_ID", "0"))

    class StopAtStep(ms.Callback):
        def __init__(self, start_step, stop_step):
            super(StopAtStep, self).__init__()
            self.start_step = start_step
            self.stop_step = stop_step
            # 按照rank_id设置性能数据落盘路径
            rank_id = get_real_rank()
            output_path = os.path.join("profiler_data", f"rank_{rank_id}")
            self.profiler = ms.Profiler(start_profile=False, output_path=output_path)

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

    对于数据下沉模式，只有在每个epoch结束后才有机会告知CANN开启和停止，因此需要基于epoch开启和关闭。可根据自定义Callback基于step开启Profiler样例代码修改训练脚本。

    ```python
    class StopAtEpoch(ms.Callback):
        def __init__(self, start_epoch, stop_epoch):
            super(StopAtEpoch, self).__init__()
            self.start_epoch = start_epoch
            self.stop_epoch = stop_epoch
            # 按照rank_id设置性能数据落盘路径
            rank_id = get_real_rank()
            output_path = os.path.join("profiler_data", f"rank_{rank_id}")
            self.profiler = ms.Profiler(start_profile=False, output_path=output_path)

        def on_train_epoch_begin(self, run_context):
            cb_params = run_context.original_args()
            epoch_num = cb_params.cur_epoch_num
            if epoch_num == self.start_epoch:
                self.profiler.start()

        def on_train_epoch_end(self, run_context):
            cb_params = run_context.original_args()
            epoch_num = cb_params.cur_epoch_num
            if epoch_num == self.stop_epoch:
                self.profiler.stop()
                self.profiler.analyse()
    ```

### 方式二：动态Profiler使能

mindspore.profiler.DynamicProfilerMonitor提供用户动态修改Profiler配置参数的能力，修改配置时无需中断训练流程，初始化生成的JSON配置文件示例如下。

```json
{
   "start_step": -1,
   "stop_step": -1,
   "aicore_metrics": -1,
   "profiler_level": -1,
   "profile_framework": -1,
   "analyse_mode": -1,
   "profile_communication": false,
   "parallel_strategy": false,
   "with_stack": false,
   "data_simplification": true
}
```

- start_step (int, 必选) - 设置Profiler开始采集的步数，为相对值，训练的第一步为1。默认值-1，表示在整个训练流程不会开始采集。

- stop_step (int, 必选) - 设置Profiler开始停止的步数，为相对值，训练的第一步为1，需要满足stop_step大于等于start_step。默认值-1，表示在整个训练流程不会开始采集。

- aicore_metrics (int, 可选) - 设置采集AI Core指标数据，取值范围与Profiler一致。默认值-1，表示不采集AI Core指标。

- profiler_level (int, 可选) - 设置采集性能数据级别，0代表ProfilerLevel.Level0，1代表ProfilerLevel.Level1，2代表ProfilerLevel.Level2。默认值-1，表示不控制性能数据采集级别。

- profile_framework (int, 可选) - 设置收集的host信息类别，0代表"all"，1代表"time"。默认值-1，表示不采集host信息。

- analyse_mode (int, 可选) - 设置在线解析的模式，对应mindspore.Profiler.analyse接口的analyse_mode参数，0代表"sync"，1代表"async"。默认值-1，表示不使用在线解析。

- profile_communication (bool, 可选) - 设置是否在多设备训练中采集通信性能数据，true代表采集，false代表不采集。默认值false，表示不采集集通信性能数据。

- parallel_strategy (bool, 可选) - 设置是否采集并行策略性能数据，true代表采集，false代表不采集。默认值false，表示不采集并行策略性能数据。

- with_stack (bool, 可选) - 设置是否采集调用栈信息，true代表采集，false代表不采集。默认值false，表示不采集调用栈。

- data_simplification (bool, 可选) - 设置开启数据精简，true代表开启，false代表不开启。默认值true，表示开启数据精简。

样例一：使用model.train进行网络训练，将DynamicProfilerMonitor注册到model.train。

步骤一：在训练代码中添加DynamicProfilerMonitor，将其注册到训练流程。

```python
import numpy as np
from mindspore import nn
from mindspore.train import Model
import mindspore as ms
import mindspore.dataset as ds
from mindspore.profiler import DynamicProfilerMonitor

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x):
        return self.fc(x)


def generator():
    for i in range(2):
        yield (np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32))


def train(net):
    optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    data = ds.GeneratorDataset(generator, ["data", "label"])

    # cfg_path参数为共享配置文件的文件夹路径，多机场景下需要满足此路径所有节点都能访问到
    # output_path参数为动态profile数据保存路径
    profile_callback = DynamicProfilerMonitor(cfg_path="./dyn_cfg", output_path="./dynprof_data")
    model = Model(net, loss, optimizer)
    model.train(10, data, callbacks=[profile_callback])


if __name__ == '__main__':
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

    # Train Mode
    net = Net()
    train(net)
```

步骤二：拉起训练流程，动态修改配置文件实现动态采集性能数据。拉起训练后，DynamicProfilerMonitor会在指定的cfg_path路径下生成配置文件profiler_config.json，用户可以动态编辑该配置文件，比如修改为下面的配置，表示DynamicProfilerMonitor将会在训练的第10个step开始采集，第10个step停止采集后在线解析。

```json
{
  "start_step": 10,
  "stop_step": 10,
  "aicore_metrics": -1,
  "profiler_level": -1,
  "profile_framework": -1,
  "analyse_mode": 0,
  "profile_communication": false,
  "parallel_strategy": true,
  "with_stack": true,
  "data_simplification": false
}
```

样例二：MindFormers中使用DynamicProfilerMonitor。  
步骤一：在MindFormers中添加DynamicProfilerMonitor，将其注册到训练流程。修改mindformers/trainer/trainer.py中的_build_profile_cb函数，将其默认的ProfileMonitor修改为DynamicProfilerMonitor，修改示例如下。

```python
def _build_profile_cb(self):
  """build profile callback from config."""
  if self.config.profile:
      sink_size = self.config.runner_config.sink_size
      sink_mode = self.config.runner_config.sink_mode
      if sink_mode:
          if self.config.profile_start_step % sink_size != 0:
              self.config.profile_start_step -= self.config.profile_start_step % sink_size
              self.config.profile_start_step = max(self.config.profile_start_step, sink_size)
              logger.warning("profile_start_step should divided by sink_size, \
                  set profile_start_step to %s", self.config.profile_start_step)
          if self.config.profile_stop_step % sink_size != 0:
              self.config.profile_stop_step += self.config.profile_stop_step % sink_size
              self.config.profile_stop_step = max(self.config.profile_stop_step, \
                  self.config.profile_start_step + sink_size)
              logger.warning("profile_stop_step should divided by sink_size, \
                  set profile_stop_step to %s", self.config.profile_stop_step)

      start_profile = self.config.init_start_profile
      profile_communication = self.config.profile_communication

      # 添加DynamicProfilerMonitor，替换原有的ProfileMonitor
      from mindspore.profiler import DynamicProfilerMonitor

      # cfg_path参数为共享配置文件的文件夹路径，多机场景下需要满足此路径所有节点都能访问到
      # output_path参数为动态profile数据保存路径
      profile_cb = DynamicProfilerMonitor(cfg_path="./dyn_cfg", output_path="./dynprof_data")

      # 原始的ProfileMonitor不再使用
      # profile_cb = ProfileMonitor(
      #     start_step=self.config.profile_start_step,
      #     stop_step=self.config.profile_stop_step,
      #     start_profile=start_profile,
      #     profile_communication=profile_communication,
      #     profile_memory=self.config.profile_memory,
      #     output_path=self.config.profile_output,
      #     config=self.config)
      self.config.auto_tune = False
      self.config.profile_cb = profile_cb
```

步骤二：在模型的yaml配置文件中开启profile功能后拉起训练，拉起训练后，DynamicProfilerMonitor会在指定的cfg_path路径下生成配置文件profiler_config.json，用户可以动态编辑该配置文件，比如修改为下面的配置，表示DynamicProfilerMonitor将会在训练的第10个step开始采集，第10个step停止采集后在线解析。

```json
{
  "start_step": 10,
  "stop_step": 10,
  "aicore_metrics": -1,
  "profiler_level": -1,
  "profile_framework": -1,
  "analyse_mode": 0,
  "profile_communication": false,
  "parallel_strategy": true,
  "with_stack": true,
  "data_simplification": false
}
```

### 方式三：环境变量使能

在运行网络脚本前，配置Profiler相关配置项。

```bash
export MS_PROFILER_OPTIONS='{"start": true, "output_path": "/XXX", "profile_memory": false, "profile_communication": false, "aicore_metrics": 0, "l2_cache": false}'

```

- start (bool，必选) - 设置为true，表示使能Profiler；设置成false，表示关闭性能数据收集，默认值：false。

- output_path (str, 可选) - 表示输出数据的路径（绝对路径）。默认值："./data"。

- op_time (bool, 可选) - 表示是否收集算子性能数据，默认值：true。

- profile_memory (bool，可选) - 表示是否收集Tensor内存数据。当值为true时，收集这些数据。使用此参数时，op_time 必须设置成true。默认值：false。

- profile_communication (bool, 可选) - 表示是否在多设备训练中收集通信性能数据。当值为true时，收集这些数据。在单台设备训练中，该参数的设置无效。使用此参数时，op_time 必须设置成true。默认值：false。

- aicore_metrics (int, 可选) - 设置AI Core指标类型，使用此参数时，op_time 必须设置成true。默认值：0。

- l2_cache (bool, 可选) - 设置是否收集l2缓存数据，默认值：false。

- timeline_limit (int, 可选) - 设置限制timeline文件存储上限大小（单位M），使用此参数时，op_time 必须设置成true。默认值：500。

- data_process (bool, 可选) - 表示是否收集数据准备性能数据，默认值：false。

- parallel_strategy (bool, 可选) - 表示是否收集并行策略性能数据，默认值：false。

- profile_framework (str, 可选) - 是否需要收集Host侧时间，可选参数为["all", "time", null]。默认值：null。

- with_stack (bool, 可选) - 是否收集Python侧的调用栈的数据，此数据在timeline中采用火焰图的形式呈现，使用此参数时， op_time 必须设置成 true 。默认值： false。

## 离线解析

当Profiler采集性能数据较大时，若在训练过程中直接使用Profiler.analyse()进行在线解析，则可能导致对系统资源占用过大，从而影响训练效率。Profiler提供了离线解析功能，支持采集完成性能数据后，使用Profiler.offline_analyse对采集数据进行离线解析。

训练脚本采集性能数据且不在线解析的部分代码示例如下：

```python
class Net(nn.Cell):
    ...


def train(net):
    ...


if __name__ == '__main__':
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

    # Init Profiler
    # Note that the Profiler should be initialized before model.train
    profiler = ms.Profiler(output_path='/path/to/profiler_data')

    # Train Model
    net = Net()
    train(net)  # Error occur.

    # Collection end
    profiler.stop()
```

在上述代码采集性能数据后，可以用离线接口来解析数据，示例代码如下：

```python
from mindspore import Profiler

Profiler.offline_analyse(path='/path/to/profiler_data', pretty=False, step_list=None, data_simplification=True)
```

离线解析接口参数描述如下：

- path (str) - 需要进行离线分析的profiling数据路径，指定到profiler上层目录。支持传入单卡和多卡数据路径。

- pretty (bool, 可选) - 对json文件进行格式化处理。此参数默认值为 False，即不进行格式化。

- step_list (list, 可选) - 只分析指定step的性能数据。此参数默认值为 None，即进行全解析。

- data_simplification (bool, 可选) - 数据精简开关功能。默认值为 True，即开启数据精简。

参数注意事项：

- step_list参数只在解析graph模式的采集数据时生效，且指定的step必须连续，step范围是从1开始计数的实际采集步数。例如：采集了5个step，则可选范围为[1,2,3,4,5]。

- data_simplification参数默认开启，若连续两次离线解析均打开该开关，第一次数据精简会将框架侧采集数据删除，进而导致第二次离线解析框架侧解析结果缺失。

离线解析传入的path路径支持单卡和多卡数据路径，不同场景描述如下。

### 单卡场景

采用离线解析解析单卡数据时，传入的profiling数据路径/path/to/profiler_data的目录结构如下：

```shell
└──── profiler_data
    └────profiler
```

解析的性能数据在/path/to/profiler_data/profiler目录下生成。

### 多卡场景

采用离线解析解析多卡数据时，传入的profiling数据路径/path/to/profiler_data的目录结构如下：

```shell
└──── profiler_data
    ├────rank_0
    │   └────profiler
    ├────rank_1
    │   └────profiler
    ├────rank_2
    │   └────profiler
    └────rank_3
        └────profiler
```

解析的性能数据在/path/to/profiler_data/profiler目录下生成。

### 目录结构

性能数据目录结构例如下：

```shell
└──── profiler
    ├──── container
    ├──── FRAMEWORK      // 框架侧采集的原始数据
    │   └──── op_range_*
    ├──── PROF_{数字}_{时间戳}_{字符串}       // msprof性能数据
    │   ├──── analyse
    │   ├──── device_*
    │   ├──── host
    │   ├──── mindstudio_profiler_log
    │   └──── mindstudio_profiler_output
    ├──── rank_* // 内存相关的原始数据
    │   ├──── memory_block.csv
    │   └──── task.csv
    ├──── rank-*_{时间戳}_ascend_ms      // MindStudio Insight可视化交付件
    │   ├──── ASCEND_PROFILER_OUTPUT      // MindSpore Profiler接口采集的性能数据
    │   ├──── profiler_info_*.json
    │   └──── profiler_metadata.json      // 记录用户自定义的meta数据，调用add_metadata或add_metadata_json接口生成该文件
    ├──── aicore_intermediate_*_detail.csv
    ├──── aicore_intermediate_*_type.csv
    ├──── aicpu_intermediate_*.csv
    ├──── ascend_cluster_analyse_model-{mode}_{stage_num}_{rank_size}_*.csv
    ├──── ascend_timeline_display_*.json
    ├──── ascend_timeline_summary_*.json
    ├──── cpu_framework_*.txt      // 异构场景生成
    ├──── cpu_ms_memory_record_*.txt
    ├──── cpu_op_detail_info_*.csv      // 异构场景生成
    ├──── cpu_op_execute_timestamp_*.txt      // 异构场景生成
    ├──── cpu_op_type_info_*.csv      // 异构场景生成
    ├──── dataset_iterator_profiling_*.txt      // 数据非下沉场景生成
    ├──── device_queue_profiling_*.txt      // 数据下沉场景生成
    ├──── dynamic_shape_info_*.json
    ├──── flops_*.txt
    ├──── flops_summary_*.json
    ├──── framework_raw_*.csv
    ├──── hccl_raw_*.csv      // 配置profiler(profiler_communication=True)生成
    ├──── minddata_aicpu_*.json      // 数据下沉场景生成
    ├──── minddata_cpu_utilization_*.json
    ├──── minddata_pipeline_raw_*.csv
    ├──── minddata_pipeline_summary_*.csv
    ├──── minddata_pipeline_summary_*.json
    ├──── operator_memory_*.csv
    ├──── output_timeline_data_*.txt
    ├──── parallel_strategy_*.json
    ├──── pipeline_profiling_*.json
    ├──── profiler_info_*.json
    ├──── step_trace_point_info_*.json
    └──── step_trace_raw_*_detail_time.csv
    └──── dataset_*.csv
```

### 性能数据文件描述

PROF_XXX目录下为CANN Profiling采集的性能数据，主要保存在mindstudio_profiler_output中，数据介绍在 [昇腾社区官网](https://www.hiascend.com/zh) 搜索"性能数据文件参考"查看。

profiler目录下包含csv、json、txt三类文件，覆盖了算子执行时间、内存占用、通信等方面的性能数据，文件说明见下表。

| 文件名 | 说明 |
|--------|------|
| step_trace_point_info_.json | step节点对应的算子信息（仅mode=GRAPH,export GRAPH_OP_RUM=0） |
| step_trace_raw__detail_time.csv | 每个step的节点的时间信息（仅mode=GRAPH,export GRAPH_OP_RUM=0） |
| dynamic_shape_info_.json | 动态shape下算子信息 |
| pipeline_profiling_.json | MindSpore数据处理，采集落盘的中间文件，用户无需关注 |
| minddata_pipeline_raw_.csv | MindSpore数据处理，采集落盘的中间文件，用户无需关注 |
| minddata_pipeline_summary_.csv | MindSpore数据处理，采集落盘的中间文件，用户无需关注 |
| minddata_pipeline_summary_.json | MindSpore数据处理，采集落盘的中间文件，用户无需关注 |
| framework_raw_.csv | MindSpore数据处理中AI Core算子的信息 |
| device_queue_profiling_.txt | MindSpore数据处理，采集落盘的中间文件，用户无需关注（仅数据下沉场景） |
| minddata_aicpu_.txt | MindSpore数据处理中AI CPU算子的性能数据（仅数据下沉场景） |
| dataset_iterator_profiling_.txt | MindSpore数据处理，采集落盘的中间文件，用户无需关注（仅数据非下沉场景） |
| aicore_intermediate__detail.csv | AI Core算子数据 |
| aicore_intermediate__type.csv | AI Core算子调用次数和耗时统计 |
| aicpu_intermediate_.csv | AI CPU算子信息解析后耗时数据 |
| flops_.txt | 记录AI Core算子的浮点计算次数（FLOPs）、每秒的浮点计算次数（FLOPS） |
| flops_summary_.json | 记录所有算子的总的FLOPs、所有算子的平均FLOPs、平均的FLOPS_Utilization |
| ascend_timeline_display_.json | timeline可视化文件，用于MindStudio Insight可视化 |
| ascend_timeline_summary_.json | timeline统计数据 |
| output_timeline_data_.txt | 算子timeline数据，只有AI Core算子数据存在时才有 |
| cpu_ms_memory_record_.txt | 内存profiling的原始文件 |
| operator_memory_.csv | 算子级内存信息 |
| minddata_cpu_utilization_.json | CPU利用率 |
| cpu_op_detail_info_.csv | CPU算子耗时数据（仅mode=GRAPH） |
| cpu_op_type_info_.csv | 具体类别CPU算子耗时统计（仅mode=GRAPH） |
| cpu_op_execute_timestamp_.txt | CPU算子执行起始时间与耗时（仅mode=GRAPH） |
| cpu_framework_.txt | 异构场景下CPU算子耗时（仅mode=GRAPH） |
| ascend_cluster_analyse_model-xxx.csv | 在模型并行或pipeline并行模式下，计算和通信等相关数据（仅mode=GRAPH） |
| hccl_raw_.csv | 基于卡的通信时间和通信等待时间（仅mode=GRAPH） |
| parallel_strategy_.json | 算子并行策略，采集落盘中间文件，用户无需关注 |
| profiler_info_.json | Profiler配置等info信息 |
| dataset_.csv | 数据处理模块各阶段执行耗时（要收集这部分数据，需要从最开始就开启profiler，至少是第一个step前） |

profiler目录下包括一些csv、json、txt文件，这些文件包含了模型计算过程中算子执行时间、内存占用、通信等性能数据，帮助用户分析性能瓶颈。下面对部分csv、txt文件中的字段进行说明，文件内容主要包括device侧算子（AI Core算子和AI CPU算子）耗时的信息、算子级内存和应用级内存占用的信息。

#### aicore_intermediate_*_detail.csv文件说明

aicore_intermediate_\*_detail.csv文件包含基于output_timeline_data_\*.txt和framework_raw_\*.csv中的内容，统计AI Core算子信息。文件中的字段说明参考下表：

| 字段名                     | 字段说明 |
|----------------------------|----------------------------|
|full_kernel_name            |device侧执行kernel算子全名|
|task_duration               |算子执行用时|
|execution_frequency         |算子执行频次|
|task_type                   |算子的任务类型|

#### aicore_intermediate_*_type.csv文件说明

aicore_intermediate_\*_type.csv文件包括基于output_timeline_data_\*.txt和framework_raw_\*.csv中的内容，统计AI Core算子具体类型的信息。文件中的字段说明参考下表：

|  字段名                      | 字段说明 |
|------------------------------|------------------------|
|  kernel_type                 | AI Core算子类型|
|  task_time                   | 该类型算子总用时|
|  execution_frequency         | 该类型算子执行频次|
|  percent                     | 该算子类型的用时的占所有算子总用时的百分比|

#### aicpu_intermediate_*.csv文件说明

aicpu_intermediate_\*.csv文件包含AI CPU算子的耗时信息。文件中的字段说明参考下表：

|  字段名                       | 字段说明 |
|------------------------------|------------------------|
|  serial_num                  | AI CPU算子序号|
|  kernel_type                 | AI CPU算子类型|
|  total_time                  | 算子耗时，等于下发耗时和执行耗时之和|
|  dispatch_time               | 下发耗时|
|  execution_time              | 执行耗时|
|  run_start                   | 算子执行起始时间|
|  run_end                     | 算子执行结束时间|

#### flops_*.txt文件说明

flops_\*.txt文件包含device侧算子的浮点计算次数、每秒浮点计算次数等信息。文件中的字段说明参考下表：

|  字段名                      | 字段说明 |
|------------------------------|------------------------|
|  full_kernel_name            | device侧执行kernel算子全名|
|  MFLOPs(10^6 cube)           | 浮点计算次数(10^6 cube)|
|  GFLOPS(10^9 cube)           | 每秒浮点计算次数(10^9 cube)|
|  MFLOPs(10^6 vector)         | 浮点计算次数(10^6 vector)|
|  GFLOPS(10^9 vector)         | 每秒浮点计算次数(10^9 vector)|

#### output_timeline_data_*.txt文件说明

output_timeline_data_\*.txt文件包括device侧算子的耗时信息。文件中的字段说明参考下表：

|  字段名                       | 字段说明 |
|------------------------------|------------------------|
|  kernel_name                 | device侧执行kernel算子全名|
|  stream_id                   | 算子所处Stream ID|
|  start_time                  | 算子执行开始时间(us)|
|  duration                    | 算子执行用时(ms)|

#### cpu_ms_memory_record_*.txt文件说明

cpu_ms_memory_record_\*.txt文件包含应用级内存占用的信息。文件中的字段说明参考下表：

|  字段名                       | 字段说明 |
|------------------------------|------------------------|
|  Timestamp                   | 内存事件发生时刻(ns)|
|  Total Allocated             | 内存分配总额(Byte)|
|  Total Reserved              | 内存预留总额(Byte)|
|  Total Active                | MindSpore中的流申请的总内存(Byte)|

#### operator_memory_*.csv文件说明

operator_memory_\*.csv文件包含算子级内存占用的信息。文件中的字段说明参考下表：

|  字段名                      | 字段说明 |
|------------------------------|------------------------|
|  Name                        | 内存占用Tensor名|
|  Size                        | 占用内存大小(KB)|
|  Allocation Time             | Tensor内存分配时间(us)|
|  Duration                    | Tensor内存占用时间(us)|
|  Allocation Total Allocated  | 算子内存分配时的内存分配总额(MB)|
|  Allocation Total Reserved   | 算子内存分配时的内存占用总额(MB)|
|  Release Total Allocated     | 算子内存释放时的内存分配总额(MB)|
|  Release Total Reserved      | 算子内存释放时的内存占用总额(MB)|
|  Device                      | device类型|
