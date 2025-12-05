# 特征值检测

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/feature/atomgit/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/tutorials/source_zh_cn/debug/sdc.md)

## 概述

### 背景

模型训练过程中，处理器可能发生特征值检测异常，产生计算错误且无上报。特征值检测异常可能会造成对模型训练的严重负面影响。

### 解决方案

MindSpore框架2.7.1版本提供了特征值与CheckSum联合检测方案，能够更加准确地定位静默故障。该方案采样参数梯度进行特征值检测，并在检测到多次特征值异常时，通过“三振出局”机制触发CheckSum校验，进一步定位故障卡。用户可以通过`MS_NPU_ASD_CONFIG`对联合检测进行配置。

关于相关环境变量的配置，见 **特性开关及配置**。

关于默认的特征值检测点的介绍，以及对于自定义特征值检测点的设计指导，见 **使用建议与检测原理** 。

### 使用建议与检测原理

处理器发生特征值检测异常时，计算得出错误结果。由于 Transformer 模型的结构，错误的计算结果会传播开来。

通过对实验结果进行统计，作以下经验性总结。

* 并非所有的特征值检测异常都一定影响模型的收敛和性能，事实上，大部分特征值检测异常对模型不产生可观测影响。可见 [文献](https://dl.acm.org/doi/abs/10.1145/3579371.3589105)。
* 统计学意义上，反向传播计算过程中的特征值检测异常影响远大于正向计算过程中的影响。
* 在并行训练场景下，计算误差结果会由于并行计算而发生传播。
* 过多的检测点设置会影响模型训练性能。
* 根据计算错误检测敏感性实验结果，MindSpore框架默认选择反向传播计算过程中的`Norm`激活值梯度作为检测特征值，基于 **Llama 2 - 7B** 测试性能损失小于 2%。

采用特征值与CheckSum联合检测方案（`MS_NPU_ASD_CONFIG`中设置`enable:true`）时，会在反向图中对参数通信前的梯度进行特征值采样，并通过算法判断是否异常。当联合CheckSum校验（`MS_NPU_ASD_CONFIG`中设置`with_checksum:true`）时，若在时间窗口内异常次数超过阈值，会进一步开启CheckSum校验，对各卡bfloat16数据类型的MatMul算子的计算结果进行校验。

特征值异常原因可分为两类：硬件错误与软件错误，可参考**故障处理**章节进行后续分析。

### 使用限制

目前本特性仅支持Atlas A2 训练系列产品，仅支持检测8维以内Transformer类模型，bfloat16和float32数据类型，训练过程中出现的特征值检测异常。

联合检测方案目前仅支持自动并行或半自动并行模式。CheckSum仅针对bfloat16数据类型的MatMul算子进行校验。

## 特性开关及配置

环境变量`MS_NPU_ASD_CONFIG`对特征值和CheckSum联合检测方案进行配置，格式为key:value，并以逗号分隔各个配置项。其中`enable`为特征值检测开关，`with_checksum`为联动CheckSum开关，`grad_sample_interval`为特征值采样间隔，`upper_thresh1`和`upper_thresh2`分别控制特征值检测的绝对阈值和相对阈值，`cooldown`为特征值异常冷却时间和单次CheckSum执行时长，`strikes_num`和`strikes_window`为触发CheckSum所需的特征值异常次数和时间窗口大小，`checksum_cooldown`为CheckSum冷却时间。默认情况下，`MS_NPU_ASD_CONFIG="enable:false,with_checksum:false,grad_sample_interval:10,upper_thresh1:1000000,upper_thresh2:100,cooldown:5,strikes_num:3,strikes_window:480,checksum_cooldown:180"`。

上述环境变量的详细说明参见[环境变量](https://www.mindspore.cn/docs/zh-CN/master/api_python/env_var_list.html)。

## 使用用例

> 本文档介绍特征值检测的使用方法以及用例。

这里构造了一个简单的神经网络，并通过MindSpore的故障注入算子模拟特征值异常。网络脚本（`silent_detect.py`）如下：

```python
"""Silent Detect Demo"""
import time
import numpy as np

import mindspore as ms
from mindspore import nn, Tensor, Parameter, context, ops, jit
from mindspore.communication import init, get_rank
from mindspore.nn import Momentum, TrainOneStepCell
from mindspore.parallel.auto_parallel import AutoParallel

context.set_context(mode=context.GRAPH_MODE)
init()
ms.set_seed(1)
np.random.seed(1)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Dense(1, 8)
        self.fc2 = nn.Dense(8, 8)
        self.relu = ops.ReLU()
        self.eod_mask = ops.auto_generate.GenerateEodMaskV2()
        self.cur_step = Parameter(Tensor(-1, ms.int64), requires_grad=False)
        rank_id = get_rank()
        if rank_id == 2:
            self.flip_mode = 'bitflip_designed'
        else:
            self.flip_mode = 'multiply'

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        ele_pos = Tensor(0, ms.int64)
        seed = Tensor(0, ms.int64)
        offset = Tensor(0, ms.int64)
        start = 0
        steps = [5]
        error_mode = 'cycle'
        multiply_factor = 1.0
        bit_pos = 0
        flip_probability = 0.0
        self.cur_step = self.cur_step + 1
        x = self.eod_mask(x, ele_pos, self.cur_step, seed, offset, start, steps, error_mode, self.flip_mode,
                          multiply_factor, bit_pos, flip_probability)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    net = Net()
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    net = TrainOneStepCell(net, optimizer)
    net.set_train()

    @jit
    def compiled_one_step(inputs):
        net(inputs)

    parallel_net = AutoParallel(compiled_one_step, parallel_mode='semi_auto')
    for i in range(200):
        inputs = Tensor(np.random.rand(8, 1).astype(np.float32))
        parallel_net(inputs)
        time.sleep(1)
```

启动命令：

```bash
export MS_NPU_ASD_CONFIG="enable:true,with_checksum:true,grad_sample_interval:1,cooldown:1,strikes_num:1"
msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=11235 --join=True python silent_detect.py
```

通过查看训练日志（默认为`worker_*.log`），可以观察到特征值异常记录、CheckSum校验结果：

```bash
$ grep -m1 'Silent detect strike' worker_0.log
[WARNING] DEBUG(2950752,fffee7e591e0,python):2025-08-26-10:46:26.665.782 [mindspore/ccsrc/tools/silent_detect/silent_detector.cc:109] SilentDetect] Silent detect strike detected: StrikeRecord{timestamp: 1756176386, name: fc1.weight, value: inf, stat: StatData{avg: 6.44326e+12, pre_value: 6.441e+14, count: 6, none_zero_count: 6}}
$ grep -m1 'Global CheckSum result is' worker_0.log
[WARNING] DEBUG(2950752,fffda37fe1e0,python):2025-08-26-10:47:28.934.305 [mindspore/ccsrc/tools/silent_detect/silent_detector.cc:316] DoCheckSum] Global CheckSum result is 0
```

## 检测结果及处理

### 异常检测结果

未检测到数值异常时，对训练任务运行无影响。

当检测到数值异常后，训练任务失败并上报告警，请通过如下方法之一定位故障设备：

* 通过搜索应用类日志，查询**ERROR**级别错误日志，关键字"accuracy sensitivity feature abnormal"；
* 通过监控NPU健康状态：Health Status显示Warning，Error Code显示80818C00，Error Information显示node type=SoC, sensor type=Check Sensor, event state=check fail；
* 通过查看[MindCluster](https://gitcode.com/Ascend/mind-cluster)事件，上报错误码80818C00，事件类型为故障事件，故障级别次要。

当使用联合检测时，若训练中发生特征值异常、CheckSum检测出静默故障，会在业务训练日志中产生告警：

* 特征值异常日志关键字为“Silent detect strike”；
* 触发CheckSum校验日志关键字为“Feature value detection strikes out”；
* 联合CheckSum识别出静默故障日志关键字为“CheckSum detects MatMul error on rank”和“SilentCheck detects SDC error”。

### 故障处理

将异常设备隔离，断点续训拉起继续训练；同时在异常设备上，通过Ascend-DMI工具执行AICore ERROR压测诊断，检测该设备上是否存在故障NPU。详情请查看[《ToolBox用户指南》](https://www.hiascend.com/document/detail/zh/mindx-dl/600/toolbox/ascenddmi/toolboxug_000002.html) “ascend-dmi工具使用 > 故障诊断”章节。

若异常设备上检测到故障卡，请联系华为工程师维修更换；若异常设备上所有NPU均正常，则为软件类问题触发特征值溢出，建议排查程序和算子原因。