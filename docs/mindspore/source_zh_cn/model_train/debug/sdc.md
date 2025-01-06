# 特征值检测

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.1/docs/mindspore/source_zh_cn/model_train/debug/sdc.md)

## 概述

### 背景

模型训练过程中，处理器可能发生特征值检测异常，产生计算错误且无上报。特征值检测异常可能会造成对模型训练的严重负面影响。

### 解决方案

MindSpore框架2.4版本提供了网络模型的特征值检测方案，该方案主要是在反向图的通信算子前插入特征值检测算子，监测异常值并防止异常值扩散到其他卡。

对于默认的特征值检测点，用户可以设置环境变量 `NPU_ASD_ENABLE` 为`1`、`2`或`3`使能检测能力，并且通过配置环境变量 `NPU_ASD_UPPER_THRESH`, `NPU_ASD_SIGMA_THRESH`，调整检测强度。

关于相关环境变量的配置，见 **特性开关及配置**。

关于默认的特征值检测点的介绍，以及对于自定义特征值检测点的设计指导，见 **使用建议与检测原理** 。

### 使用建议与检测原理

处理器发生特征值检测异常时，计算得出错误结果。由于 Transformer 模型的结构，错误的计算结果会传播开来。

通过对实验结果进行统计，作以下经验性总结。

* 并非所有的特征值检测异常都一定影响模型的收敛和性能，事实上，大部分特征值检测异常对模型不产生可观测影响。 可见 [文献](https://dl.acm.org/doi/abs/10.1145/3579371.3589105)。
* 统计学意义上，反向传播计算过程中的特征值检测异常影响远大于正向计算过程中的影响。
* 在并行训练场景下，计算误差结果会由于并行计算而发生传播。
* 过多的检测点设置会影响模型训练性能。
* 根据计算错误检测敏感性实验结果，MindSpore框架默认选择反向传播计算过程中的`Norm`激活值梯度作为检测特征值，基于 **Llama 2 - 7B** 测试性能损失小于 2%。

开启检测开关（设置`NPU_ASD_ENABLE`为`1`，`2`或`3`）后，针对Transformer结构模型训练的反向阶段，通过在反向图的通信算子前插入检测算子，采集Norm层的激活值梯度，并通过算法判断是否异常。若出现异常，则根据环境变量`NPU_ASD_ENABLE`的不同取值打印相关日志或终止训练，并将检测到异常的设备上的NPU状态置为Warning，上报故障事件。

特征值异常原因可分为两类：硬件错误与软件错误，可参考**故障处理**章节进行后续分析。

### 使用限制

目前本特性仅支持Atlas A2 训练系列产品，仅支持检测Transformer类模型，bfloat16和float32数据类型，训练过程中出现的特征值检测异常。

## 特性开关及配置

环境变量`NPU_ASD_ENABLE`作为特性开关，`export NPU_ASD_ENABLE=1`、`export NPU_ASD_ENABLE=2`或`export NPU_ASD_ENABLE=3`开启本特性；不配置该环境变量或`export NPU_ASD_ENABLE=0`关闭本特性。

环境变量`NPU_ASD_UPPER_THRESH`控制检测的绝对数值阈值，格式为整型数据对，其中第一个元素控制绝对数值一级阈值，第二个元素控制绝对数值二级阈值；减小阈值可以检出波动更小的异常数据，增加检出率，增大阈值与之相反。在不配置该环境变量的默认情况下，`NPU_ASD_UPPER_THRESH=1000000,10000`。

环境变量`NPU_ASD_SIGMA_THRESH`控制检测的相对数值阈值，格式与上者相同，其中第一个元素控制数值跳变一级阈值，第二个元素控制数值跳变二级阈值；默认情况下，`NPU_ASD_SIGMA_THRESH=100000,5000`。

上述环境变量的详细说明参见[环境变量](https://www.mindspore.cn/docs/zh-CN/r2.4.1/api_python/env_var_list.html)。

## 使用用例

> 本文档介绍特征值检测的使用方法以及用例。

### 模型与数据集准备

为了提供完整的体验，这里构造了简单的神经网络和一个模拟数据集，并通过MindSpore的故障注入算子模拟特征值异常（实际网络中不需要该步骤）来展示特征值检测的使用方法。

完整的脚本(`silent_check.py`)如下：

```python
"""Silent Check Demo"""

import os
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops
from mindspore.communication import init
from mindspore.common.initializer import initializer


ms.set_context(mode=ms.GRAPH_MODE)
ms.set_context(max_device_memory="2GB")
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
init()
ms.set_seed(1)
np.random.seed(1)

class Network(nn.Cell):
    """Network"""
    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.fc1_weight = ms.Parameter(initializer("normal", [28*28, 512], ms.float32))
        self.fc2_weight = ms.Parameter(initializer("normal", [512, 512], ms.float32))
        self.fc3_weight = ms.Parameter(initializer("normal", [512, 10], ms.float32))
        self.matmul1 = ops.MatMul()
        self.relu1 = ops.ReLU()
        self.matmul2 = ops.MatMul()
        self.relu2 = ops.ReLU()
        self.matmul3 = ops.MatMul()
        # ====== begin ====== operator and parameter for injecting fault ==========================
        self.eod_mask = ops.auto_generate.GenerateEodMaskV2()
        self.cur_step = ms.Parameter(ms.Tensor(-1, ms.int64), requires_grad=False)
        rank_id = os.environ['RANK_ID']
        print(f'rank id of process {os.getpid()} is {rank_id}')
        if rank_id == '2':
            self.flip_mode = 'bitflip_designed' # bitflip, bitflip_designed, multiply, multiply_max
        else:
            self.flip_mode = 'multiply' # bitflip, bitflip_designed, multiply, multiply_max
        # ====== *end* ====== operator and parameter for injecting fault ==========================

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.fc1_weight)
        # ====== begin ====== inject eod_mask =====================================================
        ele_pos = ms.Tensor(0, ms.int64)
        seed = ms.Tensor(0, ms.int64)
        offset = ms.Tensor(0, ms.int64)
        start = 0
        steps = [5]
        error_mode = 'cycle'    # cycle, specific
        multiply_factor = 1.0
        bit_pos = 0
        flip_probability = 0.0
        # GenerateEodMaskV2()(input=<Tensor>, ele_pos=<Tensor>, cur_step=<Tensor>, seed=<Tensor>
        #   , offset=<Tensor>, start=<int>, steps=<int, list of int, tuple of int>, error_mode=<string>
        #   , flip_mode=<string>, multiply_factor=<float>, bit_pos=<int>, flip_probability=<float>)
        self.cur_step = self.cur_step + 1
        x = self.eod_mask(x, ele_pos, self.cur_step, seed, offset, start, steps, error_mode, self.flip_mode,
                          multiply_factor, bit_pos, flip_probability)
        # ====== *end* ====== inject eod_mask =====================================================
        x = self.relu1(x)
        x = self.matmul2(x, self.fc2_weight)
        x = self.relu2(x)
        logits = self.matmul3(x, self.fc3_weight)
        return logits

net = Network()
net.matmul1.shard(((1, 4), (4, 1)))
net.relu1.shard(((4, 1),))
net.matmul2.shard(((1, 4), (4, 1)))
net.relu2.shard(((4, 1),))

# fake dataset
def create_dataset(batch_size):
    # """create dataset"""
    # Random-accessible object as input source
    class RandomAccessDataset:
        def __init__(self):
            self.dataset_size = 20

        def __getitem__(self, index):
            image_np = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)
            label_np = np.random.randint(low=0, high=10, size=batch_size, dtype=np.int32)
            return ms.Tensor(image_np), ms.Tensor(label_np)

        def __len__(self):
            return self.dataset_size

    loader = RandomAccessDataset()
    return ds.GeneratorDataset(source=loader, column_names=["image", "label"])


data_set = create_dataset(32)
optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()

def forward_fn(data, target):
    """forward propagation"""
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)

@ms.jit
def train_step(inputs, targets):
    """train_step"""
    (loss_value, _), grads = grad_fn(inputs, targets)
    optimizer(grads)
    return loss_value

# training
for epoch in range(1):
    i = 0
    for image, label in data_set:
        loss_output = train_step(image, label)
        myrank_id = os.environ['RANK_ID']
        if i % 10 == 0 and myrank_id == '0':
            print("rank %s, epoch: %s, step: %s, loss is %s" % (myrank_id, epoch, i, loss_output))
        i += 1
```

### 网络运行脚本

上面网络脚本是4卡并行，其启动脚本（`run_silent_check.sh`）内容如下：

```bash
#!/bin/bash

# set cann log level to info
export ASCEND_GLOBAL_LOG_LEVEL=1

mpirun -n 4 --output-filename log_output --merge-stderr-to-stdout python silent_check.py
```

### 不同检测级别及运行日志

#### NPU_ASD_ENABLE 取值为 1 的运行日志

`NPU_ASD_ENABLE`取值为`1`的行为是当检测到特征值异常时，只打印 ERROR 日志，不中止训练。

启动命令：

```bash
NPU_ASD_ENABLE=1 bash run_silent_check.sh
```

通过查看 CANN 的 device 日志，默认在 `~/ascend/log/` 目录下，关键 ERROR 日志如下，从中可以有多条 ERROR 日志，即检测到异常值是并未中止训练。

```bash
$ cd ~/ascend/log/debug/
$ grep -nr 'silent_check_v2.cc:1.*SilentCheck' device-*
device-0/device-109652_20240913141212740.log:2508:[ERROR] AICPU(24515,aicpu_scheduler):2024-09-13-14:12:28.956.344 [silent_check_v2.cc:136][ComputeL1Error][tid:24528]SilentCheck get L1 Error, input message: val = [nan], pre_val = [0.100089], min_val = [0.089474], max_val = [0.130423], step = [6],         c_min_steps = [100], c_thresh_l1 = [1000000.000000], c_coeff_l1 = [100000.000000], c_thresh_l2 = [10000.000000], c_coeff_l2 = [5000.000000] npu_asd_detect = [1].
device-0/device-109652_20240913141212740.log:2556:[ERROR] AICPU(24515,aicpu_scheduler):2024-09-13-14:12:28.961.061 [silent_check_v2.cc:136][ComputeL1Error][tid:24523]SilentCheck get L1 Error, input message: val = [nan], pre_val = [0.026273], min_val = [0.026273], max_val = [0.026949], step = [6],         c_min_steps = [100], c_thresh_l1 = [1000000.000000], c_coeff_l1 = [100000.000000], c_thresh_l2 = [10000.000000], c_coeff_l2 = [5000.000000] npu_asd_detect = [1].
device-0/device-109652_20240913141212740.log:2604:[ERROR] AICPU(24515,aicpu_scheduler):2024-09-13-14:12:28.969.720 [silent_check_v2.cc:136][ComputeL1Error][tid:24524]SilentCheck get L1 Error, input message: val = [nan], pre_val = [0.005759], min_val = [0.005759], max_val = [0.005931], step = [6],         c_min_steps = [100], c_thresh_l1 = [1000000.000000], c_coeff_l1 = [100000.000000], c_thresh_l2 = [10000.000000], c_coeff_l2 = [5000.000000] npu_asd_detect = [1].
device-0/device-109652_20240913141212740.log:2652:[ERROR] AICPU(24515,aicpu_scheduler):2024-09-13-14:12:28.977.256 [silent_check_v2.cc:136][ComputeL1Error][tid:24525]SilentCheck get L1 Error, input message: val = [nan], pre_val = [0.004188], min_val = [0.004188], max_val = [0.004272], step = [6],         c_min_steps = [100], c_thresh_l1 = [1000000.000000], c_coeff_l1 = [100000.000000], c_thresh_l2 = [10000.000000], c_coeff_l2 = [5000.000000] npu_asd_detect = [1].
device-0/device-109652_20240913141212740.log:2786:[ERROR] AICPU(24515,aicpu_scheduler):2024-09-13-14:12:29.057.454 [silent_check_v2.cc:136][ComputeL1Error][tid:24526]SilentCheck get L1 Error, input message: val = [nan], pre_val = [nan], min_val = [0.089474], max_val = [0.130423], step = [7],         c_min_steps = [100], c_thresh_l1 = [1000000.000000], c_coeff_l1 = [100000.000000], c_thresh_l2 = [10000.000000], c_coeff_l2 = [5000.000000] npu_asd_detect = [1].
device-0/device-109652_20240913141212740.log:2834:[ERROR] AICPU(24515,aicpu_scheduler):2024-09-13-14:12:29.062.242 [silent_check_v2.cc:136][ComputeL1Error][tid:24527]SilentCheck get L1 Error, input message: val = [nan], pre_val = [nan], min_val = [0.026273], max_val = [0.026949], step = [7],         c_min_steps = [100], c_thresh_l1 = [1000000.000000], c_coeff_l1 = [100000.000000], c_thresh_l2 = [10000.000000], c_coeff_l2 = [5000.000000] npu_asd_detect = [1].
```

#### NPU_ASD_ENABLE 取值为 2 的运行日志

`NPU_ASD_ENABLE`取值为`2`的行为是当检测到特征值异常时，打印 ERROR 日志并中止训练。

启动命令：

```bash
NPU_ASD_ENABLE=2 bash run_silent_check.sh
```

通过查看 CANN 的 device 日志，默认在 `~/ascend/log/` 目录下，关键 ERROR 日志如下，发现只有一条 ERROR 日志，即检测到异常值是中止了训练：

```bash
$ cd ~/ascend/log/debug/
$ grep -nr 'silent_check_v2.cc:1.*SilentCheck' device-*
device-2/device-134035_20240913141623807.log:2204:[ERROR] AICPU(1685,aicpu_scheduler):2024-09-13-14:16:43.130.671 [silent_check_v2.cc:136][ComputeL1Error][tid:1695]SilentCheck get L1 Error, input message: val = [inf], pre_val = [0.105156], min_val = [0.089474], max_val = [0.130423], step = [5],         c_min_steps = [100], c_thresh_l1 = [1000000.000000], c_coeff_l1 = [100000.000000], c_thresh_l2 = [10000.000000], c_coeff_l2 = [5000.000000] npu_asd_detect = [2].
$
```

#### NPU_ASD_ENABLE 取值为 3 的运行日志

`NPU_ASD_ENABLE`取值为`３`的行为是当检测到特征值异常时，与取值`2`行为类似，除打印 ERROR 日志并中止训练外，还会在 CANN 日志中打印特征值没有异常时检测算子的入参信息（需要
通过`export ASCEND_GLOBAL_LOG_LEVEL=0`开启debug级别日志或`export ASCEND_GLOBAL_LOG_LEVEL=1`开启info级别日志才会把非异常场景下的日志输出到 CANN 日志中）。

启动命令：

```bash
NPU_ASD_ENABLE=3 bash run_silent_check.sh
```

通过查看 CANN 的 device 日志，默认在 `~/ascend/log/` 目录下，关键 ERROR 日志如下，发现出了 ERROR 日志之外，还有一些 SilentCheck 的 INFO 日志：

```bash
$ cd ~/ascend/log/debug/
$ grep -nr 'silent_check_v2.cc:1.*SilentCheck' device-*
device-2/device-151329_20240913141834821.log:2056:[INFO] AICPU(2249,aicpu_scheduler):2024-09-13-14:18:53.048.647 [silent_check_v2.cc:124][SilentCheck][tid:2257]c_min_steps = [100], c_thresh_l1 = [1000000.000000], c_coeff_l1 = [100000.000000], c_thresh_l2 = [10000.000000], c_coeff_l2 = [5000.000000], npu_asd_detect = [3]
device-2/device-151329_20240913141834821.log:2104:[INFO] AICPU(2249,aicpu_scheduler):2024-09-13-14:18:53.057.632 [silent_check_v2.cc:122][SilentCheck][tid:2258]val = [0.004220], pre_val = [0.004272], min_val = [0.004189], max_val = [0.004272], step = [4].
device-2/device-151329_20240913141834821.log:2105:[INFO] AICPU(2249,aicpu_scheduler):2024-09-13-14:18:53.057.642 [silent_check_v2.cc:124][SilentCheck][tid:2258]c_min_steps = [100], c_thresh_l1 = [1000000.000000], c_coeff_l1 = [100000.000000], c_thresh_l2 = [10000.000000], c_coeff_l2 = [5000.000000], npu_asd_detect = [3]
device-2/device-151329_20240913141834821.log:2149:[INFO] AICPU(2249,aicpu_scheduler):2024-09-13-14:18:53.096.224 [silent_check_v2.cc:122][SilentCheck][tid:2257]val = [0.007812], pre_val = [0.007812], min_val = [0.007812], max_val = [0.007812], step = [5].
device-2/device-151329_20240913141834821.log:2150:[INFO] AICPU(2249,aicpu_scheduler):2024-09-13-14:18:53.096.234 [silent_check_v2.cc:124][SilentCheck][tid:2257]c_min_steps = [100], c_thresh_l1 = [1000000.000000], c_coeff_l1 = [100000.000000], c_thresh_l2 = [10000.000000], c_coeff_l2 = [5000.000000], npu_asd_detect = [3]
device-2/device-151329_20240913141834821.log:2194:[INFO] AICPU(2249,aicpu_scheduler):2024-09-13-14:18:53.099.659 [silent_check_v2.cc:122][SilentCheck][tid:2258]val = [0.000000], pre_val = [0.000000], min_val = [0.000000], max_val = [0.000000], step = [5].
device-2/device-151329_20240913141834821.log:2195:[INFO] AICPU(2249,aicpu_scheduler):2024-09-13-14:18:53.099.668 [silent_check_v2.cc:124][SilentCheck][tid:2258]c_min_steps = [100], c_thresh_l1 = [1000000.000000], c_coeff_l1 = [100000.000000], c_thresh_l2 = [10000.000000], c_coeff_l2 = [5000.000000], npu_asd_detect = [3]
device-2/device-151329_20240913141834821.log:2243:[ERROR] AICPU(2249,aicpu_scheduler):2024-09-13-14:18:53.127.190 [silent_check_v2.cc:136][ComputeL1Error][tid:2259]SilentCheck get L1 Error, input message: val = [inf], pre_val = [0.105156], min_val = [0.089474], max_val = [0.130423], step = [5],         c_min_steps = [100], c_thresh_l1 = [1000000.000000], c_coeff_l1 = [100000.000000], c_thresh_l2 = [10000.000000], c_coeff_l2 = [5000.000000] npu_asd_detect = [3].
```

## 检测结果及处理

### 异常检测结果

未检测到数值异常时，对训练任务运行无影响。

当检测到数值异常后，训练任务失败并上报告警，请通过如下方法之一定位故障设备：

* 通过搜索应用类日志，查询**ERROR**级别错误日志，关键字"accuracy sensitivity feature abnormal"；
* 通过监控NPU健康状态：Health Status显示Warning，Error Code显示80818C00，Error Information显示node type=SoC, sensor type=Check Sensor, event state=check fail；
* 通过查看[Ascend Device Plugin](https://github.com/Ascend/ascend-device-plugin)事件，上报错误码80818C00，事件类型为故障事件，故障级别次要。

### 故障处理

将异常设备隔离，断点续训拉起继续训练；同时在异常设备上，通过Ascend-DMI工具执行AICore ERROR压测诊断，检测该设备上是否存在故障NPU。详情请查看[《ToolBox用户指南》](https://www.hiascend.com/document/detail/zh/mindx-dl/600/toolbox/ascenddmi/toolboxug_000002.html) “ascend-dmi工具使用 > 故障诊断”章节。

若异常设备上检测到故障卡，请联系华为工程师维修更换；若异常设备上所有NPU均正常，则为软件类问题触发特征值溢出，建议排查程序和算子原因。