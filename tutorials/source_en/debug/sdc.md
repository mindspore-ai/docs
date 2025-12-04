# Feature Value Detection

[![View Source File](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/debug/sdc.md)

## Overview

### Background

During model training, processors may encounter feature value detection anomalies, resulting in computational errors without reporting. Feature value detection anomalies may seriously affect model training.

### Solution

The Mindspore framework version 2.7.1 provides a combined detection scheme using feature value and CheckSum, which can more accurately locate silent faults. It samples parameter gradients for feature value detection. When multiple anomalies occur, CheckSum is triggered by a "strike out" mechanism to further locate the faulty device. Users can configure the combined detection through `MS_NPU_ASD_CONFIG`.

For information on configuring related environment variables, see **Feature Switches and Configuration**.

For an introduction to default feature value detection checkpoints, and design guidelines for custom feature value detection checkpoints, see **Usage Recommendations and Detection Principles**.

### Usage Recommendations and Detection Principles

When processors encounter feature value detection anomalies, erroneous results are calculated. Due to the structure of Transformer models, these erroneous calculation results will propagate.

Based on experimental results, the following empirical conclusions are drawn:

* Not all feature value detection anomalies necessarily affect model convergence and performance. In fact, most feature value detection anomalies do not have observable effects on the model. See [reference](https://dl.acm.org/doi/abs/10.1145/3579371.3589105).
* Statistically, feature value detection anomalies during the backpropagation calculation process have a much greater impact than during the forward calculation process.
* In parallel training scenarios, calculation error results will propagate due to parallel computation.
* Setting too many checkpoints will affect model training performance.
* Based on experiments on the sensitivity of calculation errors, the MindSpore framework defaults to selecting the `Norm` activation value gradient in the backpropagation calculation process as the detection feature value, with performance loss less than 2% based on **Llama 2 - 7B** testing.

When using the combined detection scheme of feature value and CheckSum (set `enable:true` in `MS_NPU_ASD_CONFIG`), feature values are sampled from parameter gradients before communication in the backward graph, and an algorithm is used to detect if an anomaly exists. If feature value detection working with CheckSum (set `with_checksum:true` in `MS_NPU_ASD_CONFIG`), CheckSum will be performed when the number of anomalies exceeds the threshold within a time window. CheckSum verifies the calculation results of the MatMul operator of bfloat16 data type on each device to identify the faulty one.

The reasons for feature value anomalies can be divided into two categories: hardware errors and software errors, which can be referred to in the **Fault Handling** section for further analysis.

### Usage Restrictions

Currently, this feature only supports Atlas A2 training series products, detects abnormal feature value during the training process with Transformer model within 8-D and bfloat16, float32 data type.

The combined detection scheme currently only supports `auto_parallel` or `semi_auto_parallel` modes. CheckSum only verifies the MatMul operator of bfloat16 data type.

## Feature Switches and Configuration

The environment variable `MS_NPU_ASD_CONFIG` configures the combined detection scheme of feature value and CheckSum, in the format of `key:value`, with each configuration item separated by commas. `enable` is the feature value detection switch, `with_checksum` is the CheckSum linkage switch, `grad_sample_interval` is the feature value sampling interval, `upper_thresh1` and `upper_thresh2` control the absolute and relative thresholds of feature value detection respectively, `cooldown` is the feature value detection anomaly cooldown time and the CheckSum execution time, `strikes_num` and `strikes_window` are the number of feature value detection anomalies and the time window required to trigger CheckSum, and `checksum_cooldown` is the CheckSum cooldown time. By default, `MS_NPU_ASD_CONFIG="enable:false,with_checksum:false,grad_sample_interval:10,upper_thresh1:1000000,upper_thresh2:100,cooldown:5,strikes_num:3,strikes_window:480,checksum_cooldown:180"`.

For details of above environment variables, see [Environment Variables](https://www.mindspore.cn/docs/en/master/api_python/env_var_list.html).

## Use Cases

> This document describes the usage methods and use cases of feature value detection.

A simple neural network is constructed here, and feature value anomalies are simulated through MindSpore's fault injection operator. The network script (`silent_detect.py`) is as below:

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

Start command:

```bash
export MS_NPU_ASD_CONFIG="enable:true,with_checksum:true,grad_sample_interval:1,cooldown:1,strikes_num:1"
msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=11235 --join=True python silent_detect.py
```

Feature value detection anomalies and CheckSum verification results can be observed in the training logs (default is `worker_*.log`):

```bash
$ grep -m1 'Silent detect strike' worker_0.log
[WARNING] DEBUG(2950752,fffee7e591e0,python):2025-08-26-10:46:26.665.782 [mindspore/ccsrc/tools/silent_detect/silent_detector.cc:109] SilentDetect] Silent detect strike detected: StrikeRecord{timestamp: 1756176386, name: fc1.weight, value: inf, stat: StatData{avg: 6.44326e+12, pre_value: 6.441e+14, count: 6, none_zero_count: 6}}
$ grep -m1 'Global CheckSum result is' worker_0.log
[WARNING] DEBUG(2950752,fffda37fe1e0,python):2025-08-26-10:47:28.934.305 [mindspore/ccsrc/tools/silent_detect/silent_detector.cc:316] DoCheckSum] Global CheckSum result is 0
```

## Detection Results and Handling

### Abnormal Detection Results

When no numerical anomalies are detected, the training task runs without impact.

When numerical anomalies are detected, the training task fails and alerts are reported. To locate the faulty device, do one of the following:

* Search application logs for **ERROR** level error logs with the keyword "accuracy sensitivity feature abnormal";
* Monitor the NPU health status: if Health Status displays Warning, Error Code displays 80818C00, and Error Information displays node type=SoC, sensor type=Check Sensor, event state=check fail;
* Check the [MindCluster](https://atomgit.com/Ascend/mind-cluster) events, report error code 80818C00, event type is fault event, and the fault level is minor.

When using combined detection, if feature value detection anomalies occur and CheckSum detects silent faults, warning logs can be found in the training logs:

* The keyword for feature value detection anomalies is "Silent detect strike";
* The keyword for triggering CheckSum is "Feature value detection strikes out";
* The keywords for silent errors are "CheckSum detects MatMul error on rank" and "SilentCheck detects SDC error".

### Fault Handling

Isolate the abnormal device, resume training with checkpoint recovery; meanwhile, on the abnormal device, use the Ascend-DMI tool to perform AICore ERROR stress diagnostics to detect whether there are faulty NPUs on the device. For details, see [ToolBox User Guide](https://www.hiascend.com/document/detail/zh/mindx-dl/600/toolbox/ascenddmi/toolboxug_000002.html) in the "ascend-dmi tool usage > fault diagnosis" section.

If a faulty card is detected on the abnormal device, contact Huawei engineers for maintenance and replacement; if all NPUs on the abnormal device are normal, it is a software-related issue triggering feature value overflow, and it is recommended to check the processes and operators'es causes.