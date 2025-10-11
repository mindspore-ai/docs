# Feature Value Detection

[![View Source File](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.7.1/tutorials/source_en/debug/sdc.md)

## Overview

### Background

During model training, processors may encounter feature value detection anomalies, resulting in computational errors without reporting. Feature value detection anomalies may seriously affect model training.

### Solution

The MindSpore framework version 2.4 provides a solution for feature value detection of Transformer structure models. Internally a feature value detection operator is inserted before the communication operator in the backward graph to monitor feature value and prevent anomaly from spreading to other cards.

For default feature value detection checkpoints, users can enable detection capability using the environment variable `NPU_ASD_ENABLE=1`, `NPU_ASD_ENABLE=2` or `NPU_ASD_ENABLE=3`, and adjust the detection intensity by configuring the environment variables `NPU_ASD_UPPER_THRESH` and `NPU_ASD_SIGMA_THRESH`.

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

After enabling the detection switch (set `NPU_ASD_ENABLE` to `1`, `2` or `3`), during the backpropagation phase of training Transformer structure models, abnormality is determined by collecting the activation value gradients of the Norm layer through calling the detection operator inserted before the communication operator in the backward graph, and using an algorithm to determine if an anomaly exists. If an anomaly occurs, print the relevant logs or terminate the training depending on the different values of the environment variable `NPU_ASD_ENABLE`, and set the NPU state on the device where the anomaly is detected to Warning to report the fault event.

When using the combined detection scheme of feature value and CheckSum (set `enable:true` in `MS_NPU_ASD_CONFIG`), feature values are sampled from parameter gradients before communication in the backward graph, and an algorithm is used to detect if an anomaly exists. If feature value detection working with CheckSum (set `with_checksum:true` in `MS_NPU_ASD_CONFIG`), CheckSum will be performed when the number of anomalies exceeds the threshold within a time window. CheckSum verifies the calculation results of the MatMul operator of bfloat16 data type on each device to identify the faulty one.

The reasons for feature value anomalies can be divided into two categories: hardware errors and software errors, which can be referred to in the **Fault Handling** section for further analysis.

### Usage Restrictions

Currently, this feature only supports Atlas A2 training series products, detects abnormal feature value during the training process with Transformer model within 8-D and bfloat16, float32 data type.

The combined detection scheme currently only supports `auto_parallel` or `semi_auto_parallel` modes. CheckSum only verifies the MatMul operator of bfloat16 data type.

## Feature Switches and Configuration

The environment variable `NPU_ASD_ENABLE` serves as a feature switch, `export NPU_ASD_ENABLE=1`, `export NPU_ASD_ENABLE=2` or `export NPU_ASD_ENABLE=3` to enable this feature; if this environment variable is not configured or `export NPU_ASD_ENABLE=0`, this feature is disabled.

The environment variable `NPU_ASD_UPPER_THRESH` controls the absolute numerical threshold of detection, in the format of integer pairs, where the first element controls the first-level threshold of absolute numerical values, and the second element controls the second-level threshold of absolute numerical values; reducing the threshold can detect smaller fluctuations in abnormal data, increase the detection rate, and increasing the threshold is the opposite. In the default case where this environment variable is not configured, `NPU_ASD_UPPER_THRESH=1000000,10000`.

The environment variable `NPU_ASD_SIGMA_THRESH` controls the relative numerical threshold of detection, in the same format as the above, where the first element controls the first-level threshold of numerical changes, and the second element controls the second-level threshold of numerical changes; by default, `NPU_ASD_SIGMA_THRESH=100000,5000`.

The environment variable `MS_NPU_ASD_CONFIG` configures the combined detection scheme of feature value and CheckSum, in the format of `key:value`, with each configuration item separated by commas. `enable` is the feature value detection switch, `with_checksum` is the CheckSum linkage switch, `grad_sample_interval` is the feature value sampling interval, `upper_thresh1` and `upper_thresh2` control the absolute and relative thresholds of feature value detection respectively, `cooldown` is the feature value detection anomaly cooldown time and the CheckSum execution time, `strikes_num` and `strikes_window` are the number of feature value detection anomalies and the time window required to trigger CheckSum, and `checksum_cooldown` is the CheckSum cooldown time. By default, `MS_NPU_ASD_CONFIG="enable:false,with_checksum:false,grad_sample_interval:10,upper_thresh1:1000000,upper_thresh2:100,cooldown:5,strikes_num:3,strikes_window:480,checksum_cooldown:180"`.

For details of above environment variables, see [Environment Variables](https://www.mindspore.cn/docs/en/r2.7.1/api_python/env_var_list.html).

## Use Cases

> This document describes the usage methods and use cases of feature value detection.

### Model and Dataset Preparation

To provide a complete experience, simple neural networks and a simulated dataset are constructed here, and the use of feature value detection is demonstrated by simulating feature value anomalies through MindSpore's fault injection operator (a step that is not required in the actual network).

The full python script (`silent_check.py`) is as below:

```python
"""Silent Check Demo"""

import os
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops
from mindspore.communication import init
from mindspore.common.initializer import initializer
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.nn.utils import no_init_parameters


ms.set_context(mode=ms.GRAPH_MODE)
ms.runtime.set_memory(max_size="2GB")
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
        ele_pos = ms.Tensor(1, ms.int64)
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

with no_init_parameters():
    net = Network()
    optimizer = nn.SGD(net.trainable_params(), 1e-2)
net.matmul1.shard(((1, 4), (4, 1)))
net.relu1.shard(((4, 1),))
net.matmul2.shard(((1, 4), (4, 1)))
net.relu2.shard(((4, 1),))
parallel_net = AutoParallel(net, parallel_mode='semi_auto')

# fake dataset
def create_dataset(batch_size):
    # """create dataset"""
    # Random-accessible object as input source
    class RandomAccessDataset:
        def __init__(self):
            self.dataset_size = 20

        def __getitem__(self, index):
            image_np = np.random.randn(batch_size, 1, 28, 28).astype(np.float32) + 10
            label_np = np.random.randint(low=0, high=10, size=batch_size, dtype=np.int32)
            return ms.Tensor(image_np), ms.Tensor(label_np)

        def __len__(self):
            return self.dataset_size

    loader = RandomAccessDataset()
    return ds.GeneratorDataset(source=loader, column_names=["image", "label"])


data_set = create_dataset(32)
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

### Running Silent Check Script

This silent check demo uses 4 NPU cards, the start script (`run_silent_check.sh`) is as follows:

```bash
#!/bin/bash

# set cann log level to info
export ASCEND_GLOBAL_LOG_LEVEL=1

mpirun -n 4 --output-filename log_output --merge-stderr-to-stdout python silent_check.py
```

### Different Detection Levels and Running Results

#### Execution Result of Setting NPU_ASD_ENABLE to 1

When `NPU_ASD_ENABLE` was set to `1`, if error was detected, just print `ERROR` log, not stop training process.

Start command:

```bash
NPU_ASD_ENABLE=1 bash run_silent_check.sh
```

From the CANN log, by default the log path is `~/ascend/log/`, the main `ERROR` logs are as follows, there are many `ERROR` logs. After error was detected, the training process was not stopped.

```bash
$ cd ~/ascend/log/debug/
$ grep -nr 'silent_check_v[2-9].cc:.*SilentCheck' device-*
device-0/device-299066_20250225184036913.log:1968:[ERROR] AICPU(26533,aicpu_scheduler):2025-02-25-18:40:56.176.403 [silent_check_v3.cc:250][ComputeL1Error][tid:26552]SilentCheckV3 ComputeL1Error:val = [nan], max = [nan], avg=[1.128970e-09], step=[5], c_thresh_l1 = [1.000000e+06], c_thresh_l2 = [1.000000e+04], beta1 = [9.900000e-01], npu_asd_detect = [1].
device-0/device-299066_20250225184036913.log:2134:[ERROR] AICPU(26533,aicpu_scheduler):2025-02-25-18:40:56.269.071 [silent_check_v3.cc:250][ComputeL1Error][tid:26547]SilentCheckV3 ComputeL1Error:val = [nan], max = [nan], avg=[6.995705e-08], step=[6], c_thresh_l1 = [1.000000e+06], c_thresh_l2 = [1.000000e+04], beta1 = [9.900000e-01], npu_asd_detect = [1].
device-0/device-299066_20250225184036913.log:2190:[ERROR] AICPU(26533,aicpu_scheduler):2025-02-25-18:40:56.275.860 [silent_check_v3.cc:250][ComputeL1Error][tid:26548]SilentCheckV3 ComputeL1Error:val = [nan], max = [nan], avg=[nan], step=[6], c_thresh_l1 = [1.000000e+06], c_thresh_l2 = [1.000000e+04], beta1 = [9.900000e-01], npu_asd_detect = [1].
device-0/device-299066_20250225184036913.log:2246:[ERROR] AICPU(26533,aicpu_scheduler):2025-02-25-18:40:56.282.746 [silent_check_v3.cc:250][ComputeL1Error][tid:26549]SilentCheckV3 ComputeL1Error:val = [nan], max = [nan], avg=[1.526131e-09], step=[6], c_thresh_l1 = [1.000000e+06], c_thresh_l2 = [1.000000e+04], beta1 = [9.900000e-01], npu_asd_detect = [1].
device-0/device-299066_20250225184036913.log:2357:[ERROR] AICPU(26533,aicpu_scheduler):2025-02-25-18:40:56.366.766 [silent_check_v3.cc:250][ComputeL1Error][tid:26549]SilentCheckV3 ComputeL1Error:val = [nan], max = [nan], avg=[nan], step=[7], c_thresh_l1 = [1.000000e+06], c_thresh_l2 = [1.000000e+04], beta1 = [9.900000e-01], npu_asd_detect = [1].
device-0/device-299066_20250225184036913.log:2413:[ERROR] AICPU(26533,aicpu_scheduler):2025-02-25-18:40:56.373.589 [silent_check_v3.cc:250][ComputeL1Error][tid:26550]SilentCheckV3 ComputeL1Error:val = [nan], max = [nan], avg=[nan], step=[7], c_thresh_l1 = [1.000000e+06], c_thresh_l2 = [1.000000e+04], beta1 = [9.900000e-01], npu_asd_detect = [1].
```

#### Execution Result of Setting NPU_ASD_ENABLE to 2

When `NPU_ASD_ENABLE` was set to `2`, if error was detected, print `ERROR` log and stop training process.

Start command:

```bash
NPU_ASD_ENABLE=2 bash run_silent_check.sh
```

From the CANN log, by default the log path os `~/ascend/log/`, the main `ERROR` logs are as follows, there only one `ERROR` log. After error was detected, the training process was stopped.

```bash
$ cd ~/ascend/log/debug/
$ grep -nr 'silent_check_v[2-9].cc:.*SilentCheck' device-*
device-2/device-305322_20250225184310213.log:1859:[ERROR] AICPU(25787,aicpu_scheduler):2025-02-25-18:43:29.395.610 [silent_check_v3.cc:250][ComputeL1Error][tid:25799]SilentCheckV3 ComputeL1Error:val = [nan], max = [nan], avg=[5.752283e-08], step=[5], c_thresh_l1 = [1.000000e+06], c_thresh_l2 = [1.000000e+04], beta1 = [9.900000e-01], npu_asd_detect = [2].
```

#### Execution Result of Setting NPU_ASD_ENABLE to 3

When `NPU_ASD_ENABLE` was set to `3`, the action is similar to detection level `2`, i.e. if error was detected, print `ERROR` log and stop training process. Besides an `INFO` log as also output for non anomaly feature values (In oreder to see logs of level INFO, need to set `export ASCEND_GLOBAL_LOG_LEVEL=0` to enable log level `DEBUG` or set `export ASCEND_GLOBAL_LOG_LEVEL=1` to enable log level `INFO`).

Start command:

```bash
NPU_ASD_ENABLE=3 bash run_silent_check.sh
```

From the CANN log, by default the log path os `~/ascend/log/`, the main `ERROR` logs are as follows, there are `INFO` logs except for `ERROR` log about feature value info.

```bash
$ cd ~/ascend/log/debug/
$ grep -nr 'silent_check_v[2-9].cc:.*SilentCheck' device-*
device-2/device-311523_20250225184632284.log:1767:[INFO] AICPU(26559,aicpu_scheduler):2025-02-25-18:46:51.678.981 [silent_check_v3.cc:240][SilentCheck][tid:26572]SilentCheckV3 normal case, val=[3.350337e-08], max=[3.350337e-08], avg=[9.879098e-10], step=[4], c_threshold_l1=[1.000000e+06], c_threshold_l2=[1.000000e+04], beta1=[9.900000e-01], npu_asd_detect=[3].
device-2/device-311523_20250225184632284.log:1829:[INFO] AICPU(26559,aicpu_scheduler):2025-02-25-18:46:51.684.993 [silent_check_v3.cc:240][SilentCheck][tid:26570]SilentCheckV3 normal case, val=[2.016393e+02], max=[2.016393e+02], avg=[7.349676e+00], step=[4], c_threshold_l1=[1.000000e+06], c_threshold_l2=[1.000000e+04], beta1=[9.900000e-01], npu_asd_detect=[3].
device-2/device-311523_20250225184632284.log:1891:[ERROR] AICPU(26559,aicpu_scheduler):2025-02-25-18:46:51.762.577 [silent_check_v3.cc:250][ComputeL1Error][tid:26572]SilentCheckV3 ComputeL1Error:val = [nan], max = [nan], avg=[5.752281e-08], step=[5], c_thresh_l1 = [1.000000e+06], c_thresh_l2 = [1.000000e+04], beta1 = [9.900000e-01], npu_asd_detect = [3].
```

### Combined Detection

When using the combined detection scheme (set `enable:true` in `MS_NPU_ASD_CONFIG`), the feature value detection method corresponding to this scheme will be used, and `NPU_ASD_ENABLE` will be ignored.

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
* Check the [Ascend Device Plugin](https://github.com/Ascend/ascend-device-plugin) events, report error code 80818C00, event type is fault event, and the fault level is minor.

When using combined detection, if feature value detection anomalies occur and CheckSum detects silent faults, warning logs can be found in the training logs:

* The keyword for feature value detection anomalies is "Silent detect strike";
* The keyword for triggering CheckSum is "Feature value detection strikes out";
* The keywords for silent errors are "CheckSum detects MatMul error on rank" and "SilentCheck detects SDC error".

### Fault Handling

Isolate the abnormal device, resume training with checkpoint recovery; meanwhile, on the abnormal device, use the Ascend-DMI tool to perform AICore ERROR stress diagnostics to detect whether there are faulty NPUs on the device. For details, see [ToolBox User Guide](https://www.hiascend.com/document/detail/zh/mindx-dl/600/toolbox/ascenddmi/toolboxug_000002.html) in the "ascend-dmi tool usage > fault diagnosis" section.

If a faulty card is detected on the abnormal device, contact Huawei engineers for maintenance and replacement; if all NPUs on the abnormal device are normal, it is a software-related issue triggering feature value overflow, and it is recommended to check the processes and operators'es causes.