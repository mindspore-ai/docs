# Feature Value Detection

[![View Source File](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/model_train/debug/sdc.md)

## Overview

### Background

During model training, processors may encounter feature value detection anomalies, resulting in computational errors without reporting. Feature value detection anomalies may seriously affect model training.

### Solution

The MindSpore framework version 2.4 provides a solution for feature value detection of Transformer structure models. Internally a feature value detection operator is inserted before the communication operator in the backward graph to monitor feature value and prevent anomaly from spreading to other cards.

For default feature value detection checkpoints, users can enable detection capability using the environment variable `NPU_ASD_ENABLE=1`, `NPU_ASD_ENABLE=2`  or `NPU_ASD_ENABLE=3`, and adjust the detection intensity by configuring the environment variables `NPU_ASD_UPPER_THRESH` and `NPU_ASD_SIGMA_THRESH`.

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

The reasons for feature value anomalies can be divided into two categories: hardware errors and software errors, which can be referred to in the **Fault Handling** section for further analysis.

### Usage Restrictions

Currently, this feature only supports Atlas A2 training series products, detects abnormal feature value during the training process with Transformer model and bfloat16, float32 data type.

## Feature Switches and Configuration

The environment variable `NPU_ASD_ENABLE` serves as a feature switch, `export NPU_ASD_ENABLE=1`, `export NPU_ASD_ENABLE=2` or `export NPU_ASD_ENABLE=3` to enable this feature; if this environment variable is not configured or `export NPU_ASD_ENABLE=0`, this feature is disabled.

The environment variable `NPU_ASD_UPPER_THRESH` controls the absolute numerical threshold of detection, in the format of integer pairs, where the first element controls the first-level threshold of absolute numerical values, and the second element controls the second-level threshold of absolute numerical values; reducing the threshold can detect smaller fluctuations in abnormal data, increase the detection rate, and increasing the threshold is the opposite. In the default case where this environment variable is not configured, `NPU_ASD_UPPER_THRESH=1000000,10000`.

The environment variable `NPU_ASD_SIGMA_THRESH` controls the relative numerical threshold of detection, in the same format as the above, where the first element controls the first-level threshold of numerical changes, and the second element controls the second-level threshold of numerical changes; by default, `NPU_ASD_SIGMA_THRESH=100000,5000`.

For details of above environment variables, see [Environment Variables](https://www.mindspore.cn/docs/en/master/note/env_var_list.html).

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

From the CANN log, by default the log path os `~/ascend/log/`, the main `ERROR` logs are as follows, there are many `ERROR` logs. After error was detected, the training process was not stopped.

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

#### Execution Result of Setting NPU_ASD_ENABLE to 2

When `NPU_ASD_ENABLE` was set to `2`, if error was detected, print `ERROR` log and stop training process.

Start command:

```bash
NPU_ASD_ENABLE=2 bash run_silent_check.sh
```

From the CANN log, by default the log path os `~/ascend/log/`, the main `ERROR` logs are as follows, there only one `ERROR` log. After error was detected, the training process was stopped.

```bash
$ cd ~/ascend/log/debug/
$ grep -nr 'silent_check_v2.cc:1.*SilentCheck' device-*
device-2/device-134035_20240913141623807.log:2204:[ERROR] AICPU(1685,aicpu_scheduler):2024-09-13-14:16:43.130.671 [silent_check_v2.cc:136][ComputeL1Error][tid:1695]SilentCheck get L1 Error, input message: val = [inf], pre_val = [0.105156], min_val = [0.089474], max_val = [0.130423], step = [5],         c_min_steps = [100], c_thresh_l1 = [1000000.000000], c_coeff_l1 = [100000.000000], c_thresh_l2 = [10000.000000], c_coeff_l2 = [5000.000000] npu_asd_detect = [2].
$
```

#### Execution Result of Setting NPU_ASD_ENABLE to 3

When `NPU_ASD_ENABLE` was set to `3`, the action is similar to detection level `2`, i.e. if error was detected, print `ERROR` log and stop training process. Besides an `INFO` log as also output for
non anomaly feature values (In oreder to see logs of level INFO, need to set `export ASCEND_GLOBAL_LOG_LEVEL=0` to enable log level `DEBUG` or set `export ASCEND_GLOBAL_LOG_LEVEL=1` to enable log level `INFO`).

Start command:

```bash
NPU_ASD_ENABLE=3 bash run_silent_check.sh
```

From the CANN log, by default the log path os `~/ascend/log/`, the main `ERROR` logs are as follows, there are `INFO` logs except for `ERORR` log about feature value info.

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

## Detection Results and Handling

### Abnormal Detection Results

When no numerical anomalies are detected, the training task runs without impact.

When numerical anomalies are detected, the training task fails and alerts are reported. To locate the faulty device, do one of the following:

* Search application logs for **ERROR** level error logs with the keyword "accuracy sensitivity feature abnormal";
* Monitor the NPU health status: if Health Status displays Warning, Error Code displays 80818C00, and Error Information displays node type=SoC, sensor type=Check Sensor, event state=check fail;
* Check the [Ascend Device Plugin](https://github.com/Ascend/ascend-device-plugin) events, report error code 80818C00, event type is fault event, and the fault level is minor.

### Fault Handling

Isolate the abnormal device, resume training with checkpoint recovery; meanwhile, on the abnormal device, use the Ascend-DMI tool to perform AICore ERROR stress diagnostics to detect whether there are faulty NPUs on the device. For details, see [ToolBox User Guide](https://www.hiascend.com/document/detail/zh/mindx-dl/2046/dluserguide/toolboxug/toolboxug_000002.html) in the "ascend-dmi tool usage > fault diagnosis" section.

If a faulty card is detected on the abnormal device, contact Huawei engineers for maintenance and replacement; if all NPUs on the abnormal device are normal, it is a software-related issue triggering feature value overflow, and it is recommended to check the processes and operators'es causes.