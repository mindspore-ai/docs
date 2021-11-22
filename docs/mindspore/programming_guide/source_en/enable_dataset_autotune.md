# Enabling AutoTune for Dataset

`Ascend` `GPU` `CPU` `Data Preparation`

<!-- TOC -->

- [Enabling AutoTune for Dataset Pipeline](#enabling-autotune-for-dataset)
    - [Overview](#overview)
    - [Enable AutoTune](#enable-autotune)
    - [Time Interval for AutoTune](#time-interval-for-autotune)
    - [Constraints](#constraints)
    - [Example](#example)
        - [AutoTune Config](#autotune-config)
        - [Start training](#start-training)
        - [Before next training](#before-next-training)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/enable_dataset_autotune.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore provides AutoTune support to automatically tune Dataset pipelines to improve performance.

This feature can automatically detect a bottleneck operator in the dataset pipeline and respond by automatically adjusting tunable parameters for dataset ops, like increasing the number of parallel workers or updating the prefetch size of dataset ops.

![autotune](images/autotune.png)

With dataset AutoTune enabled, MindSpore will sample dataset statistics at a given interval, which is tuneable by the user.

Once AutoTune collects enough information, it will analyze whether the performance bottleneck is on the dataset side or not.
If so, it will adjust the parallelism and speedup the dataset pipeline.
If not, AutoTune will also try to reduce the memory usage of the dataset pipeline to release memory for CPU.

> AutoTune for Dataset is disabled by default.

## Enable AutoTune

To enable AutoTune for Dataset:

```python
import mindspore.dataset as ds
ds.config.set_enable_autotune(True)
```

## Time Interval for AutoTune

To set the time interval (in milliseconds) for dataset pipeline autotuning:

```python
import mindspore.dataset as ds
ds.config.set_autotune_interval(100)
```

To query the time interval (in milliseconds) for dataset pipeline autotuning:

```python
import mindspore.dataset as ds
print("time interval:", ds.config.get_autotune_interval())
```

## Constraints

AutoTune for Dataset is currently available for sink mode only (dataset_sink_mode=True).

Both dataset profiling and dataset Autotune may not be enabled concurrently.
A warning message will result if you enable Dataset AutoTune first and then Dataset Profiling, or vice versa.

## Example

Take LeNet training as example.

### AutoTune config

To enable AutoTune for Dataset, only one statement is needed.

```python
# dataset.py of LeNet in ModelZoo
# models/official/cv/lenet/src/dataset.py

def create_dataset(...)
    """
    create dataset for train or test
    """
    # enable AutoTune for Dataset
    ds.config.set_enable_autotune(True)

    # define dataset
    mnist_ds = ds.MnistDataset(data_path)
    ...
```

### Start training

Start the training process as described in [lenet/README.md](https://gitee.com/mindspore/models/blob/master/official/cv/lenet/README.md). AutoTune will display its analysis result through LOG messages.

```text
[INFO] [auto_tune.cc] LaunchThread] Launching AutoTune thread
[INFO] [auto_tune.cc] Main] AutoTune thread has started.
[INFO] [auto_tune.cc] RunIteration] Run AutoTune at epoch #1
[INFO] [auto_tune.cc] RecordPipelineTime] Epoch #1, Average Pipeline time is 6.88267 ms. The avg pipeline time for all epochs is 6.88267ms
[INFO] [auto_tune.cc] IsDSaBottleneck] Epoch #1, Device Connector Size: 4.65387, Connector Capacity: 16, Utilization: 29.0867%, Empty Freq: 48.32%
[WARNING] [auto_tune.cc] IsDSaBottleneck] Utilization: 29.0867% < 75% threshold, dataset pipeline performance needs tuning.
[WARNING] [auto_tune.cc] Analyse] Leaf op (MnistOp(ID:9)) queue utilization: 0% < 90% threshold.
[WARNING] [auto_tune.cc] RequestNumWorkerChange] Added request to change number of workers of Operator: MnistOp(ID:9) New value: 10 Old value: 8
[WARNING] [auto_tune.cc] Analyse] Op (MapOp(ID:8)) getting low average worker cpu utilization 18.8182% < 35% threshold.
[WARNING] [auto_tune.cc] RequestConnectorCapacityChange] Added request to change Connector capacity of Operator: MapOp(ID:8) New value: 5 Old value: 1
```

Some analysis to explain the meaning of the log information:

AutoTune displays common status log information at INFO level. However, when AutoTune detects a bottleneck in the dataset pipeline, it will try to modify the parameters of dataset pipeline ops, and display this analysis log information at WARNING level.

The log messages show the initial progress of AutoTune. The initial configuration of the dataset pipeline is suboptimal.
Thus, AutoTune increases the number of parallel workers from 8 to 10 for MnistOp and increases the prefetch size from 1 to 5 for MapOp(ID:8).
After tuning the configuration of the dataset pipeline, the step time is reduced.

At the end of training with AutoTune, an improved configuration is created by AutoTune.

### Before next training

Before starting the next training process, users can apply the recommended configuration changes to the dataset Python scripts.
This allows the dataset pipeline to be run at an improved speed from the beginning of the training process.

By the way, MindSpore also provides APIs to set the global value of num_parallel_workers and prefetch_size.

Please refer to [mindspore.dataset.config](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.dataset.config.html):

- [mindspore.dataset.config.set_num_parallel_workers](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.dataset.config.html#mindspore.dataset.config.set_num_parallel_workers)
- [mindspore.dataset.config.set_prefetch_size](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.dataset.config.html#mindspore.dataset.config.set_prefetch_size)

