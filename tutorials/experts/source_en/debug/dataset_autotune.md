# Dataset AutoTune for Dataset Pipeline

`Ascend` `GPU` `Data Preparation`

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/debug/enable_dataset_autotune.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore provides a tool named Dataset AutoTune for optimizing dataset.
The Dataset AutoTune can automatically tune Dataset pipelines to improve performance.

This feature can automatically detect a bottleneck operator in the dataset pipeline and respond by automatically adjusting tunable parameters for dataset ops, like increasing the number of parallel workers or updating the prefetch size of dataset ops.

![autotune](images/autotune.png)

With Dataset AutoTune enabled, MindSpore will sample dataset statistics at a given interval, which is tuneable by the user.

Once Dataset AutoTune collects enough information, it will analyze whether the performance bottleneck is on the dataset side or not.
If so, it will adjust the parallelism and speedup the dataset pipeline.
If not, Dataset AutoTune will also try to reduce the memory usage of the dataset pipeline to release memory for CPU.

> Dataset AutoTune is disabled by default.

## Enable Dataset AutoTune

To enable Dataset AutoTune and not save the optimized dataset pipeline:

```python
import mindspore.dataset as ds
ds.config.set_enable_autotune(True)
```

To enable Dataset AutoTune plus save the optimized dataset pipeline in a configuration file:

```python
import mindspore.dataset as ds
ds.config.set_enable_autotune(True, "/path/to/autotune_out")
```

## Tuning Interval for Dataset AutoTune

The frequency at which Dataset AutoTune will adjust the dataset pipeline can be customized.
To set the tuning interval in steps:

```python
import mindspore.dataset as ds
ds.config.set_autotune_interval(100)
```

> To set the tuning interval to be after every epoch, set the tuning interval to 0.

To query the tuning interval for dataset pipeline autotuning:

```python
import mindspore.dataset as ds
print("tuning interval:", ds.config.get_autotune_interval())
```

## Constraints

- Both Dataset Profiling and Dataset AutoTune cannot be enabled concurrently, otherwise it will lead to unwork of Dataset AutoTune or Profiling. If both of them are enabled at the same time, a warning message will prompt the user to check whether there is a mistake. Please make sure Profiling is disabled when using Dataset AutoTune.
- [Offload for Dataset](https://www.mindspore.cn/docs/en/master/design/dataset_offload.html) and Dataset AutoTune are enabled simultaneously, if any dataset node has been offloaded for hardware acceleration, the optimized dataset pipeline configuration file will not be stored and a warning will be logged, because the dataset pipeline that is actually running is not the predefined one.
- If the Dataset pipeline consists of a node that does not support deserialization (e.g. user-defined Python functions, GeneratorDataset), any attempt to deserialize the saved optimized dataset pipeline configuration file will report an error. In this case, it is recommended to modify the script of dataset pipeline manually based on the contents of the tuning figuration files to achieve the purpose of acceleration.

## Example

Take ResNet training as example.

### Dataset AutoTune Config

To enable Dataset AutoTune, only one statement is needed.

```python
# dataset.py of ResNet in ModelZoo
# models/official/cv/resnet/src/dataset.py

def create_dataset(...)
    """
    create dataset for train or test
    """
    # enable Dataset AutoTune
    ds.config.set_enable_autotune(True, "/path/to/autotune_out")

    # define dataset
    data_set = ds.Cifar10Dataset(data_path)
    ...
```

### Start Training

Start the training process as described in [resnet/README.md](https://gitee.com/mindspore/models/blob/master/official/cv/resnet/README.md). Dataset AutoTune will display its analysis result through LOG messages.

```text
[INFO] [auto_tune.cc:73 LaunchThread] Launching Dataset AutoTune thread
[INFO] [auto_tune.cc:35 Main] Dataset AutoTune thread has started.
[INFO] [auto_tune.cc:191 RunIteration] Run Dataset AutoTune at epoch #1
[INFO] [auto_tune.cc:203 RecordPipelineTime] Epoch #1, Average Pipeline time is 21.6624 ms. The avg pipeline time for all epochs is 21.6624ms
[INFO] [auto_tune.cc:231 IsDSaBottleneck] Epoch #1, Device Connector Size: 0.0224, Connector Capacity: 1, Utilization: 2.24%, Empty Freq: 97.76%
epoch: 1 step: 1875, loss is 1.1544309
epoch time: 72110.166 ms, per step time: 38.459 ms

[WARNING] [auto_tune.cc:236 IsDSaBottleneck] Utilization: 2.24% < 75% threshold, dataset pipeline performance needs tuning.
[WARNING] [auto_tune.cc:297 Analyse] Op (MapOp(ID:3)) is slow, input connector utilization=0.975806, output connector utilization=0.298387, diff= 0.677419 > 0.35 threshold.
[WARNING] [auto_tune.cc:253 RequestNumWorkerChange] Added request to change "num_parallel_workers" of Operator: MapOp(ID:3)From old value: [2] to new value: [4].
[WARNING] [auto_tune.cc:309 Analyse] Op (BatchOp(ID:2)) getting low average worker cpu utilization 1.64516% < 35% threshold.
[WARNING] [auto_tune.cc:263 RequestConnectorCapacityChange] Added request to change "prefetch_size" of Operator: BatchOp(ID:2)From old value: [1] to new value: [5].
epoch: 2 step: 1875, loss is 0.64530635
epoch time: 24519.360 ms, per step time: 13.077 ms

[WARNING] [auto_tune.cc:236 IsDSaBottleneck] Utilization: 0.0213516% < 75% threshold, dataset pipeline performance needs tuning.
[WARNING] [auto_tune.cc:297 Analyse] Op (MapOp(ID:3)) is slow, input connector utilization=1, output connector utilization=0, diff= 1 > 0.35 threshold.
[WARNING] [auto_tune.cc:253 RequestNumWorkerChange] Added request to change "num_parallel_workers" of Operator: MapOp(ID:3)From old value: [4] to new value: [6].
[WARNING] [auto_tune.cc:309 Analyse] Op (BatchOp(ID:2)) getting low average worker cpu utilization 4.39062% < 35% threshold.
[WARNING] [auto_tune.cc:263 RequestConnectorCapacityChange] Added request to change "prefetch_size" of Operator: BatchOp(ID:2)From old value: [5] to new value: [9].
epoch: 3 step: 1875, loss is 0.9806979
epoch time: 17116.234 ms, per step time: 9.129 ms

...

[INFO] [profiling.cc:703 Stop] MD Autotune is stopped.
[INFO] [auto_tune.cc:52 Main] Dataset AutoTune thread is finished.
[INFO] [auto_tune.cc:53 Main] Printing final tree configuration
[INFO] [auto_tune.cc:66 PrintTreeConfiguration] CifarOp(ID:5) num_parallel_workers: 2 prefetch_size: 2
[INFO] [auto_tune.cc:66 PrintTreeConfiguration] MapOp(ID:4) num_parallel_workers: 1 prefetch_size: 2
[INFO] [auto_tune.cc:66 PrintTreeConfiguration] MapOp(ID:3) num_parallel_workers: 10 prefetch_size: 2
[INFO] [auto_tune.cc:66 PrintTreeConfiguration] BatchOp(ID:2) num_parallel_workers: 8 prefetch_size: 17
[INFO] [auto_tune.cc:55 Main] Suggest to set proper num_parallel_workers for each Operation or use global setting API: mindspore.dataset.config.set_num_parallel_workers
[INFO] [auto_tune.cc:57 Main] Suggest to choose maximum prefetch_size from tuned result and set by global setting API: mindspore.dataset.config.set_prefetch_size
```

Some analysis to explain the meaning of the log information:

- **How to check process of Dataset AutoTune:**

   Dataset AutoTune displays common status log information at INFO level. However, when AutoTune detects a bottleneck in the dataset pipeline, it will try to modify the parameters of dataset pipeline ops, and display this analysis log information at WARNING level.

- **How to read LOG messages:**

  The initial configuration of the dataset pipeline is suboptimal (Utilization Device Connector is low).

  ```text
  [INFO] [auto_tune.cc:231 IsDSaBottleneck] Epoch #1, Device Connector Size: 0.0224, Connector Capacity: 1, Utilization: 2.24%, Empty Freq: 97.76%
  [WARNING] [auto_tune.cc:236 IsDSaBottleneck] Utilization: 2.24% < 75% threshold, dataset pipeline performance needs tuning.
  ```

  Then, Dataset AutoTune increases the number of parallel workers from 2 to 4 for MapOp(ID:3) and increases the prefetch size from 1 to 5 for BatchOp(ID:2).

  ```text
  [WARNING] [auto_tune.cc:297 Analyse] Op (MapOp(ID:3)) is slow, input connector utilization=0.975806, output connector utilization=0.298387, diff= 0.677419 > 0.35 threshold.
  [WARNING] [auto_tune.cc:253 RequestNumWorkerChange] Added request to change "num_parallel_workers" of Operator: MapOp(ID:3)From old value: [2] to new value: [4].
  [WARNING] [auto_tune.cc:309 Analyse] Op (BatchOp(ID:2)) getting low average worker cpu utilization 1.64516% < 35% threshold.
  [WARNING] [auto_tune.cc:263 RequestConnectorCapacityChange] Added request to change "prefetch_size" of Operator: BatchOp(ID:2)From old value: [1] to new value: [5].
  ```

  After tuning the configuration of the dataset pipeline, the step time is reduced.

  ```text
  epoch: 1 step: 1875, loss is 1.1544309
  epoch time: 72110.166 ms, per step time: 38.459 ms
  epoch: 2 step: 1875, loss is 0.64530635
  epoch time: 24519.360 ms, per step time: 13.077 ms
  epoch: 3 step: 1875, loss is 0.9806979
  epoch time: 17116.234 ms, per step time: 9.129 ms
  ```

  At the end of training, an improved configuration is created by Dataset AutoTune.
  For num_parallel_workers, Dataset AutoTune suggests to set new value for each Operation or using global setting API.
  For prefetch_size, Dataset AutoTune suggests to choose the maximum value and set by global setting API.

  ```text
  [INFO] [auto_tune.cc:66 PrintTreeConfiguration] CifarOp(ID:5) num_parallel_workers: 2 prefetch_size: 2
  [INFO] [auto_tune.cc:66 PrintTreeConfiguration] MapOp(ID:4) num_parallel_workers: 1 prefetch_size: 2
  [INFO] [auto_tune.cc:66 PrintTreeConfiguration] MapOp(ID:3) num_parallel_workers: 10 prefetch_size: 2
  [INFO] [auto_tune.cc:66 PrintTreeConfiguration] BatchOp(ID:2) num_parallel_workers: 8 prefetch_size: 17
  [INFO] [auto_tune.cc:55 Main] Suggest to set proper num_parallel_workers for each Operation or use global setting API: mindspore.dataset.config.set_num_parallel_workers
  [INFO] [auto_tune.cc:57 Main] Suggest to choose maximum prefetch_size from tuned result and set by global setting API: mindspore.dataset.config.set_prefetch_size
  ```

### The Saved AutoTune Recommended Configuration

Since Dataset AutoTune was enabled to generate an optimized dataset pipeline, the optimized dataset pipeline can be serialized (by passing in the 'filepath_prefix' parameter) and saved to the JSON configuration file.

After passing string to `filepath_prefix`, AutoTune will automatically generate JSON files corresponding to the device number according to the current training mode in a standalone or distributed.

For example, let `filepath_prefix='autotune_out'`.

- In distributed training on 4 devices, AutoTune will generate 4 tuning files: autotune_out_0.json, autotune_out_1.json, autotune_out_2.json, autotune_out_3.json, corresponding to the configuration of the data pipeline of the 4 devices.
- In a standalone training on 1 device, AutoTune will generate autotune_out_0.json, which corresponds to the configuration of the data pipeline on this device.

Example of the JSON configuration file:

```text
{
    "remark": "The following file has been auto-generated by the Dataset AutoTune.",
    "summary": [
        "CifarOp(ID:5)       (num_parallel_workers: 2, prefetch_size: 2)",
        "MapOp(ID:4)         (num_parallel_workers: 1, prefetch_size: 2)",
        "MapOp(ID:3)         (num_parallel_workers: 10, prefetch_size: 2)",
        "BatchOp(ID:2)       (num_parallel_workers: 8, prefetch_size: 17)"
    ],
    "tree": {...}
}
```

The file starts with a summary of the configuration and then is followed by the actual pipeline (`tree`) information. The file is loadable using the deserialization API `mindspore.dataset.deserialize`.

Notes on the JSON configuration file:

- Non-parallel dataset operations will show `NA` for `num_parallel_workers`.

### Load AutoTune Configuration

If Dataset AutoTune generated an optimized pipeline configuration file, use deserialize support to load the dataset pipeline:

```python
import mindspore.dataset as ds
new_dataset = ds.deserialize("/path/to/autotune_out_0.json")
```

The `new_dataset` is the tuned dataset object containing operations from Cifar to Batch as shown in the JSON content above.

### Before Next Training

Before starting the next training process, user can adjust the code of the loading part in the dataset according to the recommended configuration in the output of the automatic data acceleration module.

This allows the dataset pipeline to be run at an improved speed from the beginning of the training process.

By the way, MindSpore also provides APIs to set the global value of num_parallel_workers and prefetch_size.

Please refer to [mindspore.dataset.config](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.config.html):

- [mindspore.dataset.config.set_num_parallel_workers](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.config.html#mindspore.dataset.config.set_num_parallel_workers)
- [mindspore.dataset.config.set_prefetch_size](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.config.html#mindspore.dataset.config.set_prefetch_size)
