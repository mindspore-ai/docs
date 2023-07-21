# Application of Single-Node Tensor Cache

`Linux` `Ascend` `GPU` `CPU` `Data Preparation` `Intermediate` `Expert`

[![View Source On Gitee](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/tutorials/training/source_en/advanced_use/enable_cache.md)

## Overview

If you need to repeatedly access remote datasets or read datasets from disks, you can use the single-node cache operator to cache datasets in the local memory to accelerate dataset reading.

This tutorial demonstrates how to use the single-node cache service to cache data that has been processed with data augmentation.

## Configuring the Environment

Before using the cache service, you need to install MindSpore and set related environment variables. The Conda environment is used as an example. The setting method is as follows:

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{path_to_conda}/envs/{your_env_name}/lib/python3.7/site-packages/mindspore:{path_to_conda}/envs/{your_env_name}/lib/python3.7/site-packages/mindspore/lib
export PATH=$PATH:{path_to_conda}/envs/{your_env_name}/bin
```

## Starting the Cache Server

Before using the single-node cache service, you need to start the cache server.

```shell
$ cache_admin --start
Cache server startup completed successfully!
The cache server daemon has been created as process id 10394 and is listening on port 50052

Recommendation:
Since the server is detached into its own daemon process, monitor the server logs (under /tmp/mindspore/cache/log) for any issues that may happen after startup
```

If the system displays a message indicating that the `libpython3.7m.so.1.0` file cannot be found, search for the file path in the virtual environment and set environment variables.

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{path_to_conda}/envs/{your_env_name}/lib
```

## Creating a Cache Session

If no cache session exists on the cache server, a cache session needs to be created to obtain the cache session ID.

```shell
$ cache_admin -g
Session created for server on port 50052: 1493732251
```

The cache session ID is randomly allocated by the server.

## Creating a Cache Instance

Create the Python script `my_training_script.py`, use the `DatasetCache` API to define a cache instance named `some_cache` in the script, and specify the `session_id` parameter to a cache session ID created in the previous step.

```python
import mindspore.dataset as ds

some_cache = ds.DatasetCache(session_id=1493732251, size=0, spilling=True)
```

## Inserting a Cache Instance

The following uses the CIFAR-10 dataset as an example. Before running the sample, download and store the CIFAR-10 dataset by referring to [Loading Dataset](https://www.mindspore.cn/doc/programming_guide/en/r1.1/dataset_loading.html#cifar-10-100-dataset). The directory structure is as follows:

```text
├─my_training_script.py
└─cifar-10-batches-bin
    ├── batches.meta.txt
    ├── data_batch_1.bin
    ├── data_batch_2.bin
    ├── data_batch_3.bin
    ├── data_batch_4.bin
    ├── data_batch_5.bin
    ├── readme.html
    └── test_batch.bin
```

To cache the enhanced data processed by data augmentation of the map operator, the created `some_cache` instance is used as the input parameter of the `cache` API in the map operator.

```python
import mindspore.dataset.vision.c_transforms as c_vision

dataset_dir = "cifar-10-batches-bin/"
data = ds.Cifar10Dataset(dataset_dir=dataset_dir, num_samples=5, shuffle=False, num_parallel_workers=1)

# apply cache to map
rescale_op = c_vision.Rescale(1.0 / 255.0, -1.0)
data = data.map(input_columns=["image"], operations=rescale_op, cache=some_cache)

num_iter = 0
for item in data.create_dict_iterator(num_epochs=1):  # each data is a dictionary
    # in this example, each dictionary has a key "image"
    print("{} image shape: {}".format(num_iter, item["image"].shape))
    num_iter += 1
```

Run the Python script `my_training_script.py`. The following information is displayed:

```text
0 image shape: (32, 32, 3)
1 image shape: (32, 32, 3)
2 image shape: (32, 32, 3)
3 image shape: (32, 32, 3)
4 image shape: (32, 32, 3)
```

You can run the `cache_admin --list_sessions` command to check whether there are five data records in the current session. If yes, the data is successfully cached.

```shell
$ cache_admin --list_sessions
Listing sessions for server on port 50052

     Session    Cache Id  Mem cached  Disk cached  Avg cache size  Numa hit
  1493732251  3618046178       5          n/a          12442         5
```

## Destroying a Cache Session

After the training is complete, you can destroy the current cache and release the memory.

```shell
$ cache_admin --destroy_session 1493732251
Drop session successfully for server on port 50052
```

The preceding command is used to destroy the cache whose session ID is 1493732251.

## Stopping the Cache Server

After using the cache server, you can stop the cache server. This operation will destroy all cache sessions on the current server and release the memory.

```shell
$ cache_admin --stop
Cache server on port 50052 has been stopped successfully.
```
