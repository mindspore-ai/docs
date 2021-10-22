# Application of Single-Node Tensor Cache

`Ascend` `GPU` `CPU` `Data Preparation`

<!-- TOC -->

- [Application of Single-Node Tensor Cache](#application-of-single-node-tensor-cache)
    - [Overview](#overview)
    - [Quick Start](#quick-start)
    - [Best Practices](#best-practices)
        - [Using Cache to Speed Up ResNet Evaluation During Training](#using-cache-to-speed-up-resnet-evaluation-during-training)
        - [Using Cache to Speed Up Training with Datasets on NFS](#using-cache-to-speed-up-training-with-datasets-on-nfs)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_en/enable_cache.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## Overview

If you need to repeatedly access remote datasets or read datasets from disks, you can use the single-node cache operator to cache datasets in the local memory to accelerate dataset reading.

This tutorial demonstrates how to use the single-node cache service, and shows several best practices of using cache to improve the performance of network training or evaluating.

## Quick Start

1. Configuring the Environment

   Before using the cache service, you need to install MindSpore and set related environment variables. The Conda environment is used as an example. The setting method is as follows:

   ```text
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{path_to_conda}/envs/{your_env_name}/lib/python3.7/site-packages/mindspore:{path_to_conda}/envs/{your_env_name}/lib/python3.7/site-packages/mindspore/lib
   export PATH=$PATH:{path_to_conda}/envs/{your_env_name}/bin
   ```

2. Starting the Cache Server

   Before using the single-node cache service, you need to start the cache server.

   ```text
   $ cache_admin --start
   Cache server startup completed successfully!
   The cache server daemon has been created as process id 10394 and is listening on port 50052

   Recommendation:
   Since the server is detached into its own daemon process, monitor the server logs (under /tmp/mindspore/cache/log) for any issues that may happen after startup
   ```

   If the system displays a message indicating that the `libpython3.7m.so.1.0` file cannot be found, search for the file path in the virtual environment and set environment variables.

   ```text
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{path_to_conda}/envs/{your_env_name}/lib
   ```

3. Creating a Cache Session

   If no cache session exists on the cache server, a cache session needs to be created to obtain the cache session ID.

   ```text
   $ cache_admin -g
   Session created for server on port 50052: 1493732251
   ```

   The cache session ID is randomly allocated by the server.

4. Creating a Cache Instance

   Create the Python script `my_training_script.py`, use the `DatasetCache` API to define a cache instance named `some_cache` in the script, and specify the `session_id` parameter to a cache session ID created in the previous step.

   ```python
   import mindspore.dataset as ds

   some_cache = ds.DatasetCache(session_id=1493732251, size=0, spilling=False)
   ```

5. Inserting a Cache Instance

   The following uses the CIFAR-10 dataset as an example. Before running the sample, download and store the CIFAR-10 dataset by referring to [Loading Dataset](https://www.mindspore.cn/docs/programming_guide/en/r1.5/dataset_loading.html#cifar-10-100-dataset). The directory structure is as follows:

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

   ```text
   $ cache_admin --list_sessions
   Listing sessions for server on port 50052

        Session    Cache Id  Mem cached  Disk cached  Avg cache size  Numa hit
     1493732251  3618046178       5          n/a          12442         5
   ```

6. Destroying a Cache Session

   After the training is complete, you can destroy the current cache and release the memory.

   ```text
   $ cache_admin --destroy_session 1493732251
   Drop session successfully for server on port 50052
   ```

   The preceding command is used to destroy the cache whose session ID is 1493732251.

7. Stopping the Cache Server

   After using the cache server, you can stop the cache server. This operation will destroy all cache sessions on the current server and release the memory.

   ```text
   $ cache_admin --stop
   Cache server on port 50052 has been stopped successfully.
   ```

## Best Practices

### Using Cache to Speed Up ResNet Evaluation During Training

For a complex network, epoch training usually needs to be performed for dozens or even hundreds of times. Before training, it is difficult to know when a model can achieve required accuracy in epoch training. Therefore, the accuracy of the model is usually validated at a fixed epoch interval during training and the corresponding model is saved. After the training is completed, users can quickly select the optimal model by viewing the change of the corresponding model accuracy.

Therefore, the performance of evaluation during training will have a great impact on the total end-to-end time required. In this section, we will show an example of leveraging the cache service and caching data after augmentation in Tensor format in memory to speed up the evaluation procedure.

The inference data processing procedure usually does not contain random operations. For example, the dataset processing in ResNet50 evaluation only contains augmentations like `Decode`, `Resize`, `CenterCrop`, `Normalize`, `HWC2CHW`, `TypeCast`. Therefore, it's usually better to inject cache after the last augmentation step and directly cache data that's fully augmented, to minimize repeated computations and to yield the most performance benefits. In this section, we will follow this approach and take ResNet as an example.

For the complete sample code, please refer to [ResNet](https://gitee.com/mindspore/models/tree/master/official/cv/resnet) in ModelZoo.

1. Create a Shell script named `cache_util.sh` for cache management:

   ```bash
   bootup_cache_server()
   {
     echo "Booting up cache server..."
     result=$(cache_admin --start 2>&1)
     echo "${result}"
   }

   generate_cache_session()
   {
     result=$(cache_admin -g | awk 'END {print $NF}')
     echo "${result}"
   }
   ```

   > Complete sample code: [cache_util.sh](https://gitee.com/mindspore/docs/blob/r1.5/docs/sample_code/cache/cache_util.sh)

2. In the Shell script for starting the distributed training i.e., `run_distributed_train.sh`, start a cache server for evaluation during training scenarios and generate a cache session, saved in `CACHE_SESSION_ID` Shell variable:

   ```bash
   source cache_util.sh

   if [ "x${RUN_EVAL}" == "xTrue" ]
   then
     bootup_cache_server
     CACHE_SESSION_ID=$(generate_cache_session)
   fi
   ```

3. Pass the `CACHE_SESSION_ID` as well as other arguments when start the Python training script:

   ```text
   python train.py \
   --net=$1 \
   --dataset=$2 \
   --run_distribute=True \
   --device_num=$DEVICE_NUM \
   --dataset_path=$PATH2 \
   --run_eval=$RUN_EVAL \
   --eval_dataset_path=$EVAL_DATASET_PATH \
   --enable_cache=True \
   --cache_session_id=$CACHE_SESSION_ID \
   &> log &
   ```

4. In Python training script `train.py`, use the following code to receive `cache_session_id` that's passed in and use it when defining a eval dataset `eval_dataset`:

   ```python
   import argparse

   parser.add_argument('--enable_cache',
       type=ast.literal_eval,
       default=False,
       help='Caching the eval dataset in memory to speedup evaluation, default is False.')
   parser.add_argument('--cache_session_id',
       type=str,
       default="",
       help='The session id for cache service.')
   args_opt = parser.parse_args()

   eval_dataset = create_dataset(
       dataset_path=args_opt.eval_dataset_path,
       do_train=False,
       batch_size=config.batch_size,
       target=target,
       enable_cache=args_opt.enable_cache,
       cache_session_id=args_opt.cache_session_id)
   ```

5. In Python `dataset.py` script which creates the dataset processing pipeline，create a `DatasetCache` instance according to `enable_cache` and `cache_session_id` arguments, and inject the cache instance after the last step of data augmentation, i.e., after `TyepCast`:

   ```python
   def create_dataset2(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend", distribute=False, enable_cache=False, cache_session_id=None):
   ...
       if enable_cache:
           if not cache_session_id:
               raise ValueError("A cache session_id must be provided to use cache.")
           eval_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
           data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8, cache=eval_cache)
       else:
           data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
   ```

6. Execute the training script:

   ```text
   ...
   epoch: 40, acc: 0.5665486653645834, eval_cost:30.54
   epoch: 41, acc: 0.6212361653645834, eval_cost:2.80
   epoch: 42, acc: 0.6523844401041666, eval_cost:3.77
   ...
   ```

   By default, the evaluation starts after the 40th epoch, and `eval_cost` shows how much time it costs for each evaluation run, measured by seconds.

   The following table compares the average evaluation time with/without cache:

   ```text
   |                            | without cache | with cache |
   | -------------------------- | ------------- | ---------- |
   | 4p, resnet50, imagenet2012 | 10.59s        | 3.62s      |
   ```

   On Ascend machine with 4 parallel pipelines, it generally takes around 88 seconds for each training epoch and ResNet training usually requires 90 epochs. Therefore, using cache can shorten the total end-to-end time from 8849 seconds to 8101 seconds, thus bringing 348 seconds total time reduction.

7. After the training run is completed, you can destroy the current cache and release the memory:

   ```text
   $ cache_admin --stop
   Cache server on port 50052 has been stopped successfully.
   ```

### Using Cache to Speed Up Training with Datasets on NFS

To share a large dataset across multiple servers, many users resort to NFS (Network File System) to store their datasets (Please check [Huawei cloud - Creating an NFS Shared Directory on ECS](https://support.huaweicloud.com/intl/en-us/usermanual-functiongraph/functiongraph_01_0561.html) for how to setup and config an NFS server).

However, due to the fact that the cost of accessing NFS is usually large, running training with a dataset located on NFS is relatively slow. To improve training performance for this scenario, we can leverage cache service to cache the dataset in the form of Tensor in memory. After caching, the following training epochs can directly access from memory, thus avoiding costly remote dataset access.

Note that typically after reading the dataset, certain random operations such as `RandomCropDecodeResize` would be performed in the dataset processing procedure. Caching after these random operations would result in the loss of randomness of the data, and therefore affect the final accuracy. As a result, we choose to directly cache the source dataset. In this section, we will follow this approach and take MobileNetV2 as an example.

For the complete sample code, please refer to [MobileNetV2](https://gitee.com/mindspore/models/tree/master/official/cv/mobilenetv2)  in ModelZoo.

1. Create a Shell script namely `cache_util.sh` for cache management:

   ```bash
   bootup_cache_server()
   {
     echo "Booting up cache server..."
     result=$(cache_admin --start 2>&1)
     echo "${result}"
   }

   generate_cache_session()
   {
     result=$(cache_admin -g | awk 'END {print $NF}')
     echo "${result}"
   }
   ```

   > Complete sample code: [cache_util.sh](https://gitee.com/mindspore/docs/blob/r1.5/docs/sample_code/cache/cache_util.sh)

2. In the Shell script for starting the distributed training with NFS dataset i.e., `run_train_nfs_cache.sh`, start a cache server for scenarios where dataset is on NFS. Then generate a cache session, saved in `CACHE_SESSION_ID` Shell variable:

   ```bash
   source cache_util.sh

   bootup_cache_server
   CACHE_SESSION_ID=$(generate_cache_session)
   ```

3. Pass the `CACHE_SESSION_ID` as well as other arguments when start the Python training script:

   ```text
   python train.py \
   --platform=$1 \
   --dataset_path=$5 \
   --pretrain_ckpt=$PRETRAINED_CKPT \
   --freeze_layer=$FREEZE_LAYER \
   --filter_head=$FILTER_HEAD \
   --enable_cache=True \
   --cache_session_id=$CACHE_SESSION_ID \
   &> log$i.log &
   ```

4. In the `train_parse_args()` function of Python argument-parsing script `args.py`, use the following code to receive `cache_session_id` that's passed in:

   ```python
   import argparse

   def train_parse_args():
   ...
       train_parser.add_argument('--enable_cache',
           type=ast.literal_eval,
           default=False,
           help='Caching the dataset in memory to speedup dataset processing, default is False.')
       train_parser.add_argument('--cache_session_id',
           type=str,
           default="",
           help='The session id for cache service.')
   train_args = train_parser.parse_args()
   ```

   In Python training script`train.py`，call `train_parse_args()` to parse the arguments that's passed in such as `cache_session_id`, and use it when defining the training dataset:

   ```python
   from src.args import train_parse_args
   args_opt = train_parse_args()

   dataset = create_dataset(
       dataset_path=args_opt.dataset_path,
       do_train=True,
       config=config,
       enable_cache=args_opt.enable_cache,
       cache_session_id=args_opt.cache_session_id)
   ```

5. In Python `dataset.py` script which creates the dataset processing pipeline，create a `DatasetCache` instance according to `enable_cache` and `cache_session_id` arguments, and inject the cache instance directly after the `ImageFolderDataset`:

   ```python
   def create_dataset(dataset_path, do_train, config, repeat_num=1, enable_cache=False, cache_session_id=None):
   ...
       if enable_cache:
           nfs_dataset_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
       else:
           nfs_dataset_cache = None

       if config.platform == "Ascend":
           rank_size = int(os.getenv("RANK_SIZE", '1'))
           rank_id = int(os.getenv("RANK_ID", '0'))
           if rank_size == 1:
               data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True, cache=nfs_dataset_cache)
           else:
               data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True, num_shards=rank_size, shard_id=rank_id, cache=nfs_dataset_cache)
   ```

6. Execute the training run via `run_train_nfs_cache.sh`:

   ```text
   epoch: [  0/ 200], step:[ 2134/ 2135], loss:[4.682/4.682], time:[3364893.166], lr:[0.780]
   epoch time: 3384387.999, per step time: 1585.193, avg loss: 4.682
   epoch: [  1/ 200], step:[ 2134/ 2135], loss:[3.750/3.750], time:[430495.242], lr:[0.724]
   epoch time: 431005.885, per step time: 201.876, avg loss: 4.286
   epoch: [  2/ 200], step:[ 2134/ 2135], loss:[3.922/3.922], time:[420104.849], lr:[0.635]
   epoch time: 420669.174, per step time: 197.035, avg loss: 3.534
   epoch: [  3/ 200], step:[ 2134/ 2135], loss:[3.581/3.581], time:[420825.587], lr:[0.524]
   epoch time: 421494.842, per step time: 197.421, avg loss: 3.417
   ...
   ```

   The following table compares the average epoch time with/without cache:

   ```text
   | 4p, MobileNetV2, imagenet2012            | without cache | with cache |
   | ---------------------------------------- | ------------- | ---------- |
   | first epoch time                         | 1649s         | 3384s      |
   | average epoch time (exclude first epoch) | 458s          | 421s       |
   ```

   With cache, the first epoch time increases significantly due to cache writing overhead, but all later epochs can benefit from caching the dataset in memory. Therefore, the more epochs, the more cache case shows benefits due to per-step-time savings.

   MobileNetV2 generally requires 200 epochs in total, therefore, using cache can shorten the total end-to-end time from 92791 seconds to 87163 seconds, thus bringing 5628 seconds total time reduction.

7. After the training run is completed, you can destroy the current cache and release the memory:

   ```text
   $ cache_admin --stop
   Cache server on port 50052 has been stopped successfully.
   ```
