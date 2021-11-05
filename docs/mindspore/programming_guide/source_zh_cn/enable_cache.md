# 应用单节点数据缓存

`Ascend` `GPU` `CPU` `数据准备`

<!-- TOC -->

- [应用单节点数据缓存](#应用单节点数据缓存)
    - [概述](#概述)
    - [缓存使用入门](#缓存使用入门)
    - [最佳实践](#最佳实践)
        - [使用缓存加速ResNet训练时推理的性能](#使用缓存加速resnet训练时推理的性能)
        - [使用缓存加速NFS数据集的训练性能](#使用缓存加速nfs数据集的训练性能)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_zh_cn/enable_cache.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>&nbsp;&nbsp;
<a href="https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r1.5/notebook/mindspore_enable_cache.ipynb"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_notebook.png"></a>&nbsp;&nbsp;
<a href="https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL21pbmRzcG9yZV9lbmFibGVfY2FjaGUuaXB5bmI=&imageid=65f636a0-56cf-49df-b941-7d2a07ba8c8c" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_modelarts.png"></a>

## 概述

对于需要重复访问远程的数据集或需要重复从磁盘中读取数据集的情况，可以使用单节点缓存算子将数据集缓存于本地内存中，以加速数据集的读取。

下面，本教程将简单介绍单节点缓存服务的使用方法，并演示几个利用单节点缓存优化训练或推理性能的示例。

## 缓存使用入门

1. 配置环境。

   使用缓存服务前，需要安装MindSpore，并设置相关环境变量。以Conda环境为例，设置方法如下：

   ```text
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{path_to_conda}/envs/{your_env_name}/lib/python3.7/site-packages/mindspore:{path_to_conda}/envs/{your_env_name}/lib/python3.7/site-packages/mindspore/lib
   export PATH=$PATH:{path_to_conda}/envs/{your_env_name}/bin
   ```

2. 启动缓存服务器。

   在使用单节点缓存服务之前，首先需要启动缓存服务器：

   ```text
   $ cache_admin --start
   Cache server startup completed successfully!
   The cache server daemon has been created as process id 10394 and is listening on port 50052

   Recommendation:
   Since the server is detached into its own daemon process, monitor the server logs (under /tmp/mindspore/cache/log) for any issues that may happen after startup
   ```

   若提示找不到`libpython3.7m.so.1.0`文件，尝试在虚拟环境下查找其路径并设置环境变量：

   ```text
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{path_to_conda}/envs/{your_env_name}/lib
   ```

3. 创建缓存会话。

   若缓存服务器中不存在缓存会话，则需要创建一个缓存会话，得到缓存会话id：

   ```text
   $ cache_admin -g
   Session created for server on port 50052: 1493732251
   ```

   缓存会话id由服务器随机分配。

4. 创建缓存实例。

   创建Python脚本`my_training_script.py`，在脚本中使用`DatasetCache` API来定义一个名为`some_cache`的缓存实例，并把上一步中创建的缓存会话id传入`session_id`参数：

   ```python
   import mindspore.dataset as ds

   some_cache = ds.DatasetCache(session_id=1493732251, size=0, spilling=False)
   ```

5. 插入缓存实例。

   下面样例中使用到CIFAR-10数据集。运行样例前，需要参照[数据集加载](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/dataset_loading.html#cifar-10-100)中的方法下载并存放CIFAR-10数据集。目录结构如下：

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

   继续编写Python脚本，在应用数据增强算子时将所创建的`some_cache`作为其`cache`参数传入：

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

   运行Python脚本`my_training_script.py`，得到输出结果：

   ```text
   0 image shape: (32, 32, 3)
   1 image shape: (32, 32, 3)
   2 image shape: (32, 32, 3)
   3 image shape: (32, 32, 3)
   4 image shape: (32, 32, 3)
   ```

   通过`cache_admin --list_sessions`命令可以查看当前会话有五条数据，说明数据缓存成功。

   ```text
   $ cache_admin --list_sessions
   Listing sessions for server on port 50052

        Session    Cache Id  Mem cached  Disk cached  Avg cache size  Numa hit
     1493732251  3618046178       5          n/a          12442         5
   ```

6. 销毁缓存会话。

   在训练结束后，可以选择将当前的缓存销毁并释放内存：

   ```text
   $ cache_admin --destroy_session 1493732251
   Drop session successfully for server on port 50052
   ```

   以上命令将销毁缓存会话id为1493732251的缓存。

7. 关闭缓存服务器。

   使用完毕后，可以选择关闭缓存服务器，该操作将销毁当前服务器中存在的所有缓存会话并释放内存。

   ```text
   $ cache_admin --stop
   Cache server on port 50052 has been stopped successfully.
   ```

## 最佳实践

### 使用缓存加速ResNet训练时推理的性能

在面对复杂网络时，往往需要进行几十甚至几百次的epoch训练。在训练之前，很难掌握在训练到第几个epoch时，模型的精度能满足要求，所以经常会在训练过程中，相隔固定epoch对模型精度进行验证，并保存相应的模型，等训练完毕后，通过查看对应模型精度的变化就能迅速地挑选出相对最优的模型。

因此，训练时推理的性能将很大程度上影响完成整个训练所需要的时间。为了提高训练时推理的性能，我们可以选择使用缓存服务，将经过数据增强处理后的测试集图片以Tensor的形式缓存在内存中。

由于推理的数据处理流程中通常不包含具有随机性的操作，如ResNet50的数据处理流程仅包含`Decode`、`Resize`、`CenterCrop`、`Normalize`、`HWC2CHW`、`TypeCast`等操作，因此通常可以在数据处理的最后一个步骤之后插入缓存，直接缓存经过所有数据增强操作之后的数据，以最大限度的避免重复的计算，获得更好的性能提升。本节将采用这种方法，以ResNet网络为样本，进行示例。

完整示例代码请参考ModelZoo的[ResNet](https://gitee.com/mindspore/models/tree/r1.5/official/cv/resnet)。

1. 创建管理缓存的Shell脚本`cache_util.sh`：

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

   > 直接获取完整样例代码：[cache_util.sh](https://gitee.com/mindspore/docs/blob/r1.5/docs/sample_code/cache/cache_util.sh)

2. 在启动分布式训练的Shell脚本`run_distribute_train.sh`中，为训练时推理的场景开启缓存服务器并生成一个缓存会话保存在Shell变量`CACHE_SESSION_ID`中：

   ```bash
   source cache_util.sh

   if [ "x${RUN_EVAL}" == "xTrue" ]
   then
     bootup_cache_server
     CACHE_SESSION_ID=$(generate_cache_session)
   fi
   ```

3. 在启动Python训练时将`CACHE_SESSION_ID`以及其他参数传入：

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

4. 在Python的训练脚本`train.py`中，通过以下代码接收传入的`cache_session_id`，并在定义推理数据集`eval_dataset`时将其作为参数传入：

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

5. 在定义数据处理流程的Python脚本`dataset.py`中，根据传入的`enable_cache`以及`cache_session_id`参数，创建一个`DatasetCache`的实例并将其插入至最后一个数据增强操作`TypeCast`之后：

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

6. 运行训练时推理的脚本，得到以下结果：

   ```text
   ...
   epoch: 40, acc: 0.5665486653645834, eval_cost:30.54
   epoch: 41, acc: 0.6212361653645834, eval_cost:2.80
   epoch: 42, acc: 0.6523844401041666, eval_cost:3.77
   ...
   ```

   默认情况下从第40个epoch开始进行推理，`eval_cost`展示了每次推理的时间，单位为秒。

   下表展示了Ascend服务器上使用缓存与不使用缓存的平均每次推理时间对比：

   ```text
   |                            | without cache | with cache |
   | -------------------------- | ------------- | ---------- |
   | 4p, resnet50, imagenet2012 | 10.59s        | 3.62s      |
   ```

   若每训练一个epoch用时为88s，以运行90个epoch为例，则使用缓存可以使端到端的训练总用时从8449秒降低至8101秒，共计节省约348秒。

7. 使用完毕后，可以选择关闭缓存服务器：

   ```text
   $ cache_admin --stop
   Cache server on port 50052 has been stopped successfully.
   ```

### 使用缓存加速NFS数据集的训练性能

为了使较大的数据集在多台服务器之间共享，缓解单台服务器的磁盘空间需求，用户通常可以选择使用NFS（Network File System）即网络文件系统来存储数据集（NFS存储服务器的搭建和配置请参考[华为云-NFS存储服务器搭建](https://www.huaweicloud.com/articles/14fe58d0991fb2dfd2633a1772c175fc.html)）。

然而，对于NFS数据集的访问通常开销较大，导致使用NFS数据集进行的训练用时较长。为了提高NFS数据集的训练性能，我们可以选择使用缓存服务，将数据集以Tensor的形式缓存在内存中。经过缓存后，后序的epoch就可以直接从内存中读取数据，避免了访问远程网络存储的开销。

需要注意的是，在训练过程的数据处理流程中，数据集经加载后通常还需要进行一些带有随机性的增强操作，如`RandomCropDecodeResize`，若将缓存添加到该具有随机性的操作之后，将会导致第一次的增强操作结果被缓存下来，后序从缓存服务器中读取的结果均为第一次已缓存的数据，导致数据的随机性丢失，影响训练网络的精度。因此我们可以选择直接在数据集读取算子之后添加缓存。本节将采用这种方法，以MobileNetV2网络为样本，进行示例。

完整示例代码请参考ModelZoo的[MobileNetV2](https://gitee.com/mindspore/models/tree/r1.5/official/cv/mobilenetv2)。

1. 创建管理缓存的Shell脚本`cache_util.sh`：

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

   > 直接获取完整样例代码：[cache_util.sh](https://gitee.com/mindspore/docs/blob/r1.5/docs/sample_code/cache/cache_util.sh)

2. 在启动NFS数据集训练的Shell脚本`run_train_nfs_cache.sh`中，为使用位于NFS上的数据集训练的场景开启缓存服务器并生成一个缓存会话保存在Shell变量`CACHE_SESSION_ID`中：

   ```bash
   CURPATH="${dirname "$0"}"
   source ${CURPATH}/cache_util.sh

   bootup_cache_server
   CACHE_SESSION_ID=$(generate_cache_session)
   ```

3. 在启动Python训练时将`CACHE_SESSION_ID`以及其他参数传入：

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

4. 在Python的参数解析脚本`args.py`的`train_parse_args()`函数中，通过以下代码接收传入的`cache_session_id`：

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

   并在Python的训练脚本`train.py`中调用`train_parse_args()`函数解析传入的`cache_session_id`等参数，并在定义数据集`dataset`时将其作为参数传入。

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

5. 在定义数据处理流程的Python脚本`dataset.py`中，根据传入的`enable_cache`以及`cache_session_id`参数，创建一个`DatasetCache`的实例并将其插入至`ImageFolderDataset`之后：

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

6. 运行`run_train_nfs_cache.sh`，得到以下结果：

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

   下表展示了GPU服务器上使用缓存与不使用缓存的平均每个epoch时间对比：

   ```text
   | 4p, MobileNetV2, imagenet2012            | without cache | with cache |
   | ---------------------------------------- | ------------- | ---------- |
   | first epoch time                         | 1649s         | 3384s      |
   | average epoch time (exclude first epoch) | 458s          | 421s       |
   ```

   可以看到使用缓存后，相比于不使用缓存的情况第一个epoch的完成时间增加了较多，这主要是由于缓存数据写入至缓存服务器的开销导致的。但是，在缓存数据写入之后随后的每个epoch都可以获得较大的性能提升。因此，训练的总epoch数目越多，使用缓存的收益将越明显。

   以运行200个epoch为例，使用缓存可以使端到端的训练总用时从92791秒降低至87163秒，共计节省约5628秒。

7. 使用完毕后，可以选择关闭缓存服务器：

   ```text
   $ cache_admin --stop
   Cache server on port 50052 has been stopped successfully.
   ```
