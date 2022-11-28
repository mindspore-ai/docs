# Distributed Parallel Training Base Sample (CPU)

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/train_cpu.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

This tutorial focuses on how to use MindSpore for data parallel distributed training on CPU platforms to improve training efficiency.
> The complete sample code: [distributed_training_cpu](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training_cpu)

The directory structure is as follows:

```text
└─sample_code
    ├─distributed_training_cpu
    │      resnet.py
    │      resnet50_distributed_training.py
    │      run.sh
```

where `resnet.py` and `resnet50_distributed_training.py` are the training network definition scripts and `run.sh` is the distributed training execution script.

## Preparation

### Downloading the Datasets

This sample is taken [with CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz), consisting of 10 classes of 32*32 color images. Each class contains 6000 images, with a total of 50,000 images in the training set and 10,000 images in the test set images.

Download and unzip the dataset locally, and unzip the folder as `cifar-10-batches-bin`.

### Configuring a Distributed Environment

Data parallelism on CPU is mainly divided into two types of parallelism: single-machine multi-node and multi-machine multi-node (a training process can be understood as a node). Before running the training script, you need to set up the networking environment, mainly the environment variable configuration and the calling of the initialization interface in the training script.

The environment variable configuration is as follows:

```text
export MS_WORKER_NUM=8                # Worker number
export MS_SCHED_HOST=127.0.0.1        # Scheduler IP address
export MS_SCHED_PORT=6667             # Scheduler port
export MS_ROLE=MS_WORKER              # The role of this node: MS_SCHED represents the scheduler, MS_WORKER represents the worker
```

where

- `MS_WORKER_NUM`: denotes the number of worker nodes. In a multi-machine scenario, the number of worker nodes is the sum of worker nodes per machine.
- `MS_SCHED_HOST`: denotes the ip address of the scheduler node.
- `MS_SCHED_PORT`: denotes the service port of the scheduler node, used to receive the ip and service port sent by worker nodes, and then distribute the collected ip and port of all worker nodes down to each worker.
- `MS_ROLE`: denotes the type of the node, worker (MS_WORKER) and scheduler (MS_SCHED) two types. Whether it is single-machine multi-node or multi-machine multi-node, a scheduler node needs to be configured for networking.

The calling of the initialization interface in the training script is as follows:

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
ms.set_ps_context(enable_ssl=False)
init()
```

where

- `ms.set_context(mode=context.GRAPH_MODE, device_target="CPU")`: Specify the mode as graph mode (parallelism is not supported in PyNative mode on CPU) and the device as `CPU`.
- `ms.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)`: Specify the data parallelism mode. `gradients_mean=True` means averaging will be done after gradient normalization. Only summation is supported for gradient normalization on the current CPU.
- `ms.set_ps_context`: Configure secure encrypted communication and enable secure encrypted communication by `ms.set_ps_context(enable_ssl=True)`. Default is `False` to turn off secure encrypted communication.
- `init`: Initialize the node. The completion of initialization indicates successful network formation.

## Loading the Dataset

For distributed training, the dataset is imported as data in parallel. In the following, we introduce the method of importing CIFAR-10 dataset in a data parallel way, taking the CIFAR-10 dataset as an example. `data_path` is the path to the dataset, i.e. the path to the `cifar-10-batches-bin` folder. The sample code is as follows:

```python
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.communication import get_rank, get_group_size

def create_dataset(data_path, repeat_num=1, batch_size=32):
    """Create training dataset"""
    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # get rank_id and rank_size
    rank_size = get_group_size()
    rank_id = get_rank()
    data_set = ds.Cifar10Dataset(data_path, num_shards=rank_size, shard_id=rank_id)

    # define map operations
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize((resize_height, resize_width))
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = transforms.TypeCast(ms.int32)

    c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op, changeswap_op]

    # apply map operations on images
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=c_trans, input_columns="image")

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=10)

    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)

    return data_set
```

Unlike stand-alone, the `num_shards` and `shard_id` parameters need to be passed in when constructing the Cifar10Dataset, corresponding to the number of worker nodes and logical serial number, respectively, which can be obtained through the framework interface as follows:

- `get_group_size`: Obtain the number of worker nodes in the cluster.
- `get_rank`: Obtain the logical serial number of the current worker node in the cluster.

> When loading datasets in data parallel mode, it is recommended to specify the same dataset file for each card. If the datasets loaded for each card are different, it may affect the calculation accuracy.

## Defining the Model

The network definition in data parallel mode is written in the same way with the stand-alone, which can be found in [ResNet Network Sample Script](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/distributed_training_cpu/resnet.py).

The definitions of optimizer, loss function and training model can be found in [Training Model Definition](https://www.mindspore.cn/tutorials/en/master/beginner/train.html).

The reference sample of the full training script code and the training startup code is listed below:

```python
import os
import mindspore as ms
import mindspore.nn as nn
from mindspore.communication import init
from resnet import resnet50

def train_resnet50_with_cifar10(epoch_size=10):
    """Start the training"""
    loss_cb = ms.LossMonitor()
    data_path = os.getenv('DATA_PATH')
    dataset = create_dataset(data_path)
    batch_size = 32
    num_classes = 10
    net = resnet50(batch_size, num_classes)
    loss = SoftmaxCrossEntropyExpand(sparse=True)
    opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    model = ms.Model(net, loss_fn=loss, optimizer=opt)
    model.train(epoch_size, dataset, callbacks=[loss_cb], dataset_sink_mode=True)


if __name__ == "__main__":
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
    ms.set_ps_context(enable_ssl=False)
    init()
    train_resnet50_with_cifar10()
```

> The interfaces `create_dataset` and `SoftmaxCrossEntropyExpand` in script are referenced from [distributed_training_cpu](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/distributed_training_cpu/resnet50_distributed_training.py).
> The interfaces `resnet50` is referenced from [ResNet network sample script](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/distributed_training_cpu/resnet.py).

## Starting the Training

Distributed training is performed on a CPU platform with a single-machine 8-node as an example. Start the training with the following shell script, and command `bash run.sh /dataset/cifar-10-batches-bin`.

```bash
#!/bin/bash
# run data parallel training on CPU

echo "=============================================================================================================="
echo "Please run the script with dataset path, such as: "
echo "bash run.sh DATA_PATH"
echo "For example: bash run.sh /path/dataset"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

export MS_WORKER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8117

# Launch 1 scheduler.
export MS_ROLE=MS_SCHED
python3 resnet50_distributed_training.py >scheduler.txt 2>&1 &
echo "scheduler start success!"

# Launch 8 workers.
export MS_ROLE=MS_WORKER
for((i=0;i<${MS_WORKER_NUM};i++));
do
    python3 resnet50_distributed_training.py >worker_$i.txt 2>&1 &
    echo "worker ${i} start success with pid ${!}"
done
```

where `resnet50_distributed_training.py` is the defined training script.

For a multi-machine, multi-node scenario, the corresponding worker node needs to be started on each machine to participate in the training in this way, but there is only one scheduler node, which only needs to be started on one of the machines (i.e. MS_SCHED_HOST).

> The defined value of MS_WORKER_NUM indicates that the corresponding number of worker nodes need to be started to participate in the training, otherwise the networking is not successful.
>
> Although training scripts are also started for scheduler nodes, the scheduler is mainly used for networking and does not participate in training.

After a period of training, open the worker_0 log and the training information is as follows:

```text
epoch: 1 step: 1, loss is 2.4686084
epoch: 1 step: 2, loss is 2.3278534
epoch: 1 step: 3, loss is 2.4246798
epoch: 1 step: 4, loss is 2.4920032
epoch: 1 step: 5, loss is 2.4324203
epoch: 1 step: 6, loss is 2.432581
epoch: 1 step: 7, loss is 2.319618
epoch: 1 step: 8, loss is 2.439193
epoch: 1 step: 9, loss is 2.2922952
```

## Security Authentication

For CPU security authentication, refer to [GPU Distributed Training Security Authentication](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html#security-authentication).
