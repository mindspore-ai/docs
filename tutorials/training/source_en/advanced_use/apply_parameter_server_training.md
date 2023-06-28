# Training with Parameter Server

`Linux` `Ascend` `GPU` `Model Training` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.1/tutorials/training/source_en/advanced_use/apply_parameter_server_training.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

A parameter server is a widely used architecture in distributed training. Compared with the synchronous AllReduce training method, a parameter server has better flexibility, scalability, and node failover capabilities. Specifically, the parameter server supports both synchronous and asynchronous SGD training algorithms. In terms of scalability, model computing and update are separately deployed in the worker and server processes, so that resources of the worker and server can be independently scaled out and in horizontally. In addition, in an environment of a large-scale data center, various failures often occur in a computing device, a network, and a storage device, and consequently some nodes are abnormal. However, in an architecture of a parameter server, such a failure can be relatively easily handled without affecting a training job.

In the parameter server implementation of MindSpore, the open-source [ps-lite](https://github.com/dmlc/ps-lite) is used as the basic architecture. Based on the remote communication capability provided by the [ps-lite](https://github.com/dmlc/ps-lite) and abstract Push/Pull primitives, the distributed training algorithm of the synchronous SGD is implemented. In addition, with the high-performance collective communication library in Ascend and GPU(HCCL and NCCL), MindSpore also provides the hybrid training mode of parameter server and AllReduce. Some weights can be stored and updated through the parameter server, and other weights are still trained through the AllReduce algorithm.

The ps-lite architecture consists of three independent components: server, worker, and scheduler. Their functions are as follows:

- Server: saves model weights and backward computation gradients, and updates the model using gradients pushed by workers.

- Worker: performs forward and backward computation on the network. The gradient value for backward computation is uploaded to a server through the `Push` API, and the model updated by the server is downloaded to the worker through the `Pull` API.

- Scheduler: establishes the communication relationship between the server and worker.

## Preparations

The following describes how to use parameter server to train LeNet on Ascend 910:

### Training Script Preparation

Learn how to train a LeNet using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) by referring to <https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/cv/lenet>.

### Parameter Setting

1. First of all, use `mindspore.context.set_ps_context(enable_ps=True)` to enable Parameter Server training mode.

    - This method should be called before `mindspore.communication.management.init()`.
    - If you don't call this method, the [Environment Variable Setting](https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/apply_parameter_server_training.html#environment-variable-setting) below will not take effect.
    - Use `mindspore.context.reset_ps_context()` to disable Parameter Server training mode.

2. In this training mode, you can use either of the following methods to control whether the training parameters are updated by the Parameter Server and whether the training parameters are initialized on Worker or Server:

    - Use `mindspore.nn.Cell.set_param_ps()` to set all weight recursions of `nn.Cell`.
    - Use `mindspore.Parameter.set_param_ps()` to set the weight.
    - The size of the weight which is updated by Parameter Server should not exceed INT_MAX(2^31 - 1) bytes.
    - The interface `set_param_ps` can receive a `bool` parameter:`init_in_server`, indicating whether this training parameter is initialized on the Server side. `init_in_server` defaults to `False`, indicating that this training parameter is initialized on Worker. Currently, only the training parameter `embedding_table` of the `EmbeddingLookup` operator is supported to be initialized on Server side to solve the problem of insufficient memory caused by the initialization of a large shape `embedding_table` on Worker. The `EmbeddingLookup` operator's `target` attribute needs to be set to 'CPU'. The training parameter initialized on the Server side will no longer be synchronized to Worker. If it involves multi-Server training and saves CheckPoint, each Server will save a CheckPoint after the training.

3. On the basis of the [original training script](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/lenet/train.py), set all LeNet model weights to be trained on the parameter server:

    ```python
    context.set_ps_context(enable_ps=True)
    network = LeNet5(cfg.num_classes)
    network.set_param_ps()
    ```

4. [optional configuration] For a large shape `embedding_table`, because the device can not store a full amount of `embedding_table`. You can configure the `vocab_cache_size` of [EmbeddingLookup operator](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/nn/mindspore.nn.EmbeddingLookup.html) to enable the cache function of `EmbeddingLookup` in the Parameter Server training mode. The `vocab_cache_size` of `embedding_table` is trained on device, and a full amount of `embedding_table` is stored in the Server. The `embedding_table` of next batch is swapped to the cache in advance, and the expired `embedding_table` is put back to the Server when the cache cannot be placed, to achieve the purpose of improving the training performance. Detailed network training script can be referred to <https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/recommend/wide_and_deep>.

   ```python
    context.set_auto_parallel_context(full_batch=True)
    network = Net()
    model = Model(network)
    model.train(epoch, train_dataset, dataset_sink_mode=True)
    ```

    In the information:

    - `dataset_sink_mode`: whether to enable the sink mode of dataset or not. When `True`, it indicates enable, and pass the data through dataset channel. It must be set to `True` in this scenario.
    - `full_batch`: whether to load the dataset in full or not. When `True`, it indicates full load, and data of each device is the same. It must be set to `True` in the multi-workers scenario.

### Environment Variable Setting

MindSpore reads environment variables to control parameter server training. The environment variables include the following options (all scripts of `MS_SCHED_HOST` and `MS_SCHED_PORT` must be consistent):

```bash
export PS_VERBOSE=1                   # Print ps-lite log
export MS_SERVER_NUM=1                # Server number
export MS_WORKER_NUM=1                # Worker number
export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
export MS_SCHED_PORT=XXXX             # Scheduler port
export MS_ROLE=MS_SCHED               # The role of this process: MS_SCHED represents the scheduler, MS_WORKER represents the worker, MS_PSERVER represents the Server
```

## Training

1. Shell scripts

    Provide the shell scripts corresponding to the worker, server, and scheduler roles to start training:

    `Scheduler.sh`:

    ```bash
    #!/bin/bash
    export PS_VERBOSE=1
    export MS_SERVER_NUM=1
    export MS_WORKER_NUM=1
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX
    export MS_SCHED_PORT=XXXX
    export MS_ROLE=MS_SCHED
    python train.py --device_target=Ascend --data_path=path/to/dataset
    ```

    `Server.sh`:

    ```bash
    #!/bin/bash
    export PS_VERBOSE=1
    export MS_SERVER_NUM=1
    export MS_WORKER_NUM=1
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX
    export MS_SCHED_PORT=XXXX
    export MS_ROLE=MS_PSERVER
    python train.py --device_target=Ascend --data_path=path/to/dataset
    ```

    `Worker.sh`:

    ```bash
    #!/bin/bash
    export PS_VERBOSE=1
    export MS_SERVER_NUM=1
    export MS_WORKER_NUM=1
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX
    export MS_SCHED_PORT=XXXX
    export MS_ROLE=MS_WORKER
    python train.py --device_target=Ascend --data_path=path/to/dataset
    ```

    Run the following commands separately:

    ```bash
    sh Scheduler.sh > scheduler.log 2>&1 &
    sh Server.sh > server.log 2>&1 &
    sh Worker.sh > worker.log 2>&1 &
    ```

    Start training.

2. Viewing result

    Run the following command to view the communication logs between the server and worker in the `scheduler.log` file:

    ```text
    Bind to role=scheduler, id=1, ip=XXX.XXX.XXX.XXX, port=XXXX
    Assign rank=8 to node role=server, ip=XXX.XXX.XXX.XXX, port=XXXX
    Assign rank=9 to node role=worker, ip=XXX.XXX.XXX.XXX, port=XXXX
    the scheduler is connected to 1 workers and 1 servers
    ```

    The preceding information indicates that the communication between the server, worker, and scheduler is established successfully.

    Check the training result in the `worker.log` file:

    ```text
    epoch: 1 step: 1, loss is 2.302287
    epoch: 1 step: 2, loss is 2.304071
    epoch: 1 step: 3, loss is 2.308778
    epoch: 1 step: 4, loss is 2.301943
    ...
    ```
