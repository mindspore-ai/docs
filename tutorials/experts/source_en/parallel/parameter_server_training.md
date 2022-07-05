# Parameter Server Mode

<a href="https://gitee.com/mindspore/docs/blob/r1.8/tutorials/experts/source_en/parallel/parameter_server_training.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

A parameter server is a widely used architecture in distributed training. Compared with the synchronous AllReduce training method, a parameter server has better flexibility, scalability, and node failover capabilities. Specifically, the parameter server supports both synchronous and asynchronous SGD(Stochastic Gradient Descent) training algorithms. In terms of scalability, model computing and update are separately deployed in the worker and server processes, so that resources of the worker and server can be independently scaled out in horizontally (add or delete resources of the worker and server). In addition, in an environment of a large-scale data center, various failures often occur in a computing device, a network, and a storage device, and consequently some nodes are abnormal. However, in an architecture of a parameter server, such a failure can be relatively easily handled without affecting a training job.

## Basic Principle

In the parameter server implementation of MindSpore, the self-developed communication framework (core) is used as the basic architecture. Based on the remote communication capability provided by the core and abstract Send/Broadcast primitives, the distributed training algorithm of the synchronous SGD is implemented. In addition, with the high-performance collective communication library in Ascend and GPU(HCCL and NCCL), MindSpore also provides the hybrid training mode of parameter server and AllReduce. Some weights can be stored and updated through the parameter server, and other weights are still trained through the AllReduce algorithm.

The ps-lite architecture consists of three independent components: server, worker, and scheduler. Their functions are as follows:

- Server: saves model weights and backward computation gradients, and updates the model using gradients pushed by workers.

- Worker: performs forward and backward computation on the network. The gradient value for backward computation is uploaded to a server through the `Push` API, and the model updated by the server is downloaded to the worker through the `Pull` API.

- Scheduler: establishes the communication relationship between the server and worker.

## Preparations

The following describes how to use parameter server to train LeNet on Ascend 910:

### Training Script Preparation

Learn how to train a LeNet using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) by referring to <https://gitee.com/mindspore/models/tree/master/official/cv/lenet>.

### Parameter Setting

1. First of all, use `mindspore.set_ps_context(enable_ps=True)` to enable Parameter Server training mode.

    - This method should be called before `mindspore.communication.init()`.
    - If you don't call this method, the [Environment Variable Setting](https://www.mindspore.cn/tutorials/experts/en/r1.8/parallel/parameter_server_training.html#environment-variable-setting) below will not take effect.
    - Use `mindspore.reset_ps_context()` to disable Parameter Server training mode.

2. Secondly, call `mindspore.communication.init()` to initialize distributed training, including network building for `Server`, `Worker` and `Scheduler` nodes and initializing collective communication(HCCL, NCCL).

    - Launching Parameter Server training through `mpirun` is no longer supported in and after MindSpore-1.8.0 version. MindSpore uses built-in communication module to build the cluster and initialize collective communication, so `data parallel`/`auto parallel` modes are still available on `Worker` process side. Please refer to the link [Training without Relying on OpenMPI](https://www.mindspore.cn/tutorials/experts/en/r1.8/parallel/train_gpu.html#training-without-relying-on-openmpi) for details.

3. In this training mode, you can use either of the following methods to control whether the training parameters are updated by the Parameter Server and whether the training parameters are initialized on Worker or Server:

    - Use `mindspore.nn.Cell.set_param_ps()` to set all weight recursions of `nn.Cell`.
    - Use `mindspore.Parameter.set_param_ps()` to set the weight.
    - The size of the weight which is updated by Parameter Server should not exceed INT_MAX(2^31 - 1) bytes.
    - The interface `set_param_ps` can receive a `bool` parameter:`init_in_server`, indicating whether this training parameter is initialized on the Server side. `init_in_server` defaults to `False`, indicating that this training parameter is initialized on Worker. Currently, only the training parameter `embedding_table` of the `EmbeddingLookup` operator is supported to be initialized on Server side to solve the problem of insufficient memory caused by the initialization of a large shape `embedding_table` on Worker. The `EmbeddingLookup` operator's `target` attribute needs to be set to 'CPU'. The training parameter initialized on the Server side will no longer be synchronized to Worker. If it involves multi-Server training and saves CheckPoint, each Server will save a CheckPoint after the training.

4. On the basis of the [original training script](https://gitee.com/mindspore/models/blob/master/official/cv/lenet/train.py), set all LeNet model weights to be trained on the Parameter Server:

    ```python
    set_ps_context(enable_ps=True)
    init()
    network = LeNet5(cfg.num_classes)
    network.set_param_ps()
    ```

5. [optional configuration] For a large shape `embedding_table`, because the device can not store a full amount of `embedding_table`. You can configure the `vocab_cache_size` of [EmbeddingLookup operator](https://www.mindspore.cn/docs/en/r1.8/api_python/nn/mindspore.nn.EmbeddingLookup.html) to enable the cache function of `EmbeddingLookup` in the Parameter Server training mode. The `vocab_cache_size` of `embedding_table` is trained on device, and a full amount of `embedding_table` is stored in the Server. The `embedding_table` of the next batch is swapped to the cache in advance, and the expired `embedding_table` is put back to the Server when the cache cannot be placed, to achieve the purpose of improving the training performance. Each Server could save a checkpoint containing the trained `embedding_table` after the training. The Embedding cache supports the sparse mode. User need to set the `sparse` parameter of all the `EmbeddingLookup` operators that enable cache to True. The sparse mode will deduplicate the input feature id of the operator to reduce the amount of calculation and communication. Detailed network training script can be referred to <https://gitee.com/mindspore/models/tree/master/official/recommend/wide_and_deep>.

    ```python
    set_auto_parallel_context(full_batch=True,
    parallel_mode=ParallelMode.AUTO_PARALLEL)
    network = Net()
    model = Model(network)
    model.train(epoch, train_dataset, dataset_sink_mode=True)
    ```

    In the information:

    - `dataset_sink_mode`: whether to enable the sink mode of dataset or not. When `True`, it indicates enabled, and pass the data through the dataset channel. It must be set to `True` in this scenario (The inference during training also needs to enable the sink mode of dataset).
    - `full_batch`: whether to load the dataset in full or not. When `True`, it indicates fully load, and data of each device is the same. It must be set to `True` in the multi-workers scenario.
    - `parallel_mode`:parallel mode, auto parallel mode must be enabled in the multi-workers scenario, please set `parallel_mode`=`ParallelMode.AUTO_PARALLEL`.

> In `Parameter Server` mode, control flow is not supported. So we need to change `model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()}, amp_level="O2")` to `model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})` in `train.py`. This will unset `amp_level` and eliminate the impact of control flow.

### Environment Variable Setting

MindSpore reads environment variables to control parameter server training. The environment variables include the following options (all scripts of `MS_SCHED_HOST` and `MS_SCHED_PORT` must be consistent):

```text
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
    export MS_SERVER_NUM=8
    export MS_WORKER_NUM=8
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX
    export MS_SCHED_PORT=XXXX
    export MS_ROLE=MS_SCHED
    python train.py --device_target=Ascend --data_path=path/to/dataset > scheduler.log 2>&1 &
    ```

    `Server.sh`:

    ```bash
    #!/bin/bash
    export MS_SERVER_NUM=8
    export MS_WORKER_NUM=8
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX
    export MS_SCHED_PORT=XXXX
    export MS_ROLE=MS_PSERVER
    for((server_id=0;server_id<${MS_SERVER_NUM};server_id++))
    do
        python train.py --device_target=Ascend --data_path=path/to/dataset > server_${server_id}.log 2>&1 &
    done
    ```

    `Worker.sh`:

    ```bash
    #!/bin/bash
    export MS_SERVER_NUM=8
    export MS_WORKER_NUM=8
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX
    export MS_SCHED_PORT=XXXX
    export MS_ROLE=MS_WORKER
    for((worker_id=0;worker_id<${MS_WORKER_NUM};worker_id++))
    do
        python train.py --device_target=Ascend --data_path=path/to/dataset > worker_${worker_id}.log 2>&1 &
    done
    ```

    Run the following commands separately:

    ```bash
    sh Scheduler.sh
    sh Server.sh
    sh Worker.sh
    ```

    Start training. MindSpore launches multiple-worker and multiple-server training through the above method and has no dependency on any third-party components.

2. Viewing result

    Run the following command to view the communication logs between the server and worker in the `scheduler.log` file:

    ```text
    The server node id:b5d8a47c-46d7-49a5-aecf-d29d7f8b6124,node ip: 10.90.53.118,node port:46737 assign rank id:0
    The worker node id:55e86d4b-d717-4930-b414-ebd80082f541 assign rank id:1
    Start the scheduler node is successful!
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
