# Parameter Server模式

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/parameter_server_training.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

Parameter Server(参数服务器)是分布式训练中一种广泛使用的架构，相较于同步的AllReduce训练方法，Parameter Server具有更好的灵活性、可扩展性以及节点容灾的能力。具体来讲，参数服务器既支持同步SGD(Stochastic Gradient Descent，随机梯度下降)，也支持异步SGD的训练算法；在扩展性上，将模型的计算与模型的更新分别部署在Worker和Server两类进程中，使得Worker和Server的资源可以独立地横向扩缩(新增或者删除Worker和Server资源)；另外，在大规模数据中心的环境下，计算设备、网络以及存储经常会出现各种故障而导致部分节点异常，而在参数服务器的架构下，能够较为容易地处理此类故障而不会对训练中的任务产生影响。

## 基本原理

MindSpore的参数服务器采用了自研的通信框架作为基础架构，基于该框架提供的远程通信能力以及抽象的Send/Broadcast等原语，实现了同步SGD的分布式训练算法，另外结合Ascend和GPU中的高性能集合通信库(HCCL和NCCL)，MindSpore还提供了Parameter Server和AllReduce的混合训练模式，支持将部分权重通过参数服务器进行存储和更新，其余权重仍然通过AllReduce算法进行训练。

在参数服务器的架构设计中，一共包含三个独立的组件，分别是Server、Worker和Scheduler，作用分别是：

- Server：保存模型的权重和反向计算的梯度值，并使用优化器通过Worker上传的梯度值对模型进行更新。

- Worker：执行网络的正反向计算，反向计算的梯度值通过Push接口上传至Server中，通过Pull接口把Server更新好的模型下载到Worker本地。

- Scheduler：用于建立Server和Worker的通信关系。

> 参数服务器训练不支持`PyNative`模式。

## 操作实践

以LeNet在Ascend 910上使用Parameter Server训练为例：

### 训练脚本准备

参考<https://gitee.com/mindspore/models/tree/master/research/cv/lenet>，使用[MNIST数据集](http://yann.lecun.com/exdb/mnist/)，了解如何训练一个LeNet网络。

### 参数设置

1. 首先调用`mindspore.set_ps_context(enable_ps=True)`开启Parameter Server训练模式。

    - 此接口需在`mindspore.communication.init()`之前调用。
    - 若没有调用此接口，下面的[环境变量设置](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/parameter_server_training.html#环境变量设置)则不会生效。
    - 调用`mindspore.reset_ps_context()`可以关闭Parameter Server训练模式。

2. 然后调用`mindspore.communication.init()`，这一步骤初始化分布式训练，包括`Server`、`Worker`和`Scheduler`三种节点的组网，集合通信初始化(HCCL, NCCL)。

    - MindSpore 1.8.0版本及以后，不再支持使用`mpirun`启动Parameter Server训练，MindSpore使用内置通信模块进行集群搭建以及集合通信初始化，因此`Worker`进程侧的数据并行/自动并行等特性依旧能够使用，详见[不依赖OpenMPI进行训练](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_gpu.html#%E4%B8%8D%E4%BE%9D%E8%B5%96openmpi%E8%BF%9B%E8%A1%8C%E8%AE%AD%E7%BB%83)。

3. 在本训练模式下，有以下两种调用接口方式以控制训练参数是否通过Parameter Server进行更新，并且可以控制参数初始化位置：

    - 通过`mindspore.nn.Cell.set_param_ps()`对`nn.Cell`中所有权重递归设置。
    - 通过`mindspore.Parameter.set_param_ps()`对`mindspore.Parameter`权重进行设置。
    - 被设置为通过Parameter Server更新的单个权重大小不得超过INT_MAX(2^31 - 1)字节。
    - 接口`set_param_ps`可接收一个`bool`型参数：`init_in_server`，表示该训练参数是否在Server端初始化，`init_in_server`默认值为`False`，表示在Worker上初始化该训练参数；当前仅支持`EmbeddingLookup`算子的训练参数`embedding_table`在Server端初始化，以解决超大shape的`embedding_table`在Worker上初始化导致内存不足的问题，该算子的`target`属性需要设置为'CPU'。在Server端初始化的训练参数将不再同步到Worker上，如果涉及到多Server训练并保存CheckPoint，则训练结束后每个Server均会保存一个CheckPoint。

4. 在[LeNet原训练脚本](https://gitee.com/mindspore/models/blob/master/research/cv/lenet/train.py)基础上，设置该模型所有权重由Parameter Server训练：

    ```python
    set_ps_context(enable_ps=True)
    init()
    network = LeNet5(cfg.num_classes)
    network.set_param_ps()
    ```

5. [可选配置]针对超大shape的`embedding_table`，由于设备上存放不下全量的`embedding_table`，可以配置[EmbeddingLookup算子](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.EmbeddingLookup.html)的`vocab_cache_size`，用于开启Parameter Server训练模式下`EmbeddingLookup`的cache功能，该功能使用`vocab_cache_size`大小的`embedding_table`在设备上训练，全量`embedding_table`存储在Server，将下批次训练用到的`embedding_table`提前换入到cache上，当cache放不下时则将过期的`embedding_table`放回到Server，以达到提升训练性能的目的；训练结束后，可在Server上导出CheckPoint，保存训练后的全量`embedding_table`。Embedding cache支持sparse模式，需要将所有开启cache的`EmbeddingLookup`算子的`sparse`参数都设为True，sparse模式会对该算子输入的特征id做去重处理，以降低计算与通信量。详细网络训练脚本参考<https://gitee.com/mindspore/models/tree/master/official/recommend/Wide_and_Deep>。

    ```python
    set_auto_parallel_context(full_batch=True, parallel_mode=ParallelMode.AUTO_PARALLEL)
    network = Net()
    model = Model(network)
    model.train(epoch, train_dataset, dataset_sink_mode=True)
    ```

    其中，

    - `dataset_sink_mode`：是否开启数据下沉模式 ，为`True`时表示开启，通过数据集通道传递数据，该场景中必须设置为`True`（训练中推理也需要开启数据下沉模式）。
    - `full_batch`：是否全量导入数据集，为`True`时表示全量导入，每卡的数据相同，在多Worker场景中必须设置为`True`。
    - `parallel_mode`：并行模式，多Worker场景需要开启自动并行模式，设置`parallel_mode`=`ParallelMode.AUTO_PARALLEL`。

> `Parameter Server`模式暂时不支持控制流，因此在`train.py`中，需要将`model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()}, amp_level="O2")`修改为`model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})`，将混合精度`amp_level`选项关闭，消除控制流的影响。

### 环境变量设置

MindSpore通过读取环境变量，控制Parameter Server训练，环境变量包括以下选项(所有脚本中的`MS_SCHED_HOST`及`MS_SCHED_PORT`值需保持一致)：

```text
export MS_SERVER_NUM=1                # Server number
export MS_WORKER_NUM=1                # Worker number
export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
export MS_SCHED_PORT=XXXX             # Scheduler port
export MS_ROLE=MS_SCHED               # The role of this process: MS_SCHED represents the scheduler, MS_WORKER represents the worker, MS_PSERVER represents the Server
```

### 执行训练

1. shell脚本

    提供Worker，Server和Scheduler三个角色对应的shell脚本，以启动训练：

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

    最后分别执行：

    ```bash
    sh Scheduler.sh
    sh Server.sh
    sh Worker.sh
    ```

    启动训练。MindSpore使用以上方式启动多Worker和多Server训练，对第三方组件无依赖。

2. 查看结果

    查看`scheduler.log`中Server与Worker通信日志：

    ```text
    The server node id:b5d8a47c-46d7-49a5-aecf-d29d7f8b6124,node ip: 10.90.53.118,node port:46737 assign rank id:0
    The worker node id:55e86d4b-d717-4930-b414-ebd80082f541 assign rank id:1
    Start the scheduler node is successful!
    ```

    说明Server、Worker与Scheduler通信建立成功。

    查看`worker.log`中训练结果：

    ```text
    epoch: 1 step: 1, loss is 2.302287
    epoch: 1 step: 2, loss is 2.304071
    epoch: 1 step: 3, loss is 2.308778
    epoch: 1 step: 4, loss is 2.301943
    ...
    ```
