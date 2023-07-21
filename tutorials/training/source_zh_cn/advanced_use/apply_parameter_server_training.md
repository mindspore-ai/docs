# 使用Parameter Server训练

`Linux` `Ascend` `GPU` `模型训练` `中级` `高级`

[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/tutorials/training/source_zh_cn/advanced_use/apply_parameter_server_training.md)

## 概述

Parameter Server(参数服务器)是分布式训练中一种广泛使用的架构，相较于同步的AllReduce训练方法，Parameter Server具有更好的灵活性、可扩展性以及节点容灾的能力。具体来讲，参数服务器既支持同步SGD，也支持异步SGD的训练算法；在扩展性上，将模型的计算与模型的更新分别部署在Worker和Server两类进程中，使得Worker和Server的资源可以独立地横向扩缩；另外，在大规模数据中心的环境下，计算设备、网络以及存储经常会出现各种故障而导致部分节点异常，而在参数服务器的架构下，能够较为容易地处理此类的故障而不会对训练中的任务产生影响。

在MindSpore的参数服务器实现中，采用了开源的[ps-lite](https://github.com/dmlc/ps-lite)作为基础架构，基于其提供的远程通信能力以及抽象的Push/Pull原语，实现了同步SGD的分布式训练算法，另外结合Ascend和GPU中的高性能集合通信库(HCCL和NCCL)，MindSpore还提供了Parameter Server和AllReduce的混合训练模式，支持将部分权重通过参数服务器进行存储和更新，其余权重仍然通过AllReduce算法进行训练。

在ps-lite的架构设计中，一共包含三个独立的组件，分别是Server、Worker和Scheduler，作用分别是：

- Server：保存模型的权重和反向计算的梯度值，并使用优化器通过Worker上传的梯度值对模型进行更新。

- Worker：执行网络的正反向计算，反向计算的梯度值通过Push接口上传至Server中，通过Pull接口把Server更新好的模型下载到Worker本地。

- Scheduler：用于建立Server和Worker的通信关系。

## 准备工作

以LeNet在Ascend 910上使用Parameter Server训练为例：

### 训练脚本准备

参考<https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/cv/lenet>，使用[MNIST数据集](http://yann.lecun.com/exdb/mnist/)，了解如何训练一个LeNet网络。

### 参数设置

1. 首先调用`mindspore.context.set_ps_context(enable_ps=True)`开启Parameter Server训练模式.

    - 此接口需在`mindspore.communication.management.init()`之前调用。
    - 若没有调用此接口，下面的[环境变量设置](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/apply_parameter_server_training.html#id5)则不会生效。
    - 调用`mindspore.context.reset_ps_context()`可以关闭Parameter Server训练模式。

2. 在本训练模式下，有以下两种调用接口方式以控制训练参数是否通过Parameter Server进行更新，并且可以控制参数初始化位置：

    - 通过`mindspore.nn.Cell.set_param_ps()`对`nn.Cell`中所有权重递归设置。
    - 通过`mindspore.Parameter.set_param_ps()`对此权重进行设置。
    - 被设置为通过Parameter Server更新的单个权重大小不得超过INT_MAX(2^31 - 1)字节。
    - 接口`set_param_ps`可接收一个`bool`型参数：`init_in_server`，表示该训练参数是否在Server端初始化，`init_in_server`默认值为`False`，表示在Worker上初始化该训练参数；当前仅支持`EmbeddingLookup`算子的训练参数`embedding_table`在Server端初始化，以解决超大shape的`embedding_table`在Worker上初始化导致内存不足的问题，该算子的`target`属性需要设置为'CPU'。在Server端初始化的训练参数将不再同步到Worker上，如果涉及到多Server训练并保存CheckPoint，则训练结束后每个Server均会保存一个CheckPoint。

3. 在[原训练脚本](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/lenet/train.py)基础上，设置LeNet模型所有权重通过Parameter Server训练：

    ```python
    context.set_ps_context(enable_ps=True)
    network = LeNet5(cfg.num_classes)
    network.set_param_ps()
    ```

4. [可选配置]针对超大shape的`embedding_table`，由于设备上存放不下全量的`embedding_table`，可以配置[EmbeddingLookup算子](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/nn/mindspore.nn.EmbeddingLookup.html)的`vocab_cache_size`，用于开启Parameter Server训练模式下`EmbeddingLookup`的cache功能，该功能使用`vocab_cache_size`大小的`embedding_table`在设备上训练，全量`embedding_table`存储在Server，将下批次训练用到的`embedding_table`提前换入到cache上，当cache放不下时则将过期的`embedding_table`放回到Server，以达到提升训练性能的目的，详细网络训练脚本参考<https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/recommend/wide_and_deep>。

    ```python
    context.set_auto_parallel_context(full_batch=True)
    network = Net()
    model = Model(network)
    model.train(epoch, train_dataset, dataset_sink_mode=True)
    ```

    其中，

    - `dataset_sink_mode`：是否开启数据下沉模式 ，为`True`时表明开启，通过数据集通道传递数据，该场景中必须设置为`True`。
    - `full_batch`：是否全量导入数据集，为`True`时表明全量导入，每卡的数据相同，在多worker场景中必须设置为`True`。

### 环境变量设置

MindSpore通过读取环境变量，控制Parameter Server训练，环境变量包括以下选项(其中`MS_SCHED_HOST`及`MS_SCHED_PORT`所有脚本需保持一致)：

```text
export PS_VERBOSE=1                   # Print ps-lite log
export MS_SERVER_NUM=1                # Server number
export MS_WORKER_NUM=1                # Worker number
export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
export MS_SCHED_PORT=XXXX             # Scheduler port
export MS_ROLE=MS_SCHED               # The role of this process: MS_SCHED represents the scheduler, MS_WORKER represents the worker, MS_PSERVER represents the Server
```

## 执行训练

1. shell脚本

    提供Worker，Server和Scheduler三个角色对应的shell脚本，以启动训练：

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

    最后分别执行：

    ```bash
    sh Scheduler.sh > scheduler.log 2>&1 &
    sh Server.sh > server.log 2>&1 &
    sh Worker.sh > worker.log 2>&1 &
    ```

    启动训练

2. 查看结果

    查看`scheduler.log`中Server与Worker通信日志：

    ```text
    Bind to role=scheduler, id=1, ip=XXX.XXX.XXX.XXX, port=XXXX
    Assign rank=8 to node role=server, ip=XXX.XXX.XXX.XXX, port=XXXX
    Assign rank=9 to node role=worker, ip=XXX.XXX.XXX.XXX, port=XXXX
    the scheduler is connected to 1 workers and 1 servers
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
