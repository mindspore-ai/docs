# Data Parallel

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/features/parallel/data_parallel.md)

## Overview

Data parallel is the most commonly used parallel training approach for accelerating model training and handling large-scale datasets. In data parallel mode, the training data is divided into multiple copies and then each copy is assigned to a different compute node, such as multiple cards or multiple devices. Each node processes its own subset of data independently and uses the same model for forward and backward propagation, and ultimately performs model parameter updates after synchronizing the gradients of all nodes.

> Hardware platforms supported for data parallelism include Ascend, GPU and CPU, in addition to both PyNative and Graph modes.

Related interfaces are as follows:

1. [mindspore.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL)](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_auto_parallel_context.html): Set the data parallel mode.
2. [mindspore.nn.DistributedGradReducer()](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.DistributedGradReducer.html): Perform multi-card gradient aggregation.

## Overall Process

![Overall Process](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/design/images/data_parallel.png)

1. Environmental dependencies

    Before starting parallel training, the communication resources are initialized by calling the [mindspore.communication.init](https://www.mindspore.cn/docs/en/master/api_python/communication/mindspore.communication.init.html) interface and the global communication group `WORLD_COMM_GROUP` is automatically created. The communication group enables communication operators to distribute messages between cards and machines, and the global communication group is the largest one, including all devices in current training. The current mode is set to data parallel mode by calling `mindspore.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL)`.

2. Data distribution

    The core of data parallel lies in splitting the dataset in sample dimensions and sending it down to different cards. In all dataset loading interfaces provided by the [mindspore.dataset](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.html) module, there are `num_shards` and `shard_id` parameters which are used to split the dataset into multiple copies and cycle through the samples in a way that collects `batch` data to their respective cards, and will start from the beginning when there is a shortage of data.

3. Network composition

    The data parallel network is written in a way that does not differ from the single-card network, due to the fact that during forward propagation & backward propagation the models of each card are executed independently from each other, only the same network structure is maintained. The only thing we need to pay special attention to is that in order to ensure the training synchronization between cards, the corresponding network parameter initialization values should be the same. In `DATA_PARALLEL` mode, we can use [mindspore.set_seed](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_seed.html) to set the seed or enable `parameter_broadcast` in `mindspore.set_auto_parallel_context` to achieve the same initialization of weights between multiple cards.

4. Gradient aggregation

    Data parallel should theoretically achieve the same training effect as the single-card machine. In order to ensure the consistency of the computational logic, the gradient aggregation operation between cards is realized by calling the `mindspore.nn.DistributedGradReducer()` interface, which automatically inserts the `AllReduce` operator after the gradient computation is completed. `DistributedGradReducer()` provides the `mean` switch, which allows the user to choose whether to perform an average operation on the summed gradient values, or to treat them as hyperparameters.

5. Parameter update

    Because of the introduction of the gradient aggregation operation, the models of each card will enter the parameter update step together with the same gradient values.
