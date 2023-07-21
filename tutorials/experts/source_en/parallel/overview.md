# Distributed Parallelism Overview

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.1/tutorials/experts/source_en/parallel/overview.md)

In deep learning, as the size of the dataset and number of parameters grows, the time and hardware resources required for training will increase and eventually become a bottleneck that constrains training. Distributed parallel training, which can reduce the demand on hardware such as memory and computational performance, is an important optimization means to perform training. According to the different principles and modes of parallelism, the types of mainstream parallelism are as follows:

- Data Parallel: The parallel mode of slicing the data is generally sliced according to the batch dimension, and the data is assigned to each computational unit (worker) for model computation.
- Model Parallel: Parallel mode for slicing models. Model parallelism can be classified as: operator model parallelism, pipeline model parallelism, optimizer model parallelism.
- Hybrid Parallel: Refers to a parallel model that covers data parallelism and model parallelism.

## Distributed Parallelism Training Mode

MindSpore currently offers the following four parallel modes:

- `DATA_PARALLEL`: Data parallel mode.
- `AUTO_PARALLEL`: Automatic parallel mode, a distributed parallel mode that incorporates data parallelism and operator model parallelism, can automatically build cost models, find parallel strategies with shorter training times, and select the appropriate parallel mode for the user. MindSpore currently supports automatic search for operator parallel strategy, and provides three different strategy search algorithms as follows:

    - `dynamic_programming`: Dynamic programming strategy search algorithm. It is able to search for the optimal strategy inscribed by the cost model, but it is time consuming in searching for parallel strategies for huge network models. Its cost model is modeled based on the memory-based computational overhead and communication overhead of the Ascend 910 chip for training time.
    - `recursive_programming`: Double recursive strategy search algorithm. The optimal strategy is generated instantaneously for huge networks and large-scale multi-card slicing. Its cost model based on symbolic operations can be freely adapted to different accelerator clusters.
    - `sharding_propagation`: Sharding strategy propagation algorithm。A parallel strategy is propagated from operators configured with parallel policies to operators not configured. When propagating, the algorithm tries to select the strategy that triggers the least amount of tensor redistribution communication. For the parallel strategy configuration and tensor redistribution of the operator, refer to this [design document](https://www.mindspore.cn/docs/en/r2.1/design/distributed_training_design.html).
- `SEMI_AUTO_PARALLEL`: Semi-automatic parallelism mode, compared to automatic parallelism, requires the user to manually configure the shard strategy for the operator to achieve parallelism.
- `HYBRID_PARALLEL`: In MindSpore it specifically refers to scenarios where the user achieves hybrid parallelism by manually slicing the model.

## Reading Guide

MindSpore provides you with a series of easy-to-use parallel training components. To get a better understanding of MindSpore distributed parallel training components, we recommend that you read this tutorial in the following order.

- If your model parameter scale can be operated on a single card, you can read the Data Parallelism tutorial.
- If your model parameter scale cannot run on a single card, you can read the [operator-level parallelism](https://www.mindspore.cn/tutorials/experts/en/r2.1/parallel/operator_parallel.html) and [pipeline parallel](https://www.mindspore.cn/tutorials/experts/en/r2.1/parallel/pipeline_parallel.html) tutorials to learn how MindSpore provides you with model parallelism capabilities.
- If you want to learn how to reduce the memory occupation during model parallel, you can read the [recompute](https://www.mindspore.cn/tutorials/experts/en/r2.1/parallel/recompute.html) and [Host&Device Side Heterogeneity](https://www.mindspore.cn/tutorials/experts/en/r2.1/parallel/host_device_training.html) tutorials.
- If you want to experience MindSpore easy-to-use model parallelism interfaces, you can read the [Semi-automatic Parallelism](https://www.mindspore.cn/tutorials/experts/en/r2.1/parallel/train_ascend.html) tutorials.
- If you have an in-depth understanding of parallel training and would like to learn more about the high-level configuration and application of MindSpore distributed parallelism, please read the [Distributed Parallelism Case](https://www.mindspore.cn/tutorials/experts/en/r2.1/parallel/distributed_case.html) chapter