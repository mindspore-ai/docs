# Strategy Selection

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/parallel/strategy_select.md)

## Overview

In distributed model training, for different model sizes and data volume sizes, different parallel strategies can be chosen to improve training efficiency and resource utilization. The following are the explanation and application of different parallel strategies:

1. [Data Parallel](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/parallel/data_parallel.html): Data parallel is the process of distributing different training samples to different devices (e.g., Ascend or GPUs) during the training process, with each device computing the gradient of its assigned sample. The parameters of the model are then updated by averaging or accumulating the gradients. Data parallel is suitable for situations where the amount of data is large and the number of model parameters is small enough to be loaded on a single device. Data parallel can speed up the training process by fully utilizing the computing power of multiple devices.

2. [Operator-level Parallel](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/parallel/operator_parallel.html): Operator-level parallel means that the input tensor and model parameters are sliced into multiple devices for computation on an operator basis, with each device being responsible for computing a part of the model to improve overall speed. Operator-level parallel is subdivided into semi-automatic parallel mode, which requires manual configuration of the sharding strategy, and automatic parallel mode that requires little or even no configuration of the sharding strategy. Operator-level parallel is suitable for cases where the model architecture is large and cannot be fully loaded into a single device memory.

3. [Optimizer Parallel](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/parallel/optimizer_parallel.html): Optimizer parallel can effectively reduce memory consumption and improve network performance on large-scale networks (e.g., LLAMA, DeepSeek) by spreading the optimizer computation over cards with data parallel dimensions, and is recommended to be turned on for parallel training.

4. [Pipeline Parallel](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/parallel/pipeline_parallel.html): Pipeline parallel divides the entire training process into multiple phases, with computations in each phase performed on a different device. Data flows between stages, similar to an assembly line. This strategy is applicable when the network model is large and cannot be loaded by a single card, and when the network can be more evenly divided into multiple phases of computation with longer computation times for each phase, thus maximizing overlapping computation and communication.

The selection of an appropriate parallel strategy depends on the specific training task and resource allocation. Typically, the selection can be based on the following guidelines:

- **Data parallelism** is recommended for cases where the dataset is very large and the model can be loaded to a single device.
- **Operator-level parallelism** is recommended that the model is large and cannot be loaded into a single device memory and the user has some knowledge of the core operator computational load in the network.
- **Optimizer Parallel** is recommended that the model is large and you want to reduce memory consumption to load a larger model.
- **Pipeline Parallel** is recommended that the model is large, but the model can be divided into multiple phases in a more balanced way, with each phase taking longer to compute.

In practice, you can combine multiple parallel strategies to achieve optimal training results and resource utilization.
