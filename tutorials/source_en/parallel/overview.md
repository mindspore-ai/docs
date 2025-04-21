# Distributed Parallelism Overview

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_en/parallel/overview.md)

In deep learning, as the size of the dataset and the number of parameters grows, the time and hardware resources required for training increase and eventually become a bottleneck that constrains training. Distributed parallel training, which reduces the need for hardware such as memory and computational performance, is an important optimization for performing training. In addition, distributed parallelism is important for large model training and inference, which provides powerful computational capabilities and performance advantages for handling large-scale data and complex models.

To implement distributed parallel training and inference, you can refer to the following guidelines:

## Distributed Parallel Startup Approach

MindSpore currently supports four startup methods:

- **msrun**: the capsulation of dynamic cluster. It allows user to launch distributed jobs using one single command in each node. It could be used after MindSpore is installed. No dependency on external configurations or modules, Ascend/GPU/CPU support.
- **Dynamic cluster**: Launched via MindSpore internal dynamic cluster module, no dependency on external configurations or modules, Ascend/GPU/CPU support.
- **mpirun**: Launched via OpenMPI, a multi-process communication library with Ascend/GPU support.
- **rank table**: After configuring the rank_table table, Ascend is supported by scripts that start processes corresponding to the number of cards.

For details, refer to [Distributed Parallel Startup Approach](https://www.mindspore.cn/tutorials/en/br_base/parallel/startup_method.html).

## Data Parallel

Data parallel is the most commonly used parallel training approach for accelerating model training and handling large-scale datasets. In data parallel mode, the training data is divided into multiple copies and then each copy is assigned to a different compute node, such as multiple cards or multiple devices. Each node processes its own subset of data independently and uses the same model for forward and backward propagation, and ultimately performs model parameter updates after synchronizing the gradients of all nodes.

For details, refer to [Data Parallel](https://www.mindspore.cn/tutorials/en/br_base/parallel/data_parallel.html).

## Operator-level Parallel

With the development of deep learning, network models are becoming larger and larger, such as trillions of parametric models have emerged in the field of NLP, and the model capacity far exceeds the memory capacity of a single device, making it impossible to train on a single card or data parallel. Operator-level parallelism is achieved by slicing the tensor involved in each operator in the network model and distributing the operators to multiple devices, reducing memory consumption on a single device, thus enabling the training of large models.

MindSpore provides two levels of granularity: operator-level parallelism and higher-order operator-level parallelism. Operator-level parallelism describes the tensor dimensionality distribution through a simple slicing strategy, which meets the requirements of most scenarios. Higher-order operator parallelism supports complex slicing scenarios through open device scheduling descriptions.

For details, refer to [Operator-level Parallel](https://www.mindspore.cn/tutorials/en/br_base/parallel/operator_parallel.html).

## Optimizer Parallel

When performing data parallel training, the parameter update part of the model is computed redundantly across cards. Optimizer parallelism can effectively reduce memory consumption and improve network performance on large-scale networks (e.g., Bert, GPT) by spreading the computation of the optimizer to the cards of the data parallel dimension.

For details, refer to [Optimizer Parallel](https://www.mindspore.cn/tutorials/en/br_base/parallel/optimizer_parallel.html).

## Pipeline Parallel

In recent years, the scale of neural networks has increased exponentially. Limited by the memory on a single device, the number of devices used for training large models is also increasing. Due to the low communication bandwidth between servers, the performance of the conventional hybrid parallelism (data parallel + model parallel) is poor. Therefore, pipeline parallelism needs to be introduced. Pipeline parallel can divide a model in space based on stage. Each stage needs to execute only a part of the network, which greatly reduces memory overheads, shrinks the communication domain, and shortens the communication time. MindSpore can automatically convert a standalone model to the pipeline parallel mode based on user configurations.

For details, refer to [Pipeline Parallel](https://www.mindspore.cn/tutorials/en/br_base/parallel/pipeline_parallel.html).

## Parallel Optimization Strategies

If there is a requirement for performance, throughput, or scale, or if you don't know how to choose a parallel strategy, consider the following optimization techniques:

- **Parallel Strategy Optimization**:

    - **Strategy Selection**: Depending on the model size and data volume size, different parallel strategies can be selected by referring to [Strategy Selection](https://www.mindspore.cn/tutorials/en/br_base/parallel/strategy_select.html) to improve training efficiency and resource utilization.
    - **Sharding Techniques**: Slicing techniques are also key to efficient parallel computing. In the [Sharding Techniques](https://www.mindspore.cn/tutorials/en/br_base/parallel/split_technique.html) tutorial, you can learn how to apply various slicing techniques to improve efficiency through concrete examples.
    - **Multiply Copy Parallel**: Under the existing single-copy model, certain underlying operators cannot be computed simultaneously while communicating, leading to wasted resources. Multiple copy parallel slices the data into multiple copies according to the Batch Size dimension, which can make one copy communicate while the other copy performs the computation operation, which improves the resource utilization. For details, please refer to the [Multiply Copy Parallel](https://www.mindspore.cn/tutorials/en/br_base/parallel/multiple_copy.html) tutorial.
    - **High Dimension Tensor Parallel**: High-dimensional tensor parallelism refers to multi-dimensional slicing of activation and weight tensor in MatMul computation for model parallelism, which reduces the communication volume and improves the training efficiency by optimizing the slicing strategy. For details, please refer to the [High Dimension Tensor Parallel](https://www.mindspore.cn/tutorials/en/br_base/parallel/high_dimension_tensor_parallel.html) tutorial.
- **Memory Optimization**:

    - **Gradient Accumulation**: Gradient Accumulation updates the parameters of a neural network by computing gradients on multiple MicroBatches and summing them up, then applying this accumulated gradient at once. In this way a small number of devices can also train large Batches, effectively minimizing memory spikes. For details, please refer to the [Gradient Accumulation](https://www.mindspore.cn/tutorials/en/br_base/parallel/distributed_gradient_accumulation.html) tutorial.
    - **Recompute**: Recomputation is a time-for-space technique that saves memory space by not saving the results of certain forward operator calculations, and when calculating the reverse operator, the forward results need to be used before recomputing the forward operator. For details, please refer to the [Recompute](https://www.mindspore.cn/tutorials/en/br_base/parallel/recompute.html) tutorial.
    - **Dataset Sharding**: When a dataset is too large individually, the data can be sliced for distributed training. Slicing the dataset with model parallel is an effective way to reduce the graphics memory usage. For details, please refer to the [Dataset Sharding](https://www.mindspore.cn/tutorials/en/br_base/parallel/dataset_slice.html) tutorial.
    - **Host&Device Heterogeneous**: When the number of parameters exceeds the upper limit of Device memory, you can put some operators with large memory usage and small computation on the Host side, which can simultaneously utilize the characteristics of large memory on the Host side and fast computation on the Device side, and improve the utilization rate of the device. For details, please refer to the [Host&Device Heterogeneous](https://www.mindspore.cn/tutorials/en/br_base/parallel/host_device_training.html) tutorial.
- **Communication Optimization**:

    - **Communication Fusion**: Communication fusion can merge the communication operators of the same source and target nodes into a single communication process, avoiding the extra overhead caused by multiple communications. For details, please refer to the [Communication Fusion](https://www.mindspore.cn/tutorials/en/br_base/parallel/comm_fusion.html).

## Distributed High-Level Configuration Examples

- **Multi-dimensional Hybrid Parallel Case Based on Double Recursive Search**: Multi-dimensional hybrid parallel based on double recursive search means that the user can configure optimization methods such as recomputation, optimizer parallel, pipeline parallel. Based on the user configurations, the operator-level strategy is automatically searched by the double recursive strategy search algorithm, which generates the optimal parallel strategy. For details, please refer to the [Multi-dimensional Hybrid Parallel Case Based on Double Recursive Search](https://www.mindspore.cn/tutorials/en/br_base/parallel/multiple_mixed.html).
- **Performing Distributed Training on K8S Clusters**: MindSpore Operator is a plugin that follows Kubernetes' Operator pattern (based on the CRD-Custom Resource Definition feature) and implements distributed training on Kubernetes. MindSpore Operator defines Scheduler, PS, Worker three roles in CRD, and users can easily use MindSpore on K8S for distributed training through simple YAML file configuration. The code repository of mindSpore Operator is described in: [ms-operator](https://gitee.com/mindspore/ms-operator/). For details, please refer to the [Performing Distributed Training on K8S Clusters](https://www.mindspore.cn/tutorials/en/br_base/parallel/ms_operator.html).
