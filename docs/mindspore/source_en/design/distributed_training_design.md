# Distributed Training Design

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/design/distributed_training_design.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Background

With the rapid development of deep learning, the number of datasets and parameters are growing exponentially to improve the accuracy and generalization capability of neural networks. Parallel distributed training has become a development trend to resolve the performance bottleneck of ultra-large scale networks. MindSpore supports the mainstream distributed training paradigm and develops an automatic hybrid parallel solution. The following describes the design principles of several parallel training modes and provides guidance for users to perform custom development.

## Concepts

### Collective Communication

Collective communication is defined as communication that involves a group of processes. All processes in the group send and receive data after meeting certain conditions. MindSpore implements data transmission during parallel training through collective communication. On Ascend chips, MindSpore depends on the Huawei Collective Communication Library (`HCCL`) to implement the task. On GPU, MindSpore depends on the NVIDIA Collective Communication Library (`NCCL`) to implement the task.

### Synchronization Mode

In synchronous mode, all devices strart training at the same time and update parameter values synchronously after the backward propagation algorithm is executed. Currently, MindSpore uses the synchronous training mode.

## Data Parallelism

This section describes how the data parallel mode `ParallelMode.DATA_PARALLEL` works in MindSpore.

### Principle of Data Parallelism

![Data Parallel Description](./images/data_parallel.png)

1. Environment dependencies

    Each time before parallel training starts, the `mindspore.communication.init` API is called to initialize communication resources and the global communication group `WORLD_COMM_GROUP` is automatically created.

2. Data distribution

    The key of data parallelism is to split datasets based on the sample dimension and deliver the split datasets to different devices. Each dataset loading API provided by the `mindspore.dataset` module has the `num_shards` and `shard_id` parameters. The parameters are used to split a dataset into multiple datasets, perform cyclic sampling, and collect data of the `batch` size to each device. When the data volume is insufficient, the sampling restarts from the beginning.

3. Network structure

    The scripting method of data parallel network is the same as that of standalone network. This is because, although models of each device are executed independently during the forward and backward propagation processes, the same network structure is maintained. To ensure the synchronous training between devices, the initial values of corresponding network parameters must be the same. You are advised to enable `parameter_broadcast` to broadcast the values of weights in `DATA_PARALLEL` and `HYBRID_PARALLEL` modes. And in `AUTO_PARALLEL` and `SEMI_AUTO_PARALLEL` modes, the sharded dimensions of weights will be processed automatically by setting random seeds to ensure the initialization of weights are consistent on the devices which belongs to the same data parallel dimension.

4. Gradient aggregation

    Theoretically, the training effect of data parallel network should be the same as that of the standalone network. To ensure the consistency of the calculation logic, the `AllReduce` operator is inserted after gradient calculation to implement the gradient aggregation operation between devices. You can enable `mean` to average the sum of gradient values, or regard `mean` as a hyperparameter. Enabling `mean` is equivalent to reducing the learning rate by multiple times.

5. Parameter update

    Because the gradient aggregation operation is introduced, the models of each device perform parameter update with the same gradient value. Therefore, MindSpore implements a synchronous data parallel training mode. Theoretically, models trained by each device are the same. If the reduce operation on samples is involved on the network, the network output may be different. This is determined by the sharding attribute of data parallelism.

### Data Parallel Code

1. Collective communication

    - [management.py](https://gitee.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/communication/management.py): This file covers the `helper` function APIs commonly used during the collective communication process, for example, the APIs for obtaining the number of clusters and device ID. When collective communication is executed on the Ascend chip, the framework loads the `libhccl.so` library file in the environment and uses it to call the communication APIs from the Python layer to the underlying layer.
    - [comm_ops.py](https://gitee.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/ops/operations/comm_ops.py): MindSpore encapsulates supported collective communication operations as operators and stores the operators in this file. The operators include `AllReduce`, `AllGather`, `ReduceScatter`, and `Broadcast`. `PrimitiveWithInfer` defines the attributes required by the operators, as well as the `shape` and `dtype` inference methods from the input to the output during graph composition.

2. Gradient aggregation

    - [grad_reducer.py](https://gitee.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/nn/wrap/grad_reducer.py): This file implements the gradient aggregation process. After the input parameter `grads` is expanded by using `HyperMap`, the `AllReduce` operator is inserted. The global communication group is used. You can also perform custom development by referring to this section based on your network requirements. In MindSpore, standalone and distributed execution shares a set of network encapsulation APIs. In the `Cell`, `ParallelMode` is used to determine whether to perform gradient aggregation. For details about the network encapsulation APIs, see the `TrainOneStepCell` code implementation.

## Automatic Parallelism

As a key feature of MindSpore, automatic parallelism is used to implement hybrid parallel training that combines automatic data parallelism and model parallelism. It aims to help users express the parallel algorithm logic using standalone scripts, reduce the difficulty of distributed training, improve the algorithm R&D efficiency, and maintain the high performance of training. This section describes how the automatic parallel mode `ParallelMode.AUTO_PARALLEL` and semi-automatic parallel mode `ParallelMode.SEMI_AUTO_PARALLEL` work in MindSpore.

### Principle of Automatic Parallelism

![Automatic Parallel Description](./images/auto_parallel.png)

1. Distributed operator and tensor layout

    As shown in the preceding figure, the automatic parallel process traverses the standalone forward ANF graphs and performs shard modeling on tensors in the unit of distributed operator, indicating how the input and output tensors of an operator are distributed to each device of the cluster, that is, the tensor layout. Users do not need to know which device runs which slice of a model. The framework automatically schedules and allocates model slices.

    To obtain the tensor layout model, each operator has a shard strategy, which indicates the shard status of each input of the operator in the corresponding dimension. Generally, tensors can be sharded in any dimension as long as the value is a multiple of 2, and the even distribution principle is met. The following figure shows an example of the three-dimensional `BatchMatmul` operation. The parallel strategy consists of two tuples, indicating the sharding of `input` and `weight`, respectively. Elements in a tuple correspond to tensor dimensions one by one. `2^N` indicates the shard unit, and `1` indicates that the tuple is not sharded. If you want to express a parallel data shard strategy, that is, only data in the `batch` dimension of `input` is sharded, and data in other dimensions are not sharded, you can use `strategy=((2^N, 1, 1),(1, 1, 1))`. If you want to express a parallel model shard strategy, that is, only model in the non-`batch` dimension of `weight` is sharded, for example, only the `channel` dimension is sharded, you can use `strategy=((1, 1, 1),(1, 1, 2^N))`. If you want to express a hybrid parallel shard strategy, one of which is `strategy=((2^N, 1, 1),(1, 1, 2^N))`.

    ![Operator Sharding Definition](./images/operator_split.png)

    Based on the shard strategy of an operator, the framework automatically derives the distribution model of input tensors and output tensors of the operator. This distribution model consists of `device_matrix`, `tensor_shape`, and `tensor map`, which indicate the device matrix shape, tensor shape, and mapping between devices and tensor dimensions, respectively. Based on the tensor layout model, distributed operator determines whether to insert extra computation and communication operations in the graph to ensure that the operator computing logic is correct.

2. Tensor Redistribution

    When the output tensor model of an operator is inconsistent with the input tensor model of the next operator, computation and communication operations need to be introduced to implement the change between tensor layouts. The automatic parallel process introduces the tensor redistribution algorithm, which can be used to derive the communication conversion operations between random tensor layouts. The following three examples represent a parallel computing process of the formula `Z=(X×W)×V`, that is, a `MatMul` operation of two two-dimensional matrices, and show how to perform conversion between different parallel modes.

    In example 1, the output of the first data parallel matrix multiplication is sharded in the row rection, and the input of the second model parallel matrix multiplication requires full tensors. The framework automatically inserts the `AllGather` operator to implement redistribution.

    ![Tensor Redistribution](./images/tensor_redistribution1.png)

    In example 2, the output of parallel matrix multiplication of the first model is sharded in the column direction, and the input of parallel matrix multiplication of the second model is sharded in the row direction. The framework automatically inserts a communication operator equivalent to the `AlltoAll` operation in collective communication to implement redistribution.

    ![Tensor Redistribution](./images/tensor_redistribution2.png)

    In example 3, an output shard mode of the first hybrid parallel matrix multiplication is the same as an input shard mode of the second hybrid parallel matrix multiplication. Therefore, redistribution does not need to be introduced. In the second matrix multiplication operation, the related dimensions of the two inputs are sharded. Therefore, the `AllReduce` operator needs to be inserted to ensure the operation correctness.

    ![Tensor Redistribution](./images/tensor_redistribution3.png)

    In general, this distributed representation breaks the boundary between data parallelism and model parallelism, making it easy to implement hybrid parallelism. From the perspective of scripts, users only need to construct a standalone network to express the parallel algorithm logic. Framework automatically shards the entire graph.

3. Efficient parallel strategy search algorithm

    The `SEMI_AUTO_PARALLEL` semi-automatic parallel mode indicates that you manually configure the parallel strategy for operators when you are familiar with the operator sharding representation. This mode is helpful for manual optimization, with certain commissioning difficulty. You need to master the parallel principle and obtain a high-performance parallel solution based on the network structure and cluster topology. `SEMI_AUTO_PARALLEL` requires users to configure every operator with sharding strategy. To reduce the users' burden in configuring sharding strategies, the automatic parallel mode supports Sharding Propagation, which propagates sharding strategies from configured ops to non-configured ops. To completely liberate users from manual configuration, `AUTO_PARALLEL` introduces the automatic search feature of the parallel strategy, which builds cost models based on the hardware platform, and calculates the computation cost, memory cost, and communication cost of a certain amount of data and specific operators based on different parallel strategies. Using the dynamic programming algorithm or recursive programming algorithm and taking the memory capacity of a single device as a constraint condition, a parallel strategy with optimal performance is efficiently searched out.

    Strategy search replaces manual model sharding and provides a high-performance sharding solution within a short period of time, greatly reducing the threshold for parallel training.

4. Convenient distributed automatic differentiation

    In addition to forward network communication, the traditional manual model sharding needs to consider backward parallel computing. MindSpore encapsulates communication operations into operators and automatically generates backward propagation of communication operators based on the original automatic differentiation operations of the framework. Therefore, even during distributed training, users only need to pay attention to the forward propagation of the network to implement actual automatic parallel training.

### Automatic Parallel Code

1. Tensor layout model
    - [tensor_layout](https://gitee.com/mindspore/mindspore/tree/master/mindspore/ccsrc/frontend/parallel/tensor_layout): This directory contains the definitions and implementation of functions related to the tensor distribution model. `tensor_layout.h` declares the member variables `tensor_map_origin_`, `tensor_shape_`, and `device_arrangement_` required by a tensor distribution model. In `tensor_redistribution.h`, the related methods for implementing the `from_origin_` and `to_origin_` transformation between tensor distributions are declared. The deduced redistribution operation is stored in `operator_list_` and returned, in addition, the communication cost `comm_cost_`,, memory cost `memory_cost_`, and calculation cost `computation_cost_` required for redistribution are calculated.

2. Distributed operators
    - [ops_info](https://gitee.com/mindspore/mindspore/tree/master/mindspore/ccsrc/frontend/parallel/ops_info): This directory contains the implementation of distributed operators. In `operator_info.h`, the base class `OperatorInfo` of distributed operator implementation is defined. A distributed operator to be developed shall inherit the base class and explicitly implement related imaginary functions. The `InferTensorInfo`, `InferTensorMap`, and `InferDevMatrixShape` functions define the algorithms for deriving the input and output tensor distribution model of the operator. The `InferForwardCommunication` and `InferMirrorOps` functions define the extra calculation and communication operations to be inserted for operator sharding. The `CheckStrategy` and `GenerateStrategies` functions define the parallel strategy validation and generation for the operator. According to the parallel strategy `SetCostUnderStrategy`, the parallel cost `operator_cost_` of the distributed operator is generated.

3. Strategy search algorithm
    - [auto_parallel](https://gitee.com/mindspore/mindspore/tree/master/mindspore/ccsrc/frontend/parallel/auto_parallel): The shard strategy search algorithm is implemented in this directory. `graph_costmodel.h` defines the graph composition information. Each point indicates an operator `OperatorInfo`. The directed edge `edge_costmodel.h` indicates the input and output relationship of operators and the redistribution cost. `operator_costmodel.h` defines the cost model of each operator, including the calculation cost, communication cost, and memory cost. `dp_algorithm_costmodel.h` describes the main process of the dynamic planning algorithm, which consists of a series of graph operations. `costmodel.h` defines the data structures of cost and graph operations.

4. Device management
    - [device_manager.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/frontend/parallel/device_manager.h): This file is used to create and manage cluster device communication groups. The device matrix model is defined by `device_matrix.h`, and the communication domain is managed by `group_manager.h`.

5. Entire graph sharding
    - [step_auto_parallel.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/frontend/parallel/step_auto_parallel.h), and [step_parallel.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/frontend/parallel/step_parallel.h): The two files contain the core implementation of the automatic parallel process. `step_auto_parallel.h` calls the strategy search process and generates the `OperatorInfo` of the distributed operator. Then in `step_parallel.h`, processes such as operator sharding and tensor redistribution are processed to reconstruct the standalone computing graph in distributed mode.

6. Backward propagation of communication operators
    - [grad_comm_ops.py](https://gitee.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/ops/_grad/grad_comm_ops.py): This file defines the backward propagation of communication operators, such as `AllReduce` and `AllGather`.

## Advanced Features

As deep learning evolves, models get larger and larger. For example, in the field of NLP, in just a few years, the amount of parameters has developed from BERT's 100 million to GPT-3's 170 billion, and then to Pangu alpha 200 billion, and the current industry has even proposed a million billion. It can be seen that the scale of parameters has shown an exponential growth trend in recent years. On the other hand, with the development of related technologies in the fields of big data and the Internet, the datasets available for model training are also rapidly expanding, such as recommendations, natural language processing and other scenarios of the dataset that can reach terabytes.

In the face of large-scale data and large-scale parameter training, a single device either takes a long time to complete model training, or it cannot be trained due to insufficient display memory. Therefore, distributed training technology needs to be introduced.

Currently, the most commonly used distributed training technique is data parallelism. Data parallelization splits the training data into multiple devices, each maintaining the same model parameters and the same size of computing tasks, but processing different data. In the process of backpropagation, the parameter gradient generated by each device is globally AllReduce synchronously summed. When the dataset is large and the model is small, there is an advantage to choosing data parallelism, such as ResNet50. However, when the model is large, or the dataset and model are larger, other distributed features need to be used.

MindSpore provides the following advanced features to support distributed training of large models, and users can flexibly combine them according to their own needs.

### Operator Parallel

Operator-level parallelism is a distributed computation of operators by splitting their input tensors into multiple devices in units. On the one hand, data samples and model parameters can be split into multiple devices at the same time to complete the training of large models. On the other hand, you can make full use of cluster resources for parallel computing to improve the overall speed.

The users can set the sharding strategy of each operator in the forward network, and the framework models each operator and its input tensor according to the sharding strategy of the operator, so that the computational logic of the operator remains mathematically equivalent before and after the sharding.

### [Pipeline Parallel](https://www.mindspore.cn/tutorials/experts/en/master/parallel/pipeline_parallel.html)

When there are a large number of cluster devices, if only the operator level is used in parallel, communication needs to be carried out on the communication domain of the entire cluster, which may make communication inefficient and reduce overall performance.

Pipeline parallel can split the neural network structure into multiple stages, and each stage runs in a part of the device. The communication domain of the set communication limits to this part of the device, and the stage uses point-to-point communication.

The advantages of pipeline parallel are that they can improve communication efficiency and easily handle layered neural network structures. The disadvantage is that some nodes may be idle at the same time.

### Optimizer Parallel

When training in parallel with data or operators, the parameters of the model may have the same copy on multiple devices. This allows the optimizer to have redundant calculations across multiple devices when updating this weight. In this case, the optimizer's computational volume can be spread across multiple devices through optimizer parallelism. It has the advantage of reducing static memory consumption and reducing the amount of computation in the optimizer. The disadvantage is that it increases the communication overhead.

### [Host Device Training](https://www.mindspore.cn/docs/en/master/design/host_device_training.html)

When training large models, the overall size of the model that can be trained will be limited by the number of devices due to the limited memory capacity of each device (accelerator). In order to complete larger-scale model training, you can use the host and device heterogeneous training modes. It takes advantage of both the large memory on the host side and the fast calculation on the accelerator side, and is an effective way to reduce the number of devices during the training of the super-large model.

### [Recompute](https://www.mindspore.cn/docs/en/master/design/recompute.html)

MindSpore automatically derives the reverse graph according to the forward graph calculation process, and the forward graph and the inverse graph together form a complete calculation graph. When calculating some reverse operators, it may be necessary to use the calculation results of some forward operators, resulting in the calculation results of these forward operators, which need to reside in memory until these reverse operators are calculated, and the memory they occupy will not be reused by other operators. The compute results of these forward operators, which reside in memory for a long time, push up the peak memory footprint of the computation, especially in large-scale network models. In order to reduce memory peaks, the recomputing technique can not save the calculation results of the forward activation layer, so that the memory can be reused, and then when calculating the reverse part, recalculate the results of the forward activation layer.

### [Sharding Propagation](https://www.mindspore.cn/docs/en/master/design/sharding_propagation.html)

In operator-level parallelism, the user is required to configure a slicing strategy for each operator in the forward network (if not configured, the data-parallel policy is used by default). The slicing strategy propagation feature can configure only a few operators to automatically generate a feasible sharding strategy for operators without a sharding strategy, and achieve the effect of minimizing communication overhead.

### [Parameter Server Training](https://www.mindspore.cn/docs/en/master/design/parameter_server_training.html)

Parameter Server is a widely used architecture in distributed training, which has better flexibility, scalability, and node disaster tolerance than the AllReduce training method of data parallel synchronization. The parameter server supports both synchronous SGD (Stochastic Gradient Descent) and asynchronous SGD training algorithms. In terms of scalability, the calculation of the model and the update of the model are deployed in the worker and server processes respectively, so that the resources of the worker and server can be scaled horizontally independently (adding or removing the worker and server resources). In addition, in the environment of large-scale data centers, computing equipment, networks and storage often have various failures that lead to some node abnormalities, and under the architecture of parameter servers, such failures can be easily handled without affecting the tasks in training.

### Communication Operator Fusion

In the distributed training scenario, cross-device or even cross-node data transmission is a bottleneck that restricts scalability and computing power utilization. Communication operator fusion is an important method to improve the utilization of network resources and accelerate the efficiency of data transmission, which packages the communication operators of the same source node and the destination node and executes them at the same time to avoid the additional overhead caused by multiple single operator execution.

### Dataset Splitting

When doing distributed training, you need to import the training dataset to each device. There are two common ways to import: 1) Import in parallel with the data, that is, the data is split into match dimensions, and each device is imported as part; 2) Import full amount of data per device. In addition, when some dimensions of the data are particularly large (such as the H/W dimension of the remote sensing picture may be particularly large), even if the sample size is small, the picture needs to be split, that is, the data is split in the H/W dimension, and each device reads a part of the picture. This special performance supports splitting datasets into specific dimensions to meet training requirements in the field of large-format image processing.

### [Distributed Inference](https://www.mindspore.cn/tutorials/experts/en/master/parallel/distributed_inference.html)

For large models, the trained parameters are huge, such as Pangu alpha has 200 billion parameters. Such a huge model cannot be simply deployed on a single device when inference, and the model needs to be split into clusters when inference. Inference and training may not be in the same environment, and the cluster size used may be different. Therefore, it is also necessary to support scenarios where the number of inference devices is inconsistent with the number of training devices.

### Functional Operator Splitting

In dynamic graph mode, you specify that a part of the network structure executes in graph mode and performs various parallel operations.

## Description of the Interface Related to the Feature

| Feature category             | Feature interface                                            | Description                                                  | Function                                                     |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| operator parallel            | shard(in_strategy=None, out_strategy=None)<br />In Primitive class | Set the sharding strategy of the input and output tensors of the operator (where the sharding strategy of the output tensor only supports some operators, such as Gauther and MatMul.) | Reduce the memory capacity of a single device by slicing the tensor involved in each operator in the network model to complete the large model training/inference. Or use cluster resources to perform distributed computing to reduce the overall execution time. |
|                              | add_prim_attr(name, value)<br />In Primitive class           | Gather Operator:<br />add_prim_attr(“manual_split”, config): Configure a non-uniform sharding strategy for its first input, where config type is tuple, which describes how the first parameter, dimension 0, is split. For example , ( 10 , 20 , 30 , 4 ) means that the 0th dimension of the first input of the operator is tangent into 4 parts , and the shape size of each part is 10 , 20 , 30 , 4, respectively. | In the recommended field, there is a scene where each column of the dataset corresponds to a subtable. In this scenario, using this configuration can reduce traffic and improve overall performance. |
|                              |                                                              | EmbeddingLookUp Operator:<br />add_prim_attr(“primitive_target”, “CPU”): Configure it to execute on the CPU for heterogeneous scenarios. | In the recommended field, there is a particularly large scene of the Embedding Table, in order to save device memory, you can use this configuration to put EmbeddingLookUp on the CPU to execute to complete the training of the recommended large model. |
|                              | set_auto_parallel_context(enable_alltoall=bool_value)        | Indicate whether the AllToAll communication operator is allowed to be generated when communicating, and its value is the bool type, which defaults to False. | AllToAll communication can reduce the amount of communication data and improve communication efficiency, but it requires environmental support. |
| Pipeline parallel            | set_auto_parallel_context(pipeline_stages=stage_num)         | Set the number of pipes in pipeline parallelism, the value of which is a positive integer, and the value range is [1, number of devices]. | Specify the number of stages, limiting the communication domain of the collection communication to the stage, and the point-to-point communication between the stages. |
|                              | pipeline_stage(value)<br />In Cell class                     | Set which stage the Cell executes in.                        | Set which stage the Cell executes in.                        |
|                              | PipelineCell(network, micro_size)                            | Specify the number of MicroSizes for the training network, where the network is the network to be trained and the micro_size is a positive integer. | Specify micro_size can reduce the idle wait time between stages and improve the overall efficiency of pipeline parallel. |
| Optimizer parallel           | set_auto_parallel_context(enable_parallel_optimizer=bool_value) | Indicate whether optimizer parallelism is enabled. Its value is bool type, and the default is False. | Optimizer parallel saves static memory overhead, but increases communication overhead. |
|                              | set_auto_parallel_context(parallel_optimizer_config=config)  | This configuration takes effect only after optimizer parallel is turned on. The config is a dict and supports two key values: <br />gradient_accumulation_shard(bool): If True, the cumulative gradient variable will be sharded on the data parallelism, defaulting to False.<br />parallel_optimizer_threshold(int): This value represents the optimizer sharding threshold in KB (default value is 64KB). When the parameter size does not exceed this value, it will not be split. | gradient_accumulation_shard true saves a portion of the parameter size of static memory, but increases communication overhead. <br />Optimizer sharding thresholds allow smaller shape parameters to be not optimized for splitting to save communication resources. |
| Recompute                    | recompute(mode=True)<br />In primitive class                 | Used to specify whether the operator needs to be recalculated, and its value is bool type, which defaults to True and means that the operator recalculation is enabled. | After enabling operator recalculation, you can reduce the peak of dynamic memory, but increase the overall computation amount. |
|                              | recompute(**kwargs)<br />In Cell class                       | When this interface is called, the operator in this Cell is recalculated.<br />The input parameter has two bool class options:<br />mp_comm_recompute: Whether to enable model parallel communication operator recalculation, and the default is True.<br />parallel_optimizer_comm_recompute: Whether to enable optimizer parallel communication operator recompute, and the default is False. | Enable Cell recompute and configure whether the model parallel communication operator and the optimizer parallel communication operator are recomputed. When the communication operator is recomputed, it consumes communication resources but reduces the peak of dynamic memory. |
| Auto-parallel                | set_auto_parallel_context(search_mode=mode)                  | Specify the policy search algorithm, with a value of type string, and the optional value: <br />1. "sharding_propagation": indicate a policy search by using sharding strategy propagation algorithm;<br />2. "dynamic_programming": indicate the use of dynamic programming algorithms for policy search;<br />3. "recursive_programming": indicate the use of a double recursive algorithm for policy search; | Automatic parallel allows the user to search for sharding strategy without configuring or configuring a small number of operators, and the framework searches for the sharding strategy. |
|                              | set_algo_parameters(fully_use_devices=bool_value)            | Whether operators need to be split across all devices when setting up search policies. Its value is of type bool, which defaults to True. | If the operator is split into all devices, the search space can be reduced and the search speed can be improved, but the search strategy is not globally optimal. |
|                              | set_auto_parallel_context(all_reduce_fusion_config=config)   | Configure the gradient AllReduce operator fusion strategy with a value of type list. For example: [20, 35], which means that the first 20 AllReduces are fused into 1, the 20th to 35th AllReduce are fused into 1, and the remaining AllReduce are fused into 1. | Reduce the number of operations of the AllReduce communication operator and improve communication efficiency. |
| comm_fusion                  | set_auto_parallel_context(comm_fusion=config)                | Set the fusion configuration of the communication operator, and support the configuration of the AllReduce, AllGather, and ReduceScatter communication operators currently. Its value is of type dict, such as comm_fusion={"allreduce": {"mode": "auto", "config": None}}. There are three options for "mode" among them: <br/>"auto": Automatically perform operator fusion according to the data volume threshold of 64MB, and the configuration parameter "config" is None. <br />"size": Communicate operator fusion according to the method of manually setting the data volume threshold, and the configuration parameter "config" type is int, with the unit of MB. <br />"index": Only "allreduce" supports configuring index, which means that the configuration parameter "config" type is list according to the way the sequence number of the communication operator is fused. For example: [20, 35], which means that the first 20 AllReduces are fused into 1, the 20th to 35th AllReduce are fused into 1, and the remaining AllReduce are fused into 1. | Reduce the number of operations of the AllReduce/AllGather/ReduceScatter communication operator and improve communication efficiency. |
| Dataset slicing              | set_auto_parallel_context(dataset_strategy=config)           | Configure the sharding policy for the dataset. where config is Union[str, tuple]. <br />When a string is passed in, there are two options: <br /> "full_batch": indicates that the dataset is not tangential, and <br /> "data_parallel": indicates that the dataset is sliced in parallel with the data. <br />When passed in tuple, the content in tuple represents the shard() interface of the dataset, similar to the premiumive shard() interface. <br /> if this interface is not called, it defaults to the "data_parallel" mode. | When the number of samples is smaller than the number of cards, it can be imported in the way of "full_batch"; when the number of samples is large and the model parameters are small, it can be imported in the way of "data_parallel"; when the data set is high-resolution image data, it can be imported by configuring the tuple sharding strategy. |
| Distributed inference        | infer_predict_layout(*predict_data)                          | Use inference data to perform precompilation, which outputs the splitting information of the operator. | Obtain the sharding information of the ownership weight at the time of inference. |
|                              | load_distributed_checkpoint(network, checkpoint_filenames, predict_strategy=None, train_strategy_filename=None) | Load the distributed weights. Each machine needs to pre-place the full amount of ckpt. <br/> where network represents the inference network, checkpoint_filenames represents the checkpoint file, predict_strategy is the output of the infer_predict_layout(), and train_strategy_filename is the operator slicing strategy information saved during training. | Load distributed weights for distributed inference.          |
| Functional operator sharding | shard(in_strategy, out_strategy, device="Ascend", level=0)<br />In Cell class | Set the sharding strategy of the input and output tensors of the cell, and the parallel strategy of the remaining operators is propagated by the sharding strategy. in_strategy/out_strategy specify the sharding policy for the input/output tensor. device specifies the execution device, and level specifies the pattern of the sharding policy propagation algorithm. | In PyNative mode, specify that a cell instance executes in graph mode, and synchronizes the operator-level model according to the specified input-output sharding strategy, while the rest of the model is still executed in Python mode. |
|                              | ops.shard(fn, in_strategy, out_strategy, device="Ascend", level=0) | The incoming fn is a cell instance or function. The rest of the input is the same as shard, and the return value is a function. When this function is called, the operator-level model is executed in graph mode in parallel. | This usage allows you to specify that a function performs model parallelism at the operator level, with the same function as cell's shard method. |

