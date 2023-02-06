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

## Heterogeneous Parallelism

The heterogeneous parallel training method is to analyze the memory occupation and computational intensity of the operators on the graph, and slice the operators with huge memory consumption or suitable for CPU logic processing to the CPU subgraph, and slice the computationally intensive operators with less memory consumption to the hardware accelerator subgraph. The framework cooperates with different subgraphs for network training, so that subgraphs in different hardware and without dependencies can perform the execution process in parallel.

### Computational Process

A typical computational process for MindSpore heterogeneous parallel training is shown in the following figure:

1. Users set backend for network execution

   ```python
   import mindspore as ms
   ms.set_context(device_target="GPU")
   ```

2. Users set execution backend of specific operators

   ```python
   from mindspore import ops

   prim = ops.Add()

   prim.set_device("CPU")
   ```

3. The framework is sliced according to the computational graph operator flag.
4. The framework schedules different back-end execution subgraphs.

Current scenarios that typically use heterogeneous parallel computing are: optimizer heterogeneity, Embedding heterogeneity, and PS heterogeneity.

### Optimizer Heterogeneity

During the training of a large model in PanGu or GPT3, the optimizer state takes up a large amount of memory, which in turn limits the size of the model that can be trained. Using optimizer heterogeneity, assigning optimizers to CPUs for execution can greatly scale the trainable models:

![heterogeneous-heter-opt](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/design/images/heter-opt.png)

As shown in the figure, configuring the Adam operator to CPU execution while specifying an accelerator for FP16 computation reduces the parameter memory footprint to 1/3 of the original.

1. Configure the optimizer operators to CPU execution
2. Initialize weight parameters of FP16 and optimizer state variables of FP32
3. Convert the gradient of the input optimizer to FP16 (if the gradient is FP16, you can ignore this step)
4. The weights and gradients are converted to FP32 to participate in the optimizer operation
5. The updated FP32 weights are assigned to the FP16 weights

Sample code of the optimizer heterogeneity is as follows:

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore.common.initializer import initializer
from mindspore.nn import Optimizer
_adam_opt = ops.MultitypeFuncGraph("adam_opt")
host_assign = ops.Assign()
host_assign.set_device("CPU")
host_cast = ops.Cast()
host_cast.set_device("CPU")
device_cast = ops.Cast()

@_adam_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Number", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Bool", "Bool")
def _update_run_kernel(opt, beta1, beta2, eps, lr, weight_decay, param, m, v, gradient, decay_flags, optim_filter):
    """
    Update parameters by AdamWeightDecay op.
    """
    success = True
    if optim_filter:
        param32 = host_cast(param, ms.float32)
        gradient = device_cast(gradient, ms.float32)
        if decay_flags:
            next_param = opt(param32, m, v, lr, beta1, beta2, eps, weight_decay, gradient)
        else:
            next_param = opt(param32, m, v, lr, beta1, beta2, eps, 0.0, gradient)
        ret = host_assign(param, host_cast(ops.depend(param32, next_param), ops.dtype(param)))
        return ops.depend(success, ret)
    return success

class AdamWeightDecayOp(Optimizer):
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(AdamWeightDecayOp, self).__init__(learning_rate, params, weight_decay)
        self.beta1 = ms.Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = ms.Tensor(np.array([beta2]).astype(np.float32))
        self.eps = ms.Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self.clone_param32(prefix="adam_m", init='zeros')
        self.moments2 = self.clone_param32(prefix="adam_v", init='zeros')
        self.opt = ops.AdamWeightDecay()
        self.hyper_map = ops.HyperMap()
        self.opt.set_device("CPU")

    def construct(self, gradients):
        """AdamWeightDecayOp"""
        lr = self.get_lr()
        if self.is_group:
            if self.is_group_lr:
                optim_result = self.map_reverse(ops.partial(_adam_opt, self.opt, self.beta1, self.beta2, self.eps),
                                                lr, self.weight_decay, self.parameters, self.moments1, self.moments2,
                                                gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.map_reverse(ops.partial(_adam_opt, self.opt, self.beta1, self.beta2, self.eps, lr),
                                                self.weight_decay, self.parameters, self.moments1, self.moments2,
                                                gradients, self.decay_flags, self.optim_filter)
        else:
            optim_result = self.map_reverse(ops.partial(_adam_opt, self.opt, self.beta1, self.beta2, self.eps, lr,
                                                        self.weight_decay), self.parameters, self.moments1, self.moments2,
                                            gradients, self.decay_flags, self.optim_filter)
        return optim_result

    def clone_param32(self, prefix, init=None):
        new = []
        for old_param in self.parameters:
            param_init = init
            if init is None:
                param_init = old_param.init
            new_state = old_param.clone()
            new_state.set_dtype(ms.float32)
            new_state.set_data(initializer(param_init, shape=old_param.shape, dtype=ms.float32))
            new_state.name = prefix + '.' + new_state.name
            new.append(new_state)
        return ms.ParameterTuple(new)
```

Steps 4 and 5 can also be directly fused into the optimizer operator for further optimization. The complete optimizer heterogeneous training process can be found at: <https://gitee.com/mindspore/models/tree/master/official/nlp/Pangu_alpha>.

### Embedding Heterogeneity

In some networks where large Embedding tables need to be checked, the Embedding tables are often hundreds of gigabytes in size, which is limited by the accelerator memory size and cannot be executed by loading the entire table directly onto the accelerator. By putting the operators connected to the weight table on the CPU for execution, we avoid the problem that the accelerator cannot train the network due to memory limitation.

![heterogeneous-heter-embed](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/design/images/heter-embed.png)

1. Configure EmbeddingLookup operator to CPU execution

   ```python
   import mindspore.nn as nn
   import mindspore.ops as ops
   import mindspore as ms
   from mindspore.common.initializer import initializer

   class EmbeddingLookupNet(nn.Cell):
       def __init__(self, vocab_size, embedding_size, param_init='normal'):
           super(EmbeddingLookupNet, self).__init__()
           self.embeddinglookup = ops.EmbeddingLookup().set_device('CPU')
           self.embedding_table = ms.Parameter(initializer(param_init, [vocab_size, embedding_size]),
                                  name='embedding_table')

       def construct(self, indices):
           out = self.embeddinglookup(self.embedding_table, indices, 0)
           return out
   ```

2. Configure related sparse optimizer of EmbeddingLookup to CPU execution

   ```python
   from mindspore.nn.optim import LazyAdam
   net = EmbeddingLookupNet(1000, 100)
   params = net.trainable_params()
   optimizer = LazyAdam(params)
   optimizer.target = "CPU"
   ```

A sample code for setting up the EmbeddingLookup operator is as follows:

```python
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from mindspore.common.initializer import initializer

class EmbeddingLookup(nn.Cell):
    def __init__(self, vocab_size, embedding_size, param_init='normal',
                 target='CPU', sparse=True):
        """Initialize EmbeddingLookup."""
        super(EmbeddingLookup, self).__init__()
        validator.check_value_type('sparse', sparse, [bool], self.cls_name)
        self.vocab_size = validator.check_positive_int(vocab_size, 'vocab_size')
        self.target = target
        self.sparse = sparse
        if target not in ('CPU', 'DEVICE'):
            raise ValueError('Attr \'target\' of \'EmbeddingLookup\' Op passed '
                             + str(target) + ', should be one of values in \'CPU\', \'DEVICE\'.')
        if not sparse and target == 'CPU':
            raise ValueError('When target is CPU, embedding_lookup must be sparse.')
        if sparse:
            self.gatherv2 = ops.SparseGatherV2()
        else:
            self.gatherv2 = ops.Gather()
        self.embeddinglookup = ops.EmbeddingLookup().set_device('CPU')
        self.embedding_size = validator.check_positive_int(embedding_size, 'embedding_size')
        self.embedding_table = ms.Parameter(initializer(param_init, [self.vocab_size, self.embedding_size]),
                                            name='embedding_table')

    def construct(self, indices):
        if self.target == "CPU":
            out = self.embeddinglookup(self.embedding_table, indices, 0)
        else:
            out = self.gatherv2(self.embedding_table, indices, 0)
        return out
```

EmbeddingLookup, FTRL, LazyAdam and other operators in the current nn directory are encapsulated the heterogeneous interface, and the user only needs to set the target attribute to CPU or DEVICE to switch the execution backend.

For the overall calling process, refer to <https://gitee.com/mindspore/models/tree/master/official/recommend/Wide_and_Deep>.

### PS Heterogeneity

When the EmbeddingTable reaches T level and the single machine memory cannot be put down, Parameter Server is used to pull and update the weights by heterogeneous Pull/Push operators.

![heterogeneous-heter-ps](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/design/images/heter-ps.png)

Parameter Server encapsulates heterogeneous processes, and users only need to configure parameters to use PS. For the detailed configuration process, refer to [Parameter Server training process](https://www.mindspore.cn/tutorials/experts/en/master/parallel/parameter_server_training.html).

In addition, the process of using PS is also available in the wide&deep network and can be found at: <https://gitee.com/mindspore/models/tree/master/official/recommend/Wide_and_Deep>.

### Constraints

Currently requires the user to specify the back-end of the operator execution and does not support automatic configuration based on the network.