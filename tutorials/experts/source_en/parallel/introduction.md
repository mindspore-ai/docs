# Distributed Parallel Training Mode

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/tutorials/experts/source_en/parallel/introduction.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

In deep learning, as the size of the dataset and the number of parameters grows, the time and hardware resources required for training increase, which eventually become a bottleneck that constrains training. Distributed parallel training, which can reduce the need for hardware such as memory and computational performance, is an important optimization tool for conducting training. According to the different principles and modes of parallelism, the following types of parallelism are mainstream in the industry:

- Data Parallel: The parallel mode of slicing the data is generally sliced according to the batch dimension, and the data is assigned to each computational unit (worker) for model computation.
- Model Parallel: Parallel mode for slicing models. Model parallel can be classified as: operator-level model parallel, pipeline model parallel, optimizer model parallel, etc.
- Hybrid Parallel: Refers to a parallel model that covers both data parallel and model parallel.

Currently MindSpore also provides distributed parallel training, which supports a variety of models including:

- `DATA_PARALLEL`: data parallel mode.
- `AUTO_PARALLEL`: automatic parallel mode, a distributed parallel mode that incorporates data parallel and operator-level model parallel, which can automatically build cost models, find the parallel strategy with shorter training times, and select the appropriate parallel mode for the user. MindSpore currently supports automatic search for the operator-level parallel strategy, providing three different strategy search algorithms as follows:

    - `dynamic_programming`: Dynamic programming strategy search algorithm. Capable of searching the optimal strategy inscribed by the cost model, but time consuming in searching parallel strategy for huge network models. Its cost model models training time based on the memory-based computational overhead and communication overhead of the Ascend 910 chip.
    - `recursive_programming`: Double recursive strategy search algorithm. The optimal strategy can be generated instantaneously for huge networks and large-scale multi-card slicing. Its cost model based on symbolic operations can be freely adapted to different accelerator clusters.
    - `sharding_propagation`: Sharding strategy propagation algorithm. The parallel strategy are propagated from operators configured with parallel strategy to operators that are not configured. When propagating, the algorithm tries to select the strategy that triggers the least amount of tensor rescheduling communication. For the parallel strategy configuration and tensor rescheduling of the operator, refer to this [design document](https://www.mindspore.cn/docs/en/r2.0.0-alpha/design/distributed_training_design.html#principle-of-automatic-parallelism).

- `SEMI_AUTO_PARALLEL`: Semi-automatic parallel mode, compared to automatic parallel, requires the user to manually configure the shard strategy for the operator to achieve parallelism.
- `HYBRID_PARALLEL`: In MindSpore, it refers specifically to scenarios where the user achieves hybrid parallel by manually slicing the model.

## Distributed Parallel Training Mode

MindSpore currently supports the following four parallelism modes:

- Data Parallel: Used when the size of the user's network parameters can be calculated on a single card. This mode will replicate the same network parameters on each card, with different training data input during training, and suitable for most users.
- Semi-automatic Parallel: Used when neural network of the user cannot be computed on a single card and there is a large demand for the performance of slicing. Users can set this operation mode to manually specify the shard strategy for each operator to achieve better training performance.
- Automatic Parallel: Used when neural network of the user cannot be computed on a single card, but does not know how to configure the operator strategy. When users start this mode, MindSpore will automatically configure the strategy for each operator, suitable for users who want to perform parallel train but don't know how to configure the strategy.
- Hybrid Parallel: It is entirely up to the user to design the logic and implementation of parallel training, and the user can define the communication operators such as `AllGather` in the network by himself. Suitable for users who are familiar with parallel training.

The usage and considerations of these four modes will be described in detail in the following document.

Currently MindSpore provides distributed parallel training. It supports multiple modes as mentioned above, and the corresponding parallel mode can be set via the `set_auto_parallel_context()` interface.

When users invoke the distributed training process, they need to call the following code to initialize the communication and configure the corresponding rank_table_file, which can be found the **Multi-host Training** section in the [Distributed Training (Ascend)](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/train_ascend.html#multi-host-training).

```python
from mindspore.communication import init, get_rank, get_group_size
import mindspore as ms
init()
device_num = get_group_size()
rank = get_rank()
print("rank_id is {}, device_num is {}".format(rank, device_num))
ms.reset_auto_parallel_context()
# The following parallel configuration users only need to configure one of these modes
# Data parallel mode
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
# Semi-automatic parallel mode
# ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
# Automatic parallel mode
# ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL)
# Hybrid parallel mode
# ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.HYBRID_PARALLEL)
```

The following involves automatic parallel interfaces, such as the interface configuration in `set_auto_parallel_context`. Distributed parallel training is supported in the following table for each scenario.

| Parallel modes | Configuration | Dynamic graph | Static graph | Supported devices |
| ---------- | ------ | ------ | ---------- | ---------- |
| Data parallel   | DATA_PARALLEL | Support   | Support   | CPU, GPU, Ascend 910 |
| Semi-automatic parallel | SEMI_AUTO_PARALLEL | Not support | Support   | GPU, Ascend 910 |
| Automatic parallel | AUTO_PARALLEL | Not support | Support   | GPU, Ascend 910 |
| Hybrid parallel   | HYBRID_PARALLEL | Not support | Support   | GPU, Ascend 910 |

### Data Parallelism

In data parallelism, the way that the user defines the network is the same as a standalone script, but call [init()](https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.communication.html#mindspore.communication.init) before the network definition to initialize the device communication state.

```python
import numpy as np
import mindspore as ms
from mindspore.train import Model
from mindspore.communication import init
from mindspore import ops, nn

class DataParallelNet(nn.Cell):
    def __init__(self):
        super(DataParallelNet, self).__init__()
        # Initialize weights
        weight_init = np.random.rand(512, 128).astype(np.float32)
        self.weight = ms.Parameter(ms.Tensor(weight_init))
        self.fc = ops.MatMul()
        self.reduce = ops.ReduceSum()

    def construct(self, x):
        x = self.fc(x, self.weight)
        x = self.reduce(x, -1)
        return x

init()
# Set parallel mode to data parallelism, and other modes  are the same
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
net = DataParallelNet()
model = Model(net)
model.train(*args, **kwargs)
```

### Semi-automatic Parallism

Compared to automatic parallelism, the semi-automatic parallel mode requires the user to manually configure the shard **strategy** for the operator to achieve parallelism. For the operator parallel strategy definition, refer to this [design document](https://www.mindspore.cn/docs/en/r2.0.0-alpha/design/distributed_training_design.html#principle-of-automatic-parallelism).

- When starting semi-automatic and automatic modes for training, training **must** be performed via the `model.train(*args, **kwargs)` interface. Custom loops for network training are not supported.

    ```python
    # Training method 1: Call through the Model interface, and only this method is supported
    model = Model(net, *args, **kwargs)
    model.train(*args, **kwargs)

    # Training method 2: custom loop, and this method is not supported
    for d in dataset.create_dict_iterator():
        for i in range(300):
            train_net(d["data"], d["label"])
            print(net_with_criterion(d["data"], d["label"]))
    ```

- Semi-automatic parallel mode requires the user to **manually configure** the **shard** interface of each operator to tune the parallel strategy compared to automatic parallel mode.

    Taking `SemiAutoParallelNet` as an example, the script code in semi-automatic parallel mode is as follows. The shard strategy of `MatMul` is `((1, 1),(1, 2))`, and `self.weight` is specified to be sliced in two copies in the second dimension.

    ```python
    import numpy as np
    import mindspore as ms
    from mindspore.train import Model
    from mindspore.communication import init
    from mindspore import ops, nn

    class SemiAutoParallelNet(nn.Cell):
        def __init__(self):
            super(SemiAutoParallelNet, self).__init__()
            # Initialize weights
            weight_init = np.random.rand(128, 128).astype(np.float32)
            self.weight = ms.Parameter(ms.Tensor(weight_init))
            self.weight2 = ms.Parameter(ms.Tensor(weight_init))
            # Set the shard strategy. There are two inputs of fc in construct, and the first input is x and the second input is the weight self.weight
            # Therefore, the shard needs to provide a tuple, which corresponds to the number of copies of each input tensor in the corresponding dimension
            # (1,1) means that each dimension of the input x is unsliced
            # (1,2) means that the second dimension of self.weight is sliced into two parts
            # The slicing process is during the graph compilation, and the shape of self.weight is changed after the compilation is completed
            self.fc = ops.MatMul().shard(((1, 1),(1, 2)))
            self.reduce = ops.ReduceSum()

        def construct(self, x):
            x = self.fc(x, self.weight)
            # When to initialize and parallelly call operation operator in the construct function, it is equivalent to the user not setting the strategy for the matmul operator. Then the default strategy will automatically configure data parallelism, i.e. ((8, 1), (1, 1)). 8 indicates the number of cards the user runs this time
            x = ops.MatMul()(x, self.weight2)
            x = self.reduce(x, -1)
            return x

    init()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
    net = SemiAutoParallelNet()
    model = Model(net)
    model.train(*args, **kwargs)
    ```

In case the device matrices of the preceding and following operators do not match, a [rescheduling](https://www.mindspore.cn/docs/en/r2.0.0-alpha/design/distributed_training_design.html#principle-of-automatic-parallelism) is automatically inserted to ensure that the shard state of `tensor` matches the next operator input requirement. For example, the following example code is used in the training of single machine eight-card：

```python
import numpy as np
import mindspore as ms
from mindspore import ops, nn
class SemiAutoParallelNet(nn.Cell):
    def __init__(self):
        super(SemiAutoParallelNet, self).__init__()
        # Initialize weights
        weight_init = np.random.rand(128, 128).astype(np.float32)
        self.weight = ms.Parameter(ms.Tensor(weight_init))
        self.weight2 = ms.Parameter(ms.Tensor(weight_init))
        # Setting a shard strategy
        self.fc = ops.MatMul().shard(((1, 1),(1, 2)))
        self.fc2 = ops.MatMul().shard(((8, 1),(1, 1)))
        self.reduce = ops.ReduceSum()

    def construct(self, x):
        # In __init__, we configure the second input of fc as (1,2)
        # So the output tensor after fc is cut into two in the second dimension of the output, from 128 to 64, so its output shape is [batch, 64]
        x = self.fc(x, self.weight)
        # In __init__, we configure (8,1) for the 0th input of fc2 by way of shard, which means that the 0th dimension of this input is requested to be cut into 8 parts
        # The output of the last operator fc is still [batch,64], and the 0th dimension is not cut, so there is a problem of inconsistent tensor shape
        # So the auto-parallel framework will insert the StrideSlice operator here, which is not declared in the user script, to slice x
        # to ensure the consistency of the tensor shape before and after.
        # In addition, the 1st dimension of the output of fc is cut into 2 parts, but the 1st dimension of the 0th input of fc2 is taken as a whole part, so the allgather operator will be inserted.
        x = self.fc2(x, self.weight2)
        # The framework will automatically insert an AllGather operator and a StridedSlice operation here
        x = self.reduce(x, -1)
        return x
```

Therefore, the inserted rescheduling operators may be `AllGather`, `Split`, `Concat` and `StridedSlice` operators if the operators before and after have different requirements for input slicing, which will increase the computation and communication time consuming of the network. The user can [save ir graph](https://www.mindspore.cn/docs/en/r2.0.0-alpha/design/mindir.html) to view the operator status of the whole network. The automatic parallel process produces `ir` graphs named `step_parallel_begin_xxxx.ir` and `step_parallel_end_xxxx.ir`. The former indicates the graph state before entering the parallel process, and the latter indicates the graph state after the automatic parallel process. Users can view this latter one to find the operators inserted in automatic parallelism.

> - In semi-automatic parallel mode, the operators that are not configured with strategy is executed in data parallel by default, corresponding to the data parallelism of all cards.
> - Automatic parallel mode supports automatic acquisition of efficient operator parallel strategy through strategy search algorithms, and also supports users manually configure specific parallel strategy for operators.
> - If a `parameter` is used by more than one operator, the shard strategy of each operator for this `parameter` needs to be consistent, otherwise an error will be reported.

Pipeline parallel is also possible in automatic and semi-automatic modes by configuring the `pipeline_stage` property on the `Cell`. The corresponding tutorial on pipeline parallelism can be found in [Applying Pipeline Parallel](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/pipeline_parallel.html).

### Fully Automatic Parallelism

Automatic parallel mode, a distributed parallel mode that combines data parallel, model parallel and hybrid parallel in, can automatically build cost models, find parallel strategies with shorter training time, and select the appropriate parallel mode for users. MindSpore provides the following three different strategy search algorithms:

- `dynamic_programming`: Dynamic programming strategy search algorithm. It is able to search for the optimal strategy inscribed by the cost model, but it is time-consuming to search for parallel policies for huge network models. Its cost model is modeled around the memory-based computational overhead and communication overhead of the Ascend 910 chip for training time.
- `recursive_programming`: Double recursive strategy search algorithm. The optimal strategy can be generated instantaneously for huge networks and large-scale multi-card slicing. Its cost model based on symbolic operations can be freely adapted to different accelerator clusters.
- `sharding_propagation`: Sharding strategy propagation algorithm. The parallel strategy are propagated from operators configured with parallel strategy to operators that are not configured. When propagating, the algorithm tries to select the strategy that triggers the least amount of tensor rescheduling communication. For the parallel strategy configuration and tensor rescheduling of the operator, refer to [semi-automatic parallel](#semi-automatic-parallism).

The user can set the above-mentioned strategy search algorithm with the following code:

```python
import mindspore as ms
# Set dynamic programming algorithm for strategy search
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, search_mode="dynamic_programming")
# Set a double recursive method for strategy search
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, search_mode="recursive_programming")
# Set the shard strategy propagation algorithm
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, search_mode="sharding_propagation")
```

> - In `sharding_propagation` mode, the algorithm is propagated throughout the model according to the `shard` strategy set by the user. In `dynamic_programming` mode, the `shard` strategy set by the user will also take effect and will not be overwritten by the searched strategy.
> - In fully-automatic parallel mode, if you need to manually configure the data parallel strategy for all the operators in a Cell, you can use Cell.set_data_parallel() to set it uniformly.

### Hybrid Parallelism

In MindSpore, specifically refer to the scenario where the user implements hybrid parallelism by manually slicing the model. The user can define the communication operator primitives `AllReduce` and `AllGather`, etc. in the network structure and execute the parallel process manually. In this case, the user needs to implement the parameter slicing, communication after the operator slicing. For example, the code example is as follows:

```python
import numpy as np
import mindspore as ms
from mindspore.train import Model
from mindspore.communication import init
from mindspore import ops, nn

class HybridParallelNet(nn.Cell):
    def __init__(self):
        super(HybridParallelNet, self).__init__()
        # The following 2-card running scenario is used as an example to implement distributed matrix multiplication to simulate the results of single-card matrix multiplication.
        # The original logic
        #         Shapes of the inputs x and weight are (32, 512), (512, 128) respectively
        #        after calculating matmul(x, weight)
        #        The output is a tensor with shape (32, 128)
        # Here we implement the above matrix multiplication logic manually
        # We need to manually specify the shape of the current weight of the slice, which we want to slice in the relevant dimension of matmul. In the case of correlated dimensional slicing,
        # an AllReduce operation needs to be performed on the matmul results to ensure that the values are consistent with those of the standalone machine
        #
        # distributed logic
        #         The shapes of inputs x and weight are (32, 256), (256, 128) respectively
        #         after calculating output = matmul(x, weight)
        #                  output = allreduce(output)
        #         The output is a tensor with shape (32, 128)
        weight_init = np.random.rand(256, 128).astype(np.float32)
        self.weight = ms.Parameter(ms.Tensor(weight_init))
        self.fc = ops.MatMul()
        self.reduce = ops.AllReduce()

    def construct(self, x):
        x = self.fc(x, self.weight)
        x = self.reduce(x)
        return x

init()
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.HYBRID_PARALLEL)
net = HybridParallelNet()
model = Model(net)
model.train(*args, **kwargs)
```

## Multi-card Startup Method

Currently GPU, Ascend and CPU support multiple startup methods respectively. The three main methods are OpenMPI, dynamic networking and multi-process startup.

- Multi-process startup method. The user needs to start the processes corresponding to the number of cards, as well as configure the rank_table table. You can visit [Running Script](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/train_ascend.html#running-the-script) to learn how to start multi-card tasks by multi-processing.
- OpenMPI. The user can start running the script with the mpirun command, at which point the user needs to provide the host file. Users can visit [Run Scripts via OpenMPI](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/train_ascend.html#running-the-script-through-openmpi) to learn how to use OpenMPI to start multi-card tasks.
- Dynamic Networking. MindSpore uses built-in dynamic networking module and has no need to rely on external configuration files or modules to help implement multi-card tasks. Users can visit [Training without relying on OpenMPI](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/train_gpu.html#training-without-relying-on-openmpi) to learn how to use dynamic networking to start multi-card tasks.

|              | GPU  |  Ascend| CPU |
| ------------ | ---- | -----  | ------------ |
|  OpenMPI     | Support  |  Support  | Not support |
|  Multi-process startup    | Not support |  Support  | Not support |
|  Dynamic Networking    |  Support |  Support  | Support |

## Data Import Method

In parallel training, three types of data import are supported.

- Full import. Only works in **semi-automatic** and **fully automatic** parallel mode. User can turn it on with `set_auto_parallel_context(full_batch=True)`. After turning on full import, the read `batch` is considered as a complete shape of the network input in the automatic parallel process. For example, in the 8-card training, suppose the shape returned by each card `dataset` is `[32, 8]`, the data trained in the current iteration of training is `[32, 8]`. Therefore, **the user needs to ensure that the data input is consistent for each card in each iteration**. For example, make sure that the `shuffle` order is consistent for each card dataset.

- Data parallel import. If the user does not set `full_batch`, the data read in each card is a slice of the current training iteration, so the content of the data read in each card is required to be **different**. For example, in the 8-card training, the `shape` of the read data per card is `[32,8]`, so the total amount of data for the current iteration of training is `[32*8, 8]`.

- Model parallel import. The model parallel import is used in the scenrio where the image size is too large to calculate in the single-card, and the image is sliced right in the input process. MindSpore provides `dataset_strategy` interface in `set_auto_parallel_context`, and users can configure the input strategy more flexible through this interface. It should be noted that when the users use this interface, `tensor` returned by `dataset` needs to match corresponding shard strategy. The code is as follows:

    ```python
    import mindspore as ms
    # Set the input to be sliced in dimension 1, which requires the user to ensure that the input returned by the dataset is sliced in dimension 1
    ms.set_auto_parallel_context(dataset_strategy=((1, 8), (1, 8)))
    # Equivalent to setting full_batch=False
    ms.set_auto_parallel_context(dataset_strategy="data_parallel")
    # Equivalent to setting full_batch=True
    ms.set_auto_parallel_context(dataset_strategy="full_batch")
    ```

Therefore, after the user sets the above configuration, it is necessary to **manually** set the obtaining order of the dataset to ensure that the data is expected for each card.
