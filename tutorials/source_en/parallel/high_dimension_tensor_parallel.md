# High Dimension Tensor Parallel

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_en/parallel/high_dimension_tensor_parallel.md)

## Overview

Model parallelism can effectively reduce the memory load in large model training, but the communication it introduces is a significant performance bottleneck. Therefore, the whole network model slicing strategy needs to be optimized in order to introduce minimal amount of communication.

Tensor Parallel (TP) training is the process of dividing a tensor into `N` blocks along a specific dimension, with each device holding only `1/N` of the entire tensor, performing MatMul/BatchMatMul and other arithmetic computations, and introducing additional communication to ensure that the final result is correct. The high-dimensional tensor parallelism allows flexible control of the number of slices and axes of the tensor, and supports 1D, 2D, and 3D slices. 2D/3D slices are slower to grow with the number of TP devices under a suitable slicing strategy compared to 1D slices, and have lower extra communication when the number of TP devices is larger, which achieves the purpose of improving training speed.

> The hardware platform supported by this feature is Ascend, which needs to be run in Graph and semi-automatic parallelism mode.

Usage Scenario: In semi-automatic mode, when there is tensor parallelism in the network and the number of training cards is large (generally not less than 8 cards), 2D/3D tensor parallelism strategy configuration of MatMul/BatchMatMul and adapting slicing strategy of the upstream and downstream operators can be used to obtain the training performance improvement.

### Basic Principle

#### 1D Tensor Parallel Computing Communication Behavior

In 1D tensor parallelism, the full data of activation bsh is stored on each card, and slices are made on only one dimension of weights he and eh. After the first matrix product of the weights of the activation and column slicing, a second matrix product is performed with the weights of the second row slicing, and the resulting `partial sums` are computed after one AllReduce communication between all cards to compute the final correct result.

![image](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/tutorials/source_zh_cn/parallel/images/high_dimension_tensor_parallel_image_0.png)

*Figure: 1D tensor computing communication behavior (4 cards in parallel)*

#### 2D Tensor Parallel Computing Communication Behavior

The 2D tensor parallelism slices both the activation bsh and the weight he by two communication groups, x and y. The weights are sliced in both dimensions. As an example in the following figure, Rank0-Rank2 are `communication group x` and Rank0-Rank1 are `communication group y`. After activating the AllGather that passes through the first communication group y and matrix product with the weights, the obtained part and the ReduceScatter that passes between the first communication group x, the correct result of the first MatMul is computed. The second MatMul communication computes the communication behavior similar to the first one, which is not shown in the following figure.

![image](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/tutorials/source_zh_cn/parallel/images/high_dimension_tensor_parallel_image_1.png)

*Figure : 2D tensor parallel computing communication behavior (as an example of a MatMul computation under 4-card parallelism)*

#### 3D Tensor Parallel Computing Communication Behavior

3D tensor parallelism further splits the total cardinality into x, y, and z communication groups for finer-grained slicing. Relative to 2D tensor parallelism, 3D tensor parallelism shifts a portion of the AllGather communication to weight he. This operation reduces the total communication introduced when the relative weight of the shape of the activated bsh is large. As shown in the 8-card parallel case in the following figure, the overall process is: activation in communication group y for AllGather, weights in communication group z for AllGather -> matrix product, the resulting partial sum -> ReduceScatter in communication group x to get the final result. The last 4 cards communication calculation is similar to the first 4 cards, the second MatMul communication calculation communication is similar to the first MatMul, none of the following figures are shown.

![image](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/tutorials/source_zh_cn/parallel/images/high_dimension_tensor_parallel_image_2.png)

*Figure : 3D tensor parallel computing communication behavior (as an example of a MatMul computation in the first 5 cards under 8-card parallelism)*

A comprehensive comparison of the theoretical computation, storage, and communication overheads for 1D/2D/3D is as follows:

| TP Type | Compution | Memory(parameters) | Memory(activation) | Communication Volume(Single Device) |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| 1D tensor parallel computing | O(1/P) | O(1/P) | O(1) | 2(P-1)bsh/P |
| 2D tensor parallel computing | O(1/xy) | O(1/xy) | O(1/xy) | 2bs[e(x-1)+h (y-1)]/xy |
| 3D tensor parallel computing | O(1/xyz) | O(1/xyz) | O(1/xyz) | 2[bse(x-1)+bsh (y-1)+he(z-1)]/xyz |

- The number of processors is in order P, P = xy, P = xyz
- The tensor shape with two matmul operations is: activation: (bs, h), weight1: (h, e), weight2: (e, h)

### Related Interfaces

1. `mindspore.ops.MatMul().add_prim_attr("enable_nd_tp", True)`: To turn on the 2D/3D communication/computation mode using AllGather, MatMul and ReduceScatter, you must configure MatMul's shard slice using Layout.
2. `mindspore.ops.BatchMatMul().add_prim_attr("enable_nd_tp", True)`: To turn on the 2D/3D communication/computation mode using AllGather, MatMul and ReduceScatter, you must configure MatMul's shard slice using Layout.

With the above switch turned on, shard slicing determines whether 2D or 3D parallel mode is used depending on the in_strategy:

1. 2D tensor parallel in_strategy configurations, mainly limiting the slicing rule for the reduce of the activation tensor and the last two dimensions of the weight tensor: `mindspore.ops.MatMul().shard(in_strategy = (layout("None",("x","y") ), layout("x", "y")))`

2. 3D tensor parallel in_strategy configurations, mainly limiting the activation tensor and the last two dimensions of the weight tensor: `mindspore.ops.MatMul().shard(in_strategy = (layout(("z","y"),"x" ), layout(("x","z"), "y")))`

> 1. The x, y, z in the above slicing rule, i.e., the number of slicing devices for high-dimensional TP in different dimensions, should be determined by the user according to the shape of the tensor involved in the computation, and the principle of evenly slicing the weight tensor configuration has a better performance gain.
> 2. If MatMul / BatchMatMul has transpose_a or trainspose_b turned on, the slice layout involved in the high-dimensional TP is also switched to the corresponding position.

Taking the typical `MatMul -> Other Computational Operators -> MatMul` model structure of the Attention and FeedForward layers of a large model as an example, the computational communication behaviors of 1D, 2D, and 3D models in parallel are shown below.

## Operation Practice

The following is an illustration of 2D tensor parallel operation in an Ascend stand-alone 8-card environment, using the `MatMul -> Other Computational Operators -> MatMul` operator structure, which is common in large models, as an example:

### Sample Code Description

> Download the full sample code: [high_dimension_tensor_parallel](https://gitee.com/mindspore/docs/tree/br_base/docs/sample_code/high_dimension_tensor_parallel).

The directory structure is as follows:

```text
└─ sample_code
    ├─ high_dimension_tensor_parallel
       ├── high_dimension_tensor_parallel.py
       └── run.sh
    ...
```

Among them, `high_dimension_tensor_parallel.py` is the script that defines the network structure and the running process. `run.sh` is the execution script.

### Configuring a Distributed Environment

Initialize the communication with init.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
init()
```

### Constructing the Network and Computing

The operator definition needs to call the add_prim_attr method to specify the MatMul operator to open the high-dimensional TP, and specify the Matmul operator slice method via Layout. Initialization of network parameters is deferred by the `no_init_parameters` interface and parallel mode is set to semi-automatic parallel mode by wrapping `net` via `AutoParallel`. The code is as follows:

```python
# sample code
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.parallel import Layout
from mindspore.common.initializer import initializer
from mindspore.nn.utils import no_init_parameters

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc1_weight = ms.Parameter(initializer("normal", [256, 512], ms.float32))
        self.fc2_weight = ms.Parameter(initializer("normal", [512, 256], ms.float32))
        self.matmul1 = ops.MatMul()
        self.relu = ops.ReLU()
        self.matmul2 = ops.MatMul()

    def construct(self, x):
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu(x)
        output = self.matmul2(x, self.fc2_weight)
        return output

with no_init_parameters():
    net = Network()

in_layout = Layout((2, 4), ("x", "y"))
net.matmul1.add_prim_attr("enable_nd_tp", True)
net.matmul1.shard(in_strategy = (in_layout("None",("x","y")), in_layout("x", "y")))
net.relu.shard(in_strategy = (in_layout("None", ("y","x")),))
net.matmul2.add_prim_attr("enable_nd_tp", True)
net.matmul2.shard(in_strategy = (in_layout("None", ("y","x")), in_layout("y","x")))

input_data = Tensor(np.ones((1024, 256)), dtype=ms.float32)
net = AutoParallel(net, parallel_mode="semi_auto")
output=net(input_data)
print("The output is:", output)
```

### Running a Standalone Eight-Card Script

Next, the corresponding scripts are called by commands, using the `msrun` startup method and the 8-card distributed training script as an example:

```bash
bash run.sh
```

After running, the log results are saved in `./log_output/worker_*.log`, and an example is shown below:

```text
...
The output is: [[-0.02838172 0.00305654 ... 0.02173008]
 ...
 [-0.02838172 0.00305654 ... 0.02173008]]
...
```