# Double Recursive Strategy Search Algorithm

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/sapp.md)

## Overview

The double recursive strategy search algorithm is based on Symbolic Automatic Parallel Planner (SAPP). The SAPP algorithm is able to instantly generate optimal strategy for huge networks and large-scale slicing. SAPP is modeled based on the principle of parallel, and describes the topology of hardware clusters by building an abstraction machine, and optimizes the cost model through symbolic simplicity. The cost model compares the relative costs of different parallel strategy rather than the predicted absolute delay, thus greatly compressing the search space and guaranteeing minute-level search times for 100-card clusters.

> Hardware platforms supported by the double recursive strategy search algorithm include Ascend, GPU, and need to run in Graph mode.

Related interfaces:

`mindspore.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, search_mode="recursive_programming")`: Set the parallel mode to auto-parallel and the search mode to a double recursive strategy search algorithm.

No additional configuration is required for the double recursive strategy search algorithm, except for the context above.

## Basic Principles

The double recursive strategy search algorithm is a fully automatic operator-level strategy search scheme, where the user does not need to configure the model in any way, and the algorithm automatically searches for parallel policies that minimize the communication cost. The core difficulty of automatic operator-level strategy search is that the exponential slicing may entail a large search space as well as the cost of acquiring the profiling information that is required in constructing the cost model.

For the first problem, the double recursive strategy search algorithm summarizes its symmetric multi-order characteristics by abstracting the AI training cluster, so it can equivalently perform a recursive dichotomy to compress the search space due to the number of devices; on the other hand, the double recursive strategy search algorithm categorizes the communication cost of operators, compares the communication cost within the operators as well as the cost of rearrangement of the operators, and compresses the exponentially complex search complexity to a linear one by ranking the weights of the operators.

For the second problem, the double recursive strategy search algorithm builds a symbolic cost model, whereas the cost model of the traditional approach focuses on how to accurately predict the absolute delay of different strategies. The cost model of the double recursive strategy search algorithm compares the relative cost of different strategies, and thus saves significantly the cost of profiling.

Therefore, the double recursive strategy search algorithm is able to quickly generate optimal strategies for huge networks and large-scale cluster slicing. In summary, the double recursive strategy search algorithm is modeled based on the parallel principle, describes the hardware cluster topology by building an abstract machine, and simplifies the cost model by symbolization. Its cost model compares not the predicted absolute latency, but the relative cost of different parallel strategies, which can greatly compress the search space and guarantee minute-level search times for 100-card clusters.

## Operation Practice

The following is an illustration of the double recursive strategy search algorithm using the Ascend or GPU stand-alone 8-card example:

### Example Code Description

> Download the complete example code: [sapp](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/sapp).

The directory structure is as follows:

```text
└─ sample_code
    ├─ sapp
       ├── train.py
       └── run.sh
    ...
```

`train.py` is the script that defines the network structure and the training process. `run.sh` is the execution script.

### Configuring Distributed Environment

Specify the run mode, run device, run card number through the context interface. Unlike single card scripts, parallel scripts also need to specify the parallel mode `parallel_mode` as auto-parallel mode, the search mode `search_mode` as double recursive strategy, and initialize HCCL or NCCL communication through init. The `device_target` is automatically specified as the backend hardware device corresponding to the MindSpore package.

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE, save_graphs=2)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, search_mode="recursive_programming")
init()
ms.set_seed(1)
```

### Loading the Dataset, and Defining and Training the Network

The dataset is loaded, the network is defined and the network is trained in the same way as the single card model, with the following code:

```python
import os
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops

def create_dataset(batch_size):
    dataset_path = os.getenv("DATA_PATH")
    dataset = ds.MnistDataset(dataset_path)
    image_transforms = [
        ds.vision.Rescale(1.0 / 255.0, 0),
        ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ds.vision.HWC2CHW()
    ]
    label_transform = ds.transforms.TypeCast(ms.int32)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

data_set = create_dataset(32)

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Dense(28*28, 512)
        self.layer2 = nn.Dense(512, 512)
        self.layer3 = nn.Dense(512, 1)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        logits = self.layer3(x)
        return logits

net = Network()
net.set_train()

optimizer = nn.Momentum(net.trainable_params(), 1e-3, 0.1)
loss_fn = nn.MAELoss()

def forward_fn(data, target):
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)

@ms.jit
def train_step(inputs, targets):
    (loss_value, _), grads = grad_fn(inputs, targets)
    optimizer(grads)
    return loss_value

for epoch in range(10):
    i = 0
    for image, label in data_set:
        loss_output = train_step(image, label)
        if i % 100 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_output))
        i += 1
```

### Running a Stand-alone Eight-Card Script

Next, the corresponding scripts are invoked by commands, using the `mpirun` startup method and the 8-card distributed training script as an example of distributed training:

```bash
bash run.sh
```

After training, the log files are saved to the `log_output` directory. Set context: `save_graphs=2` in `train.py`, and you can print out the IR graphs of the compilation process, where some of the file directories are structured as follows:

```text
├─ log_output
|   └─ 1
|       ├─ rank.0
|       |   └─ stdout
|       ├─ rank.1
|       |   └─ stdout
|       ...
├─ rank_0
|   ├─ step_parallel_begin_xxxx.ir
|   ├─ xx_validate_xxx.ir
|   ...
├─ rank_1
|   ├─ step_parallel_begin_xxxx.ir
|   ├─ xx_validate_xxx.ir
|   ...
...
...
```

The part of Loss results are saved in `log_output/1/rank.*/stdout`, and the example is as follows:

```text
epoch: 0, step: 0, loss is 1.2023287
epoch: 0, step: 100, loss is 1.1595023
epoch: 0, step: 200, loss is 1.1859324
epoch: 0, step: 300, loss is 0.9567921
...
```

In `step_parallel_begin_xxxx.ir`, you can see that each computational operator is configured with a sharding strategy:

```text
...
  %2(logits) = Flatten(%1) primitive_attrs: {BatchParallel: Bool(1)} {in_strategy: ((8, 1, 1, 1))}
      : (<Tensor[Float32], (256, 1, 28, 28)>) -> (<Tensor[Float32], (256, 784)>)
      # Scope: (Default)
  %3([CNode]2161) = Load($(@1_train_step.1797:para3_layer1.weight), %para20_u)
      : (<Ref[Tensor[Float32]], (512, 784), ref_key=:layer1.weight>, <UMonad, NoShape>) -> (<Tensor[Float32], (512, 784)>)
      # Scope: (Default)
  %4(logits) = MatMul(%2, %3) {instance name: matmul} primitive_attrs: {output_names: [output], transpose_a: Bool(0), input_names: [x1, x2], transpose_x2: Bool(1), transpose_x1: Bool(0), transpose_b: Bool(1)} {in_strategy: ((4, 2), (1, 2))}
      : (<Tensor[Float32], (256, 784)>, <Tensor[Float32], (512, 784)>) -> (<Tensor[Float32], (256, 512)>)
      # Scope: (Default)
  %5([CNode]2162) = Load($(@1_train_step.1797:para4_layer1.bias), %para20_u)
      : (<Ref[Tensor[Float32]], (512), ref_key=:layer1.bias>, <UMonad, NoShape>) -> (<Tensor[Float32], (512)>)
      # Scope: (Default)
  %6(logits) = BiasAdd(%4, %5) {instance name: bias_add} primitive_attrs: {output_names: [output], format: "NCHW", input_names: [x, b], data_format: "NCHW"} {in_strategy: ((4, 1), (1))}
      : (<Tensor[Float32], (256, 512)>, <Tensor[Float32], (512)>) -> (<Tensor[Float32], (256, 512)>)
      # Scope: (Default)
  %7(logits) = ReLU(%6) {instance name: relu} primitive_attrs: {output_names: [output], input_names: [x]} {in_strategy: ((4, 1))}
      : (<Tensor[Float32], (256, 512)>) -> (<Tensor[Float32], (256, 512)>)
      # Scope: (Default)
...
```

For example, for the first MatMul operator, the input strategy in_strategy has been configured as ((4, 2), (1, 2)).

```text
input_names: [x1, x2], transpose_x2: Bool(1), transpose_x1: Bool(0), transpose_b: Bool(1)
```

Transpose exists in the second input that represents the MatMul operator.

```text
(<Tensor[Float32], (256, 784)>, <Tensor[Float32], (512, 784)>) -> (<Tensor[Float32], (256, 512)>)
```

The shapes representing the first and second inputs are (256, 784), (512, 784), respectively. The transpose exists in the second input, which outputs a shape of (256, 512).

In `xx_validate_xxx.ir`, you can see that the input and output tensor of each operator is sliced, and some communication operators such as `AllReduce` have been inserted between the original operators of the network:

```text
...
  %14(equiv[CNode]4) = MatMul(%12, %13) {instance name: matmul} primitive_attrs: {output_names: [output], transpose_a: Bool(0), input_names: [x1, x2], transpose_x2: Bool(1), transpose_x1: Bool(0), transpose_b: Bool(1)} cnode_attrs: {related_comm_node_id: "37501"} cnode_primal_attrs: {unique_id: "37896", related_fusion_key: "all_reduce_4-5226697808808137312_1", related_node_id: "34001"} {in_strategy: ((4, 2), (1, 2))}
      : (<Tensor[Float32], (64, 392)>, <Tensor[Float32], (512, 392)>) -> (<Tensor[Float32], (64, 512)>)
      # Scope: (Default)
      # In file /home/liguanxin/anaconda3/envs/py38/lib/python3.8/site-packages/mindspore/nn/layer/basic.py:625/        x = self.matmul(x, self.weight)/
  %15(equiv[CNode]2229) = AllReduce(%14) {instance name: forward_op_15773666391001111732} primitive_attrs: {comm_reuse: Bool(1), group: "2-5004544844489628105", fusion: I64(0), op: "sum", rank_list: (0, 1), group_ranks: "0-1", index: I64(0), group_rank_ids: (0, 1), no_eliminate: Bool(1)} cnode_primal_attrs: {unique_id: "38092", forward_comm_node_unique_id: "37499"}
      : (<Tensor[Float32], (64, 512)>) -> (<Tensor[Float32], (64, 512)>)
      # Scope: (Default)
  %16(equiv[CNode]2162) = Load(%para4_layer1.bias, U) cnode_primal_attrs: {unique_id: "37918"}
      : (<Ref[Tensor[Float32]], (512), ref_key=:layer1.bias>, <UMonad, NoShape>) -> (<Tensor[Float32], (512)>)
      # Scope: (Default)
  %17(equiv[CNode]4) = BiasAdd(%15, %16) {instance name: bias_add} primitive_attrs: {output_names: [output], format: "NCHW", input_names: [x, b], data_format: "NCHW"} cnode_attrs: {related_comm_node_id: "37503"} cnode_primal_attrs: {unique_id: "37916", related_fusion_key: "all_reduce_nccl_world_group_1", related_node_id: "33999"} {in_strategy: ((4, 1), (1))}
      : (<Tensor[Float32], (64, 512)>, <Tensor[Float32], (512)>) -> (<Tensor[Float32], (64, 512)>)
      # Scope: (Default)
      # In file /home/liguanxin/anaconda3/envs/py38/lib/python3.8/site-packages/mindspore/nn/layer/basic.py:627/            x = self.bias_add(x, self.bias)/
  %18(equiv[CNode]4) = ReLU(%17) {instance name: relu} primitive_attrs: {output_names: [output], input_names: [x]} cnode_primal_attrs: {unique_id: "37878"} {in_strategy: ((4, 1))}
      : (<Tensor[Float32], (64, 512)>) -> (<Tensor[Float32], (64, 512)>)
...
```

For the first MatMul operator, its two inputs are sliced from the original (256, 784), (512, 784) into (64, 392), (512, 392), and after the transpose of the second input, the output of the operator is (64, 512).

Other startup methods such as dynamic networking and `rank table` startup can be found in [startup methods](https://www.mindspore.cn/tutorials/experts/en/master/parallel/startup_method.html).
