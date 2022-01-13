# Entire Graph Training Using Graph Convolutional Network (GCN)

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/graphlearning/docs/source_en/full_training_of_gcn.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>
&nbsp;&nbsp;

## Overview

MindSpore Graph Learning provides abundant dataset read, graph operation, and network module APIs. To train graph neural networks, perform the following steps:

1. Define a network model. You can directly call the API provided by mindspore_gl.nn or define your own graph learning model by referring to the implementation of mindspore_gl.nn.
2. Define a loss function.
3. Construct a dataset. mindspore_gl.dataset provides the function of reading and constructing some public datasets for research.
4. Train and validate the network.

In addition, MindSpore Graph Learning provides a point-centric GNN programming paradigm. Its built-in code parsing functions translate point-centric computing expressions into graph data computing operations. To facilitate debugging, a translation comparison between the user input code and the calculation code is printed during the parsing process.

This document describes how to use MindSpore Graph Learning to train and verify GCN. If graph nodes and edge features can be stored in a GPU, you do not need to sample the entire graph for training.
For details about the code, see <https://gitee.com/mindspore/graphlearning/blob/r1.6/examples/vc_gcn_datanet.py>.

The following is an example of graph training using GCN:

## GCN Principles

Paper: <https://arxiv.org/abs/1609.02907>

## Defining a Network Model

mindspore_gl.nn implements GCNConv, which can be directly imported for use. You can also define your own convolutional layer. The code for implementing a two-layer GCN network using GCNConv is as follows:

```python
import mindspore
from mindspore_gl.nn import GNNCell
from mindspore_gl import Graph
from mindspore_gl.nn import GCNConv


class GCNNet(GNNCell):
    def __init__(self,
                 data_feat_size: int,
                 hidden_dim_size: int,
                 n_classes: int,
                 dropout: float,
                 activation):
        super().__init__()
        self.layer0 = GCNConv(data_feat_size, hidden_dim_size, activation(), dropout)
        self.layer1 = GCNConv(hidden_dim_size, n_classes, None, dropout)

    def construct(self, x, in_deg, out_deg, g: Graph):
        x = self.layer0(x, in_deg, out_deg, g)
        x = self.layer1(x, in_deg, out_deg, g)
        return x
```

GCNNet is inherited from GNNCell. The last input of the construct function in GNNCell must be a graph or BatchedGraph, that is, the graph structure class supported by MindSpore Graph Learning. In addition, you must import mindspore at the header of the file to identify the execution backend when the code is translated.

In GCNConv, data_feat_size indicates the feature dimension of the input node, hidden_dim_size indicates the feature dimension of the hidden layer, n_classes indicates the dimension of the output classification, and in_deg and out_deg indicate the indegree and outdegree of the node in the graph data, respectively.
For details about GCN implementation, see the API code of mindspore_gl.nn.GCNConv: <https://gitee.com/mindspore/graphlearning/blob/r1.6/mindspore_gl/nn/conv/gcnconv.py>.

## Defining a Loss Function

Define LossNet, including a network backbone net and a loss function. In this example, mindspore.nn.SoftmaxCrossEntropyWithLogits is used to implement a cross entropy loss.

```python
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore_gl.nn import GNNCell


class LossNet(GNNCell):
    """ LossNet definition """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, x, in_deg, out_deg, train_mask, target, g: Graph):
        predict = self.net(x, in_deg, out_deg, g)
        target = ops.Squeeze()(target)
        loss = self.loss_fn(predict, target)
        loss = loss * train_mask
        return ms.ops.ReduceSum()(loss) / ms.ops.ReduceSum()(train_mask)
```

In the preceding code, net can be transferred to GCNNet by constructing a LossNet instance. predict indicates the predicted value output by the net, and target indicates the actual value. Because the training is based on the entire graph, train_mask is used to obtain a part of the entire graph as the training data. Only this part of nodes are involved in the loss calculation.
LossNet and GCNNet are inherited from GNNCell.

## Constructing a Dataset

The mindspore_gl.dataset directory contains some dataset class definitions for reference. You can directly read some common datasets. The following uses the CORA dataset as an example. Enter the data path to construct a data class.

```python
from mindspore_gl.dataset import CoraV2

ds = CoraV2(args.data_path)
```

The CORA data can be downloaded at <https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz> and decompressed to args.data_path.

## Network Training and Validation

### Setting Environment Variables

The settings of environment variables are the same as those for other MindSpore network training. Especially, if enable_graph_kernel is set to True, the graph kernel build optimization is enabled to accelerate the graph model training.

```python
import mindspore.context as context

context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True)
```

### Defining a Training Network

Similar to other supervised learning models, in addition to the instantiation of the model body GCNNet and LossNet, the graph neural network training requires the definition of an optimizer. Here, mindspore.nn.Adam is used.
Input the LossNet instance and optimizer to mindspore.nn.TrainOneStepCell to construct a single-step training network train_net.

```python
import mindspore.nn as nn

net = GCNNet(data_feat_size=feature_size,
             hidden_dim_size=args.num_hidden,
             n_classes=ds.n_classes,
             dropout=args.dropout,
             activation=ms.nn.ELU)
optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay)
loss = LossNet(net)
train_net = nn.TrainOneStepCell(loss, optimizer)
```

### Network Training and Validation

Because the entire graph is trained, one training step covers the entire dataset. Each epoch is one training step. Similarly, the verification node is obtained through test_mask. To calculate the verification accuracy, you only need to compare the verification node in the entire graph with the actual value label.
If the predicted value is consistent with the actual value, the verification is correct. The ratio of the number of correct nodes (count) to the total number of verification nodes is the verification accuracy.

```python
for e in range(epochs):
    train_net.set_train()
    # input Graph with * because graph is a name tuple
    train_loss = train_net(ds.x, ds.in_deg, ds.out_deg, ms.Tensor(ds.train_mask, ms.float32), ds.y, *ds.g)

    net.set_train(False)
    out = net(ds.x, ds.in_deg, ds.out_deg, *ds.g).asnumpy()
    # validation
    test_mask = ds.test_mask
    labels = ds.y.asnumpy()
    predict = np.argmax(out[test_mask], axis=1)
    label = labels[test_mask]
    count = np.equal(predict, label)
    test_acc = np.sum(count) / label.shape[0]
    print('Epoch {}, Train loss {}, Test acc {:.3f}'.format(e, train_loss, np.sum(count) / label.shape[0]))
```

## Executing Jobs and Viewing Results

### Running Process

After running the program, you can view the comparison diagram of all translated functions. By default, the construct function in GNNCell is translated. The following figure shows the GCNConv translation comparison. The left part is the GCNConv source code, and the right part is the translated code.
You can see the code implementation after the graph API is replaced by mindspore_gl. For example, the code implementation after the called graph aggregate function g.sum is replaced by Gather-Scatter.
It can be seen that the node-centric programming paradigm greatly reduces the amount of code implemented by the graph model.

```bash
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
|    def construct(self, x, in_deg, out_deg, g: Graph):                                        1   ||  1      def construct(                                                                           |
|                                                                                                  ||             self,                                                                                |
|                                                                                                  ||             x,                                                                                   |
|                                                                                                  ||             in_deg,                                                                              |
|                                                                                                  ||             out_deg,                                                                             |
|                                                                                                  ||             src_idx,                                                                             |
|                                                                                                  ||             dst_idx,                                                                             |
|                                                                                                  ||             n_nodes,                                                                             |
|                                                                                                  ||             n_edges,                                                                             |
|                                                                                                  ||             ver_subgraph_idx=None,                                                               |
|                                                                                                  ||             edge_subgraph_idx=None,                                                              |
|                                                                                                  ||             graph_mask=None                                                                      |
|                                                                                                  ||         ):                                                                                       |
|                                                                                                  ||  2          SCATTER_ADD = ms.ops.TensorScatterAdd()                                              |
|                                                                                                  ||  3          SCATTER_MAX = ms.ops.TensorScatterMax()                                              |
|                                                                                                  ||  4          SCATTER_MIN = ms.ops.TensorScatterMin()                                              |
|                                                                                                  ||  5          GATHER = ms.ops.Gather()                                                             |
|                                                                                                  ||  6          ZEROS = ms.ops.Zeros()                                                               |
|                                                                                                  ||  7          SHAPE = ms.ops.Shape()                                                               |
|                                                                                                  ||  8          RESHAPE = ms.ops.Reshape()                                                           |
|                                                                                                  ||  9          scatter_src_idx = RESHAPE(src_idx, (SHAPE(src_idx)[0], 1))                           |
|                                                                                                  ||  10         scatter_dst_idx = RESHAPE(dst_idx, (SHAPE(dst_idx)[0], 1))                           |
|        out_deg = ms.ops.clip_by_value(out_deg, self.min_clip, self.max_clip)                 2   ||  11         out_deg = ms.ops.clip_by_value(out_deg, self.min_clip, self.max_clip)                |
|        out_deg = ms.ops.Reshape()(                                                           3   ||  12         out_deg = ms.ops.Reshape()(                                                          |
|            ms.ops.Pow()(out_deg, -0.5),                                                          ||                 ms.ops.Pow()(out_deg, -0.5),                                                     |
|            ms.ops.Shape()(out_deg) + (1,)                                                        ||                 ms.ops.Shape()(out_deg) + (1,)                                                   |
|        )                                                                                         ||             )                                                                                    |
|        x = self.drop_out(x)                                                                  4   ||  13         x = self.drop_out(x)                                                                 |
|        x = ms.ops.Squeeze()(x)                                                               5   ||  14         x = ms.ops.Squeeze()(x)                                                              |
|        x = x * out_deg                                                                       6   ||  15         x = x * out_deg                                                                      |
|        x = self.fc(x)                                                                        7   ||  16         x = self.fc(x)                                                                       |
|        g.set_vertex_attr({'x': x})                                                           8   ||  17         VERTEX_SHAPE = SHAPE(x)[0]                                                           |
|                                                                                                  ||  18         x, = [x]                                                                             |
|        for v in g.dst_vertex:                                                                9   ||                                                                                                  |
|            v.x = g.sum([u.x for u in v.innbs])                                               10  ||  19         SCATTER_INPUT_SNAPSHOT2 = GATHER(x, src_idx, 0)                                      |
|                                                                                                  ||  20         x = SCATTER_ADD(                                                                     |
|                                                                                                  ||                 ZEROS((VERTEX_SHAPE,) + SHAPE(SCATTER_INPUT_SNAPSHOT2)[1:], ms.float32),         |
|                                                                                                  ||                 scatter_dst_idx,                                                                 |
|                                                                                                  ||                 SCATTER_INPUT_SNAPSHOT2                                                          |
|                                                                                                  ||             )                                                                                    |
|        in_deg = ms.ops.clip_by_value(in_deg, self.min_clip, self.max_clip)                   11  ||  21         in_deg = ms.ops.clip_by_value(in_deg, self.min_clip, self.max_clip)                  |
|        in_deg = ms.ops.Reshape()(ms.ops.Pow()(in_deg, -0.5), ms.ops.Shape()(in_deg) + (1,))  12  ||  22         in_deg = ms.ops.Reshape()(ms.ops.Pow()(in_deg, -0.5), ms.ops.Shape()(in_deg) + (1,)) |
|        x = [v.x for v in g.dst_vertex] * in_deg                                              13  ||  23         x = x * in_deg                                                                       |
|        x = x + self.bias                                                                     14  ||  24         x = x + self.bias                                                                    |
|        return x                                                                              15  ||  25         return x                                                                             |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```

### Execution Results

Run the vc_gcn_datanet.py script to start training.

```bash
cd examples
python vc_gcn_datanet.py --data-path={path} --fuse=True
```

{path} indicates the dataset storage path.

The training result (of the last five epochs) is as follows:

```bash
...
Epoch 196, Train loss 0.30630863, Test acc 0.822
Epoch 197, Train loss 0.30918056, Test acc 0.819
Epoch 198, Train loss 0.3299482, Test acc 0.819
Epoch 199, Train loss 0.2945389, Test acc 0.821
Epoch 200, Train loss 0.27628058, Test acc 0.819
```

Accuracy verified on CORA: 0.82 (thesis: 0.815)

The preceding is the usage guide of the entire graph training. For more examples, see [examples directory](<https://gitee.com/mindspore/graphlearning/tree/r1.6/examples>).