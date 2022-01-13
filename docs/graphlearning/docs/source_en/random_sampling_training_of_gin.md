# Training the Graph Isomorphism Network (GIN) by Graph Sampling

<a href="https://gitee.com/mindspore/docs/blob/master/docs/Graph Learning/docs/source_zh_cn/random_sampling_training_of_GIN.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>
&nbsp;&nbsp;

## Overview

In many application scenarios of graph learning, when an input graph is too large, graph sampling is usually used for random batch training. During each random batch training, a graph is randomly sampled so as to train the graph in batches.  
MindSpore Graph Learning also provides graph sampling APIs and supports random batch training. You can use MindSpore Graph Learning for random batch training by performing the following steps:

1. Define a network model. You can directly call the API provided by mindspore_gl.nn or define your own graph learning module by referring to the implementation of mindspore_gl.nn.
2. Define a loss function.
3. Generate batch graph data. The following provides an example of constructing graph sampling data.
4. Train and validate the graph sampling network.

The following shows how to use MindSpore Graph Learning to implement GIN random batch training with the IMDBinary dataset. For details about the complete code, see <https://gitee.com/mindspore/graphlearning/blob/master/examples/vc_gin.py>.

## GIN Principles

GIN can implement graph classification tasks. Paper: <https://arxiv.org/abs/1810.00826>

## Defining a Network Model

mindspore_gl.nn implements GINConv which can be directly imported for use. In addition to information aggregation, GIN uses the single mapping function MLP to map node features and applies the mapping function to node features.
In addition to graph operations in the node-centric programming paradigm, MindSpore Graph Learning supports combination with other neural networks, such as GIN.

### Mapping Function

The mapping function is defined as MLP which is based on mindspore.nn.Cell and is a stack of the Dense, BatchNorm1d, and Relu layers. The function applied to node features is defined as ApplyNodeFunc. BN and activation functions are added after the output of MLP.

```python
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore_gl.nn import GNNCell
from mindspore_gl.nn import GINConv
from mindspore_gl import BatchedGraph


class MLP(nn.Cell):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
        num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        '''
        super().__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Dense(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linears = nn.CellList()
            self.batch_norms = nn.CellList()

            self.linears.append(nn.Dense(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Dense(hidden_dim, hidden_dim))
            self.linears.append(nn.Dense(hidden_dim, output_dim))

            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def construct(self, x):
        if self.num_layers == 1:
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = ms.ops.ReLU()(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class ApplyNodeFunc(nn.Cell):
    '''Update the node feature hv with MLP, BN and ReLU.'''

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def construct(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = nn.ReLU()(h)
        return h
```  

### GIN

GinNet consists of GINConv-based graph convolution and graph pooling that summarizes node features. A node feature input by GINConv is a feature obtained after MLP mapping, and a multi-layer graph convolution is a stack of the GINConv, BatchNorm1d, and Relu layers.  
Graph pooling is performed on the output of the hidden layer of each convolution layer in sum or avg mode. Then the linear mapping linears_prediction is performed. The final output is the sum of the dropout values of graph pooling at each layer.

```python
class GinNet(GNNCell):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 final_dropout=0.1,
                 learn_eps=False,
                 graph_pooling_type='sum',
                 neighbor_pooling_type='sum'
                 ):
        super().__init__()
        self.final_dropout = final_dropout
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps

        self.mlps = nn.CellList()
        self.convs = nn.CellList()
        self.batch_norms = nn.CellList()

        if self.graph_pooling_type not in ("sum", "avg"):
            raise SyntaxError("Graph pooling type not supported yet.")
        for layer in range(num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.convs.append(GINConv(ApplyNodeFunc(self.mlps[layer]), learn_eps=self.learn_eps,
                                      aggregation_type=self.neighbor_pooling_type))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears_prediction = nn.CellList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Dense(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Dense(hidden_dim, output_dim))

    def construct(self, x, edge_weight, g: BatchedGraph):
        hidden_rep = [x]
        h = x
        # graph convolution
        for layer in range(self.num_layers - 1):
            h = self.convs[layer](h, edge_weight, g)
            h = self.batch_norms[layer](h)
            h = nn.ReLU()(h)
            hidden_rep.append(h)
        # graph pooling
        score_over_layer = 0
        for layer, h in enumerate(hidden_rep):
            if self.graph_pooling_type == 'sum':
                pooled_h = g.sum_nodes(h)
            else:
                pooled_h = g.avg_nodes(h)
            score_over_layer = score_over_layer + nn.Dropout(self.final_dropout)(
                self.linears_prediction[layer](pooled_h))

        return score_over_layer
```

Note: GinNet is inherited from GNNCell. The last input of the construct function in GNNCell must be a graph or BatchedGraph, that is, the graph structure class supported by MindSpore Graph Learning.

## Defining a Loss Function

Define LossNet, including a network backbone net and a loss function. In this example, mindspore.nn.SoftmaxCrossEntropyWithLogits is used to implement a cross entropy loss.

```python
import mindspore as ms
import mindspore.nn as nn
from mindspore_gl import BatchedGraph
from mindspore_gl.nn import GNNCell


class LossNet(GNNCell):
    ''' LossNet definition '''

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, node_feat, edge_weight, target, g: BatchedGraph):
        predict = self.net(node_feat, edge_weight, g)
        target = ops.Squeeze()(target)
        loss = self.loss_fn(predict, target)
        # Mask Loss
        return ms.ops.ReduceSum()(loss * g.graph_mask)
```

In the preceding code, net can be transferred to GinNet by constructing a LossNet instance. predict indicates the predicted value output by the net, and target indicates the actual value. Because the training is based on the graph sampling, graph_mask is used to obtain a real graph as the training data. Only this part of nodes are involved in the loss calculation.
LossNet and GinNet are inherited from GNNCell.

## Constructing Sampling Data

Constructing random batch data for graph sampling includes data reading, graph data sampling, and batch graph constructing. The dataloader summarizes the data as the interface for generating batch data.

### Data Reading and Sampling

#### Reading

The mindspore_gl.dataset directory contains some dataset class definitions for reference. You can directly read some common datasets. The following uses the IMDB-BINARY dataset as an example. Enter the data path to construct a data class.
The IMDB-BINARY data can be downloaded at <https://networkrepository.com/IMDB-BINARY.php> and decompressed to the input path.

#### Sampler

Call the graph sampler RandomBatchSampler in mindspore_gl.dataloader to generate random batch graph data.

#### Batch Graph Data

Inherit mindspore_gl.dataloader.Dataset to construct the batch graph data. The following describes the implementation in detail.

#### Data Loader

For random batch training, you need to construct the training and test data loaders. In this example, DataLoader in mindspore_gl.dataloader is called to transfer the defined sampler and batch graph data.
MindSpore Graph Learning supports multi-thread sampling. You can specify the number of sampling threads (num_workers).

```python
from mindspore_gl.dataloader import RandomBatchSampler, Dataset, DataLoader
from mindspore_gl.dataset import IMDBBinary

# read data
dataset = IMDBBinary(args.data_path)
# define sampler for train dataset
train_batch_sampler = RandomBatchSampler(dataset.train_graphs, batch_size=args.batch_size)
# define multigraph data
multi_graph_dataset = MultiHomoGraphDataset(dataset, args.batch_size)
# define dataloader support multi-processing
train_dataloader = DataLoader(dataset=multi_graph_dataset, sampler=train_batch_sampler, num_workers=4,
                              persistent_workers=True, prefetch_factor=4)
# define sampler for test dataset
test_batch_sampler = RandomBatchSampler(dataset.val_graphs, batch_size=args.batch_size)
# define dataloader for test dataset
test_dataloader = DataLoader(dataset=multi_graph_dataset, sampler=test_batch_sampler, num_workers=0,
                             persistent_workers=False)
```

#### Constructing graph_mask

When batch graphs are generated, padding is performed on the graphs. constant_graph_mask is used to indicate the real graphs.

```python
import mindspore as ms

np_graph_mask = [1] * (args.batch_size + 1)
np_graph_mask[-1] = 0
constant_graph_mask = ms.Tensor(np_graph_mask, dtype=ms.int32)
```

### Defining Batch Graph Data

Inherit mindspore_gl.dataloader.Dataset to build batch graph data. Combine the small graphs in the batch into a large graph by combining the node and edge information of each small graph in the batch.
To prevent rebuild in graph mode, the number of points and edges in each batch must be the same. That is, the input node_size, edge_size, and feature dimensions are fixed for each batch. Therefore, padding is required.
Padding expands the node feature (node_feat) and edge feature (edge_feat) to the specified size. All the src_idx, dst_idx, and edge_subgraph_idx in the graph change.

```python
from mindspore_gl.graph import BatchHomoGraph, PadArray2d, PadHomoGraph, PadMode, PadDirection
from mindspore_gl.dataloader import Dataset


class MultiHomoGraphDataset(Dataset):
    def __init__(self, dataset, batch_size, node_size=1500, edge_size=15000):
        self._dataset = dataset
        self._batch_size = batch_size
        # construct a big graph from subgraphs in a batch
        self.batch_fn = BatchHomoGraph()
        self.batched_edge_feat = None
        # padding which is needed for graph mode
        self.node_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.CONST, direction=PadDirection.COL,
                                           size=(1500, dataset.num_features), fill_value=0)
        self.edge_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.CONST, direction=PadDirection.COL,
                                           size=(edge_size, dataset.num_edge_features), fill_value=0)
        self.graph_pad_op = PadHomoGraph(n_edge=edge_size, n_node=node_size, mode=PadMode.CONST)
        # generate mask, padding nodes is put in the last graph.
        self.train_mask = np.array([True] * (self._batch_size + 1))
        self.train_mask[-1] = False

    # generate a batchGraph
    def __getitem__(self, batch_graph_idx):
        graph_list = []
        feature_list = []
        for idx in range(batch_graph_idx.shape[0]):
            graph_list.append(self._dataset[batch_graph_idx[idx]])
            feature_list.append(self._dataset.graph_feat(batch_graph_idx[idx]))

        # construct a batchGraph from graph_list
        batch_graph = self.batch_fn(graph_list)
        # padding for batchGraph
        batch_graph = self.graph_pad_op(batch_graph)

        # concatenate node feat in batch as batchGraph node feat
        batched_node_feat = np.concatenate(feature_list)
        # padding node feat of batchGraph
        batched_node_feat = self.node_feat_pad_op(batched_node_feat)

        # get label of batchGraph
        batched_label = self._dataset.graph_label[batch_graph_idx]
        # padding for label
        batched_label = np.append(batched_label, batched_label[-1] * 0)

        # get edge feat for model training
        if self.batched_edge_feat is None or self.batched_edge_feat.shape[0] < batch_graph.edge_count:
            del self.batched_edge_feat
            self.batched_edge_feat = np.ones([batch_graph.edge_count, 1], dtype=np.float32)

        # Node_Map_Idx/Edge_Map_Idx computing
        _ = batch_graph.batch_meta.node_map_idx
        _ = batch_graph.batch_meta.edge_map_idx

        # return new batchGraph
        return batch_graph, batched_label, batched_node_feat, self.batched_edge_feat[:batch_graph.edge_count, :]
```

## Network Training and Validation

### Setting Environment Variables

The settings of environment variables are the same as those for other MindSpore network training. Especially, if enable_graph_kernel is set to True, the graph kernel build optimization is enabled to accelerate the graph model training.

```python
import mindspore.context as context

context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True)
```

### Defining a Training Network

Similar to other supervised learning models, in addition to the instantiation of the model body GinNet and LossNet, the graph neural network training requires the definition of an optimizer. Here, mindspore.nn.Adam is used. A segmented linear learning rate change function learning_rates is transferred.
Input the LossNet instance and optimizer to mindspore.nn.TrainOneStepCell to construct a single-step training network train_net.

```python
import mindspore.nn as nn

net = GinNet(num_layers=args.num_layers,
             num_mlp_layers=args.num_mlp_layers,
             input_dim=dataset.num_features,
             hidden_dim=args.hidden_dim,
             output_dim=dataset.num_classes,
             final_dropout=args.final_dropout,
             learn_eps=args.learn_eps,
             graph_pooling_type=args.graph_pooling_type,
             neighbor_pooling_type=args.neighbor_pooling_type)

learning_rates = nn.piecewise_constant_lr([50, 100, 150, 200, 250, 300, 350],
                                          [0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125, 0.00015625])
optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=learning_rates)
loss = LossNet(net)
train_net = nn.TrainOneStepCell(loss, optimizer)
```

### Network Training and Validation

Because the network is trained in batches by graph sampling, each epoch is a training set. The data downloader (train_dataloader) traverses and obtains the graph data of each batch for computing.
Validation is performed once for each epoch. The validation data downloader (test_dataloader) traverses the validation graph data to compare the predicted value with the actual value label. If the predicted value is consistent with the actual value, the validation is accurate. The ratio of the number of accurate nodes (count) to the total number of validation nodes is the validation accuracy.
Although padding is performed, np_graph_mask is used to obtain the real graph to calculate the loss or accuracy during training or validation .

```python
for epoch in range(args.epochs):
    train_net.set_train(True)
    train_loss = 0
    total_iter = 0
    while True:
        # get batch data from dataloader
        for data in train_dataloader:
            batch_graph, label, node_feat, edge_feat = data
            # construct node feat and edge feat tensor
            node_feat = ms.Tensor.from_numpy(node_feat)
            edge_feat = ms.Tensor.from_numpy(edge_feat)
            label = ms.Tensor.from_numpy(label)
            # construct BatchedGraph
            batch_homo = BatchedGraph(
                ms.Tensor.from_numpy(batch_graph.adj_coo[0]),  # src_idx
                ms.Tensor.from_numpy(batch_graph.adj_coo[1]),  # dst_idx
                ms.Tensor(batch_graph.node_count, ms.int32),  # n_nodes
                ms.Tensor(batch_graph.edge_count, ms.int32),  # n_edges
                ms.Tensor.from_numpy(batch_graph.batch_meta.node_map_idx),  # ver_subgraph_idx
                ms.Tensor.from_numpy(batch_graph.batch_meta.edge_map_idx),  # edge_subgraph_idx
                constant_graph_mask  # graph_mask
            )
            # training
            train_loss += train_net(node_feat, edge_feat, label, *batch_homo) / args.batch_size
            total_iter += 1
            if total_iter == args.iters_per_epoch:
                break
        if total_iter == args.iters_per_epoch:
            break
    train_loss /= args.iters_per_epoch

    # model validation using test data
    train_net.set_train(False)
    test_count = 0
    for data in test_dataloader:
        batch_graph, label, node_feat, edge_feat = data
        node_feat = ms.Tensor.from_numpy(node_feat)
        edge_feat = ms.Tensor.from_numpy(edge_feat)
        batch_homo = BatchedGraph(
            ms.Tensor.from_numpy(batch_graph.adj_coo[0]),
            ms.Tensor.from_numpy(batch_graph.adj_coo[1]),
            ms.Tensor(batch_graph.node_count, ms.int32),
            ms.Tensor(batch_graph.edge_count, ms.int32),
            ms.Tensor.from_numpy(batch_graph.batch_meta.node_map_idx),
            ms.Tensor.from_numpy(batch_graph.batch_meta.edge_map_idx),
            constant_graph_mask
        )
        output = net(node_feat, edge_feat, *batch_homo).asnumpy()
        label = label
        predict = np.argmax(output, axis=1)
        test_count += np.sum(np.equal(predict, label) * np_graph_mask)
    test_acc = test_count / len(test_dataloader) / args.batch_size
    print('Epoch {}, Train loss {}, Test acc {:.3f}'.format(epoch, train_loss, test_acc))
```

## Executing Jobs and Viewing Results

### Running Process

After running the program, you can view the comparison of all translated functions. The following shows the GINConv translation comparison. The left part is the GINConv source code, and the right part is the translated code.
You can see the code implementation after the graph API is replaced by mindspore_gl. For example, the code implementation after the called graph aggregate function g.sum is replaced by Gather-Scatter. It can be seen that the node-centric programming paradigm greatly reduces the amount of code implemented by the graph model.

```bash
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
|    def construct(self, x, edge_weight, g: Graph):                                            1   ||  1      def construct(                                                                           |
|                                                                                                  ||             self,                                                                                |
|                                                                                                  ||             x,                                                                                   |
|                                                                                                  ||             edge_weight,                                                                         |
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
|        g.set_vertex_attr({'h': x})                                                           2   ||  11         VERTEX_SHAPE = SHAPE(x)[0]                                                           |
|                                                                                                  ||  12         h, = [x]                                                                             |
|        g.set_edge_attr({'w': edge_weight})                                                   3   ||  13         EDGE_SHAPE = SHAPE(edge_weight)[0]                                                   |
|                                                                                                  ||  14         w, = [edge_weight]                                                                   |
|        for v in g.dst_vertex:                                                                         4   ||                                                                                                  |
|            if self.agg_type == 'sum':                                                        5   ||  15         if self.agg_type == 'sum':                                                           |
|                ret = g.sum([s.h * e.w for s, e in v.inedges])                                6   ||  16             SCATTER_INPUT_SNAPSHOT12 = GATHER(h, src_idx, 0) * w                             |
|                                                                                                  ||  17             ret = SCATTER_ADD(                                                               |
|                                                                                                  ||                     ZEROS((VERTEX_SHAPE,) + SHAPE(SCATTER_INPUT_SNAPSHOT12)[1:], ms.float32),    |
|                                                                                                  ||                     scatter_dst_idx,                                                             |
|                                                                                                  ||                     SCATTER_INPUT_SNAPSHOT12                                                     |
|                                                                                                  ||                 )                                                                                |
|            elif self.agg_type == 'max':                                                      7   ||  18         elif self.agg_type == 'max':                                                         |
|                ret = g.max([s.h * e.w for s, e in v.inedges])                                8   ||  19             SCATTER_INPUT_SNAPSHOT13 = GATHER(h, src_idx, 0) * w                             |
|                                                                                                  ||  20             ret = SCATTER_MAX(                                                               |
|                                                                                                  ||                     ZEROS((VERTEX_SHAPE,) + SHAPE(SCATTER_INPUT_SNAPSHOT13)[1:], ms.float32),    |
|                                                                                                  ||                     scatter_dst_idx,                                                             |
|                                                                                                  ||                     SCATTER_INPUT_SNAPSHOT13                                                     |
|                                                                                                  ||                 )                                                                                |
|            else:                                                                             9   ||  21         else:                                                                                |
|                ret = g.avg([s.h * e.w for s, e in v.inedges])                                10  ||  22             SCATTER_INPUT_SNAPSHOT14 = GATHER(h, src_idx, 0) * w                             |
|                                                                                                  ||  23             ret = SCATTER_ADD(                                                               |
|                                                                                                  ||                     ZEROS((VERTEX_SHAPE,) + SHAPE(SCATTER_INPUT_SNAPSHOT14)[1:], ms.float32),    |
|                                                                                                  ||                     scatter_dst_idx,                                                             |
|                                                                                                  ||                     SCATTER_INPUT_SNAPSHOT14                                                     |
|                                                                                                  ||                 ) / SCATTER_ADD(                                                                 |
|                                                                                                  ||                     ZEROS((VERTEX_SHAPE, 1), ms.int32),                                          |
|                                                                                                  ||                     scatter_dst_idx,                                                             |
|                                                                                                  ||                     ms.ops.ones_like(scatter_dst_idx)                                            |
|                                                                                                  ||                 )                                                                                |
|            v.h = (1 + self.eps) * v.h + ret                                                  11  ||  24         h = (1 + self.eps) * h + ret                                                         |
|            if self.act is not None:                                                          12  ||  25         if self.act is not None:                                                             |
|                v.h = self.act(v.h)                                                           13  ||  26             h = self.act(h)                                                                  |
|        return [v.h for v in g.dst_vertex]                                                             14  ||  27         return h                                                                             |
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```

### Execution Results

Run the vc_gin.py script to start training.

```bash
cd examples
python vc_gin.py --data-path={path} --fuse=True
```

{path} indicates the dataset storage path.

The training result (of five epochs) is as follows:

```bash
...
Epoch 36, Train loss 0.563505, Test acc 0.708
Epoch 37, Train loss 0.5687446, Test acc 0.729
Epoch 38, Train loss 0.5561573, Test acc 0.734
Epoch 39, Train loss 0.5564656, Test acc 0.719
Epoch 40, Train loss 0.5511463, Test acc 0.714
```

Accuracy validated on IMDB-BINARY: 0.734 (paper: 0.75)

This is the usage guide of random batch training by graph sampling. For more examples, see [examples](<https://gitee.com/mindspore/graphlearning/tree/master/examples>).
