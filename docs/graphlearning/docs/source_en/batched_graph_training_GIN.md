# Batched Graph Training Network

<a href="https://gitee.com/mindspore/docs/blob/master/docs/graphlearning/docs/source_en/batched_graph_training_GIN.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>
&nbsp;&nbsp;

## Overview

In this example, it will show how to classify the social network with Graph Isomorphism Network.

GIN is inspired by the close connection between GNNs and the Weisfeiler-Lehman (WL) graph isomorphism test, a powerful test known to distinguish a broad class of graphs. GNN can have as large discriminative power as the WL test if the GNN’s aggregation scheme is highly expressive and can model injective functions.

IMDB-BINARY is a movie collaboration dataset that consists of the ego-networks of 1,000 actors/actresses who played roles in movies in IMDB. In each graph, nodes represent actors/actress, and there is an edge between them if they appear in the same movie. These graphs are derived from the Action and Romance genres.
Get batched graph data from the IMDB-BINARY dataset. Each graph is a movie composed of actors. The GIN is used to classify the graphs and predict the genres of the movie.

In the batched graph, multiple graphs can be trained at the same time, and the number of nodes/edges of each graph is different. mindspore_gl integrates the sub graph in the batch into a whole graph, and adds a virtual graph to unify the graph data to reduce memory consumption and speed up calculation.

> Download the complete sample code here: [GIN](https://gitee.com/mindspore/graphlearning/tree/master/model_zoo/gin).

## GIN Principles

Paper: [How Powerful are Graph Neural Networks?](https://arxiv.org/pdf/1810.00826.pdf)

## Defining a Network Model

GINConv parses graph `g` into `BatchedGraph`, and `BatchedGraph` can support more graph operations than `Graph`. The input data is the whole graph, but when updating the node features of each subgraph, it can still find the corresponding neighbor nodes according to its own nodes, and will not connect to the nodes of other subgraphs.

mindspore_gl.nn implements GINConv, which can be directly imported for use. The code for implementing a multi-layer GinNet network using GINConv, batch normalization, and pooling is as follows:

```python
class GinNet(GNNCell):
    """GIN net"""
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

        if self.graph_pooling_type not in ('sum', 'avg'):
            raise SyntaxError("graph pooling type not supported.")
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
        """construct function"""
        hidden_rep = [x]
        h = x
        for layer in range(self.num_layers - 1):
            h = self.convs[layer](h, edge_weight, g)
            h = self.batch_norms[layer](h)
            h = nn.ReLU()(h)
            hidden_rep.append(h)

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

For details about GINConv implementation, see the [API](https://gitee.com/mindspore/graphlearning/blob/master/mindspore_gl/nn/conv/ginconv.py) code of mindspore_gl.nn.GINConv.

## Constructing a Dataset

From mindspore_gl.dataset calls the dataset of IMDB-BINARY,the method can refer to [GCN](https://www.mindspore.cn/graphlearning/docs/zh-CN/master/full_training_of_GCN.html#%E6%9E%84%E9%80%A0%E6%95%B0%E6%8D%AE%E9%9B%86). Then use mindpoint_gl.dataloader.RandomBatchSampler defines a sampler and returns the sampling index.
MultiHomeGraphDataset obtains data from the dataset according to the sampling index, packages the data into a batch, and generates the dataset generator.
After building a generator, invoke the API of mindspore.dataset.GeneratorDataset to construct a dataloader.

```python
dataset = IMDBBinary(arguments.data_path)
train_batch_sampler = RandomBatchSampler(dataset.train_graphs, batch_size=arguments.batch_size)
train_multi_graph_dataset = MultiHomoGraphDataset(dataset, arguments.batch_size, len(list(train_batch_sampler)))
test_batch_sampler = RandomBatchSampler(dataset.val_graphs, batch_size=arguments.batch_size)
test_multi_graph_dataset = MultiHomoGraphDataset(dataset, arguments.batch_size, len(list(test_batch_sampler)))

train_dataloader = ds.GeneratorDataset(train_multi_graph_dataset, ['row', 'col', 'node_count', 'edge_count',
                                                                   'node_map_idx', 'edge_map_idx', 'graph_mask',
                                                                   'batched_label', 'batched_node_feat',
                                                                   'batched_edge_feat'],
                                       sampler=train_batch_sampler)

test_dataloader = ds.GeneratorDataset(test_multi_graph_dataset, ['row', 'col', 'node_count', 'edge_count',
                                                                 'node_map_idx', 'edge_map_idx', 'graph_mask',
                                                                 'batched_label', 'batched_node_feat',
                                                                 'batched_edge_feat'],
                                      sampler=test_batch_sampler)
```

Use mindspore_gl.graph.BatchHomeGraph merges multiple sub-graphs into one whole graph. During model training, all graphs in the batch will be calculated in the form of whole graph.

To reduce the generation of calculation graphs and speed up calculation, the generator unifies the data of each batch to the same size during returning data.

Assume number of nodes is `node_size`and number of edges is `edge_size`, which is  satisfies that the sum of nodes and edges for all graph data in batch is less than or equal to `node_size * batch` and `edge_size * batch`.
Create a new virtual graph in the batch, so that the sum of nodes and edges in the batch is equal to `node_size * batch` and `edge_size * batch`.
When calculating loss, this graph will not participate in the calculation.

Call mindspore_gl.graph.PadArray2d to define the operation of node feature filling and edge feature filling, and set the node feature and edge feature on the virtual graph to 0.
Call mindspore_gl.graph.PadHomoGraph to define the operation of filling the nodes and edges on the graph structure, so that the number of nodes in the batch is equal to `node_size * batch`, and the number of edges is equal to `edge_size * batch`.

```python
class MultiHomoGraphDataset(Dataset):
    """MultiHomoGraph Dataset"""
    def __init__(self, dataset, batch_size, length, mode=PadMode.CONST, node_size=50, edge_size=350):
        self._dataset = dataset
        self._batch_size = batch_size
        self._length = length
        self.batch_fn = BatchHomoGraph()
        self.batched_edge_feat = None
        node_size *= batch_size
        edge_size *= batch_size
        if mode == PadMode.CONST:
            self.node_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.CONST, direction=PadDirection.COL,
                                               size=(node_size, dataset.node_feat_size), fill_value=0)
            self.edge_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.CONST, direction=PadDirection.COL,
                                               size=(edge_size, dataset.edge_feat_size), fill_value=0)
            self.graph_pad_op = PadHomoGraph(n_edge=edge_size, n_node=node_size, mode=PadMode.CONST)
        else:
            self.node_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.AUTO, direction=PadDirection.COL,
                                               fill_value=0)
            self.edge_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.AUTO, direction=PadDirection.COL,
                                               fill_value=0)
            self.graph_pad_op = PadHomoGraph(mode=PadMode.AUTO)

        # For Padding
        self.train_mask = np.array([True] * (self._batch_size + 1))
        self.train_mask[-1] = False

    def __getitem__(self, batch_graph_idx):
        graph_list = []
        feature_list = []
        for idx in range(batch_graph_idx.shape[0]):
            graph_list.append(self._dataset[batch_graph_idx[idx]])
            feature_list.append(self._dataset.graph_node_feat(batch_graph_idx[idx]))

        # Batch Graph
        batch_graph = self.batch_fn(graph_list)

        # Pad Graph
        batch_graph = self.graph_pad_op(batch_graph)

        # Batch Node Feat
        batched_node_feat = np.concatenate(feature_list)

        # Pad NodeFeat
        batched_node_feat = self.node_feat_pad_op(batched_node_feat)
        batched_label = self._dataset.graph_label[batch_graph_idx]

        # Pad Label
        batched_label = np.append(batched_label, batched_label[-1] * 0)

        # Get Edge Feat
        if self.batched_edge_feat is None or self.batched_edge_feat.shape[0] < batch_graph.edge_count:
            del self.batched_edge_feat
            self.batched_edge_feat = np.ones([batch_graph.edge_count, 1], dtype=np.float32)

        # Trigger Node_Map_Idx/Edge_Map_Idx Computation, Because It Is Lazily Computed
        _ = batch_graph.batch_meta.node_map_idx
        _ = batch_graph.batch_meta.edge_map_idx

        np_graph_mask = [1] * (self._batch_size + 1)
        np_graph_mask[-1] = 0
        constant_graph_mask = ms.Tensor(np_graph_mask, dtype=ms.int32)
        batchedgraphfiled = self.get_batched_graph_field(batch_graph, constant_graph_mask)
        row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask = batchedgraphfiled.get_batched_graph()
        return row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask, batched_label,\
               batched_node_feat, self.batched_edge_feat[:batch_graph.edge_count, :]
```

## Defining a Loss Function

Since this is a classification task, the cross entropy can be used as the loss function, and the implementation method is similar to that of [GCN](https://www.mindspore.cn/graphlearning/docs/en/master/full_training_of_GCN.html#defining-a-loss-function).

Different from GCN, this tutorial is for graph classification. Therefore, when parsing batch graphs, the mindspore_gl.BatchedGraph interface is invoked.

The last value in `g.graph_mask` is the mask of the virtual graph, which is 0. Therefore, the last loss value is also 0.

```python
class LossNet(GNNCell):
    """ LossNet definition """
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = ms.nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, node_feat, edge_weight, target, g: BatchedGraph):
        predict = self.net(node_feat, edge_weight, g)
        target = ops.Squeeze()(target)
        loss = self.loss_fn(predict, target)
        loss = ops.ReduceSum()(loss * g.graph_mask)
        return loss
```

## Network Training and Validation

### Setting Environment Variables

The method of setting environment variables is similar to that of setting [GCN](https://www.mindspore.cn/graphlearning/docs/en/master/full_training_of_GCN.html#setting-environment-variables).

### Defining a Training Network

Instantiation of the model body GinNet and LossNet and optimizer.
Input the LossNet instance and optimizer to mindspore.nn.TrainOneStepCell to construct a single-step training network train_net.
The implementation method is similar to that of the [GCN](https://www.mindspore.cn/graphlearning/docs/en/master/full_training_of_GCN.html#defining-a-training-network).

### Network Training and Validation

Because the graph is trained in batch, the API invoked during graph composition is mindspore_gl.BatchedGraphField, which is different from mindspore_gl.GraphField. It added the parameters of `node_map_idx`, `edge_map_idx`, and `graph_mask`.
The `graph_mask` is the mask information of each graph in the batch. The last graph is the virtual graph. Therefore, in the `graph_mask`, the last value is 0 and the rest is 1.

```python
from mindspore_gl import BatchedGraph, BatchedGraphField

for data in train_dataloader:
    row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask, label, node_feat, edge_feat = data
    batch_homo = BatchedGraphField(row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask)
    output = net(node_feat, edge_feat, *batch_homo.get_batched_graph()).asnumpy()
```

## Executing Jobs and Viewing Results

### Running Process

After running the program, translate the code and start training.

### Execution Results

Run the [trainval_imdb_binary.py](https://gitee.com/mindspore/graphlearning/blob/master/model_zoo/gin/trainval_imdb_binary.py) script to start training.

```bash
cd model_zoo/gin
python trainval_imdb_binary.py --data_path={path}
```

`{path}` indicates the dataset storage path.

The training result is as follows:

```bash
...
Epoch 52, Time 3.547 s, Train loss 0.49981827, Train acc 0.74219, Test acc 0.594
Epoch 53, Time 3.599 s, Train loss 0.5046462, Train acc 0.74219, Test acc 0.656
Epoch 54, Time 3.505 s, Train loss 0.49653444, Train acc 0.74777, Test acc 0.766
Epoch 55, Time 3.468 s, Train loss 0.49411067, Train acc 0.74219, Test acc 0.750
```

The best accuracy verified on IMDBBinary: 0.766
