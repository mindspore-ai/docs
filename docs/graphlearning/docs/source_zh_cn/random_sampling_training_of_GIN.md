# 图采样训练GIN网络

<a href="https://gitee.com/mindspore/docs/blob/master/docs/graphlearning/docs/source_zh_cn/random_sampling_training_of_GIN.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>
&nbsp;&nbsp;

## 概述

在很多图学习的应用场景中，当输入的图过大时，往往采用图采样随机批次训练。图采样随机批次训练会每次在图中进行采样，将图分批次进行训练。 MindSpore Graph
Learning也提供了图采样的接口并支持随机批次训练，用户使用MindSpore Graph Learning进行图采样随机批次训练只需要以下几步：

1. 定义网络结构，用户可以直接调用mindspore_gl.nn提供的接口，也可以参考这里的实现自定义图学习模块。
2. 定义loss函数。
3. 生成batch图数据，这里提供了一个构造图采样数据的示例。
4. 图采样网络训练和验证。

本文档展示了使用MindSpore Graph Learning实现GIN在IMDBBinary数据集下图采样随机批次训练。完整代码详见<https://gitee.com/mindspore/graphlearning/blob/master/examples/vc_gin.py>。

## GIN原理

GIN为图同构网络，可以实现图分类的任务。 论文链接：<https://arxiv.org/abs/1810.00826>。

## 定义网络结构

mindspore_gl.nn实现了GINConv，可以直接导入使用，GIN除了信息汇聚还采用单射函数MLP对节点特征进行映射， 将映射函数作用在节点特征上。MindSpore Graph
Learning除了支持以节点为中心编程范式的图操作，支持与其他神经网络的结合，GIN网络就是这样的范例。

### 映射函数

映射函数定义为MLP，MLP基于mindspore.nn.Cell为多层Dense、BatchNorm1d和Relu的堆叠。 作用在节点特征上的函数定义为ApplyNodeFunc在MLP的输出后面增加了BN和激活函数。

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

### GIN网络

GinNet主要包含基于GINConv的图卷积，以及汇总节点特征的graph pooling两部分。 GINConv输入的节点特征为经过MLP映射的特征，多层图卷积为多层GINConv、BatchNorm1d和Relu的堆叠。 graph
pooling在每层卷积的隐层输出上都做，支持sum和avg两种，然后接线性映射linears_prediction， 最后输出为各层graph pooling的dropout的加和。

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

注意：GinNet继承于GNNCell。GNNCell中construct函数的最后一项输入必须为Graph或者BatchedGraph，也就是MindSpore Graph Learning内置支持的图结构类。

## 定义loss函数

接下来定义LossNet，包含了网络主干net和loss function两部分，这里利用mindspore.nn.SoftmaxCrossEntropyWithLogits实现交叉熵loss。

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

其中net可以通过构建一个LossNet的实例传入GinNet。predict为net输出的预测值，target为预测真实值，由于是图采样训练，通过graph_mask从整图中获取真实图作为训练数据，仅这部分节点参与loss计算。
LossNet和GinNet一样继承于GNNCell。

## 图采样数据构造

构造图采样随机批次数据包含数据读入、图数据采样和batch图构造几个部分，由dataloader汇总作为生成batch数据的接口。

### 数据读入和采样

#### 读入

在mindspore_gl.dataset目录下提供了一些dataset类定义的参考。可以直接读入一些研究常用数据集，这里用IMDBBINARY数据集为例， 输入数据路径data_path即可构建数据类。
其中IMDBBinary数据可通过<https://networkrepository.com/IMDB-BINARY.php>链接页面下载获取，解压路径即为输入路径。

#### 采样器

调用mindspore_gl.dataloader里的图采样器RandomBatchSampler生成随机图batch数据。

#### batch图数据

继承mindspore_gl.dataloader.Dataset来自定义构建batch图数据。后面将详细介绍实现。

#### 数据加载器

随机批次训练需要分别构造训练数据和测试数据加载器，这里调用mindspore_gl.dataloader里的DataLoader，传入定义的采样器、batch图数据。MindSpore Graph
Learning支持多线程采样，可以指定采样线程数num_workers。

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

#### 构建graph_mask

在生成batch图时会对图进行padding，constant_graph_mask用来表示哪些图为实际存在的图。

```python
import mindspore as ms

np_graph_mask = [1] * (args.batch_size + 1)
np_graph_mask[-1] = 0
constant_graph_mask = ms.Tensor(np_graph_mask, dtype=ms.int32)
```

### 定义batch图数据

继承mindspore_gl.dataloader.Dataset来构建batch图数据，采用的方法是将batch里的小图组合成一张大图，将batch里的每个小图的节点、边等信息组合起来。
为了使图模式下不进行重新编译，每次batch出来点数和边数必须相同，也就是传入的节点个数node_size和边数edge_size以及特征维度对于每个batch是固定的。 因此需要使用padding的操作。
padding操作会将节点特征node_feat，边特征edge feat扩张为指定大小。Graph里的src_idx，dst_idx以及 edge_subgraph_idx都会改变。

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

## 网络训练和验证

### 设置环境变量

环境变量的设置同MindSpore其他网络训练，特别的是设置enable_graph_kernel=True可以启动图算编译优化，加速图模型的训练。

```python
import mindspore.context as context

context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True)
```

### 定义训练网络

图神经网络的训练如同其他监督学习模型，除了实例化模型主体GinNet以及LossNet，还需定义优化器，这里用的mindspore.nn.Adam，并传入一个分段线性学习率变化函数learning_rates。
将LossNet实例和optimizer传入mindspore.nn.TrainOneStepCell构建一个单步训练网络train_net。

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

### 网络训练及验证

由于是图采样分批次训练，每个epoch为一次训练集数据下载器train_dataloader遍历获取每个batch的图数据进行计算。
每个epoch进行一次验证，通过验证数据下载器test_dataloader遍历验证图数据进行预测与真实值label进行比较计算， 预测值与真实值一致即为正确，正确节点数count与验证节点总数的比值即为验证准确率。
虽然进行了padding，无论是训练还是验证都会通过np_graph_mask取出真实图计算loss或准确率。

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

## 执行并查看结果

### 运行过程

运行程序后，可以看到所有被翻译后的函数的对比图。此处展示出GINConv的翻译对比图，左边为GINConv的源代码；右边为翻译后的代码。
可以看到graph的API会被mindspore_gl替换后的代码实现。比如这里调用的graph aggregate函数g.sum将被替换为Gather-Scatter的实现。 可以看出以节点为中心的编程范式大大降低了图模型实现的代码量。

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

### 执行结果

执行脚本vc_gin.py启动训练。

```bash
cd examples
python vc_gin.py --data-path={path} --fuse=True
```

其中{path}为数据集存放路径。

可以看到训练的结果（截取五个epoch）：

```bash
...
Epoch 36, Train loss 0.563505, Test acc 0.708
Epoch 37, Train loss 0.5687446, Test acc 0.729
Epoch 38, Train loss 0.5561573, Test acc 0.734
Epoch 39, Train loss 0.5564656, Test acc 0.719
Epoch 40, Train loss 0.5511463, Test acc 0.714
```

在IMDBBinary上验证精度：0.734 (论文：0.75)

以上就是图采样随机批次训练的使用指南。更多样例可参考[examples目录](<https://gitee.com/mindspore/graphlearning/tree/master/examples>)。

