# 批次图训练网络

<a href="https://gitee.com/mindspore/docs/blob/master/docs/graphlearning/docs/source_zh_cn/batched_graph_training_GIN.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>
&nbsp;&nbsp;

## 概述

在本例中将展示如何基于图同构网络的进行社会关系网络分类。

GIN的灵感来自GNN和Weisfeiler-Lehman (WL)图同构测试。WL测试是一个强大的测试，可以区分广泛的图类。如果GNN的聚合方案具有高度的表达能力，并且可以建模内射函数，GNN可以具有与WL测试一样大的鉴别力。

IMDB-BINARY是一个电影协作数据集，由1000名在IMDB中扮演电影角色的演员的角色网络组成。在每张图中，节点代表演员，如果他们出演过同一部电影，在节点直接建立一条边。这些图都来源于动作或浪漫电影。
分批次从IMDB-BINARY数据集中取出图数据，每张图都是由演员构成的电影，利用GIN对图进行分类，预测电影属于什么风格。

批次图模式中每次能够对多张图同时进行训练，并且每张图的节点数/边数都完全不同。mindspore_gl提供了构建虚拟图的方法将对批次内图整合成一张整图，并对整图数据进行统一，以降低内存消耗及加速计算。

> 下载完整的样例[GIN](https://gitee.com/mindspore/graphlearning/tree/master/model_zoo/gin)代码。

## GIN原理

论文链接：[How Powerful are Graph Neural Networks?](https://arxiv.org/pdf/1810.00826.pdf)

## 定义网络结构

GINConv将图`g`解析为`BatchedGraph`，与`Graph`相比`BatchedGraph`能够支持更多图操作。输入的数据为整图，但是每张子图进行节点特征更新时，还是能根据自身的节点找到对应的邻居节点，而不会连接到其他子图的节点。

mindspore_gl.nn提供了GINConv的API可以直接调用。使用GINConv，再配合批次归一化、池化等操作实现一个多层的GinNet网络代码如下：

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

GINConv执行的更多细节可以看mindspore_gl.nn.GINConv的[API](https://gitee.com/mindspore/graphlearning/blob/master/mindspore_gl/nn/conv/ginconv.py)代码。

## 构造数据集

从mindspore_gl.dataset调用了IMDB-BINARY的数据集，调用方法可以参考[GCN](https://www.mindspore.cn/graphlearning/docs/zh-CN/master/full_training_of_GCN.html#%E6%9E%84%E9%80%A0%E6%95%B0%E6%8D%AE%E9%9B%86)。然后利用mindspore_gl.dataloader.RandomBatchSampler定义了一个采样器，来生成采样索引。
MultiHomoGraphDataset根据采样索引从数据集里获取数据，将返回数据打包成batch，做出数据集的生成器。构建生成器后，调用mindspore.dataset.GeneratorDataset的API，完成数据加载器构建。

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

利用mindspore_gl.graph.BatchHomoGraph将多张子图合并成一张整图。在模型训练时，batch内所有图将以一张整图的形式进行计算。

为了减少计算图的生成，加快计算速度，生成器在返回数据时，将每个batch中的数据统一到相同的尺寸。

假设节点数`node_size`与边数`edge_size`，并满足batch内所有图数据的节点数之和与边数之和都要都小于等于`node_size * batch`和`edge_size * batch`。
在batch内新建张虚拟图，使得batch内图节点数和、边数和等于`node_size * batch`和`edge_size * batch`。在计算loss时，这张图将不参与计算。

调用mindspore_gl.graph.PadArray2d定义节点和边特征填充的操作，将虚拟图上的节点特征和边特征都设置为0。
调用mindspore_gl.graph.PadHomoGraph定义对图结构上的节点和边进行填充的操作，使得batch内节点数等于`node_size * batch`，边数等于`edge_size * batch`。

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

## 定义loss函数

由于本次任务为分类任务，可以采用交叉熵来作为损失函数，实现方法与[GCN](https://www.mindspore.cn/graphlearning/docs/zh-CN/master/full_training_of_GCN.html#%E5%AE%9A%E4%B9%89loss%E5%87%BD%E6%95%B0)类似。

与GCN不同的是，本次教程为图分类，因此在解析批次图时，调用的为mindspore_gl.BatchedGraph接口。

在`g.graph_mask`中最后一位为虚拟图的mask，等于0，因此在计算loss时，最后1个值也为0。

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

## 网络训练和验证

### 设置环境变量

环境变量设置方法可以[GCN](https://www.mindspore.cn/graphlearning/docs/zh-CN/master/full_training_of_GCN.html#%E8%AE%BE%E7%BD%AE%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F)。

### 定义训练网络

实例化模型主体GinNet以及LossNet和优化器。
将LossNet实例和optimizer传入mindspore.nn.TrainOneStepCell构建一个单步训练网络train_net。
实现方法与GCN类似，可以参考[GCN](https://www.mindspore.cn/graphlearning/docs/zh-CN/master/full_training_of_GCN.html#%E5%AE%9A%E4%B9%89%E8%AE%AD%E7%BB%83%E7%BD%91%E7%BB%9C)。

### 网络训练及验证

由于是批次图训练，构图时调用的API为mindspore_gl.BatchedGraphField，与mindspore_gl.GraphField不同的是，增加了`node_map_idx`、`edge_map_idx`、`graph_mask`三个参数。
其中在`graph_mask`为batch中每个图的掩码信息，由于最后1张图为虚构图，因此在`graph_mask`数组中，最后1位为0，其余为1。

```python
from mindspore_gl import BatchedGraph, BatchedGraphField

for data in train_dataloader:
    row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask, label, node_feat, edge_feat = data
    batch_homo = BatchedGraphField(row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask)
    output = net(node_feat, edge_feat, *batch_homo.get_batched_graph()).asnumpy()
```

## 执行并查看结果

### 运行过程

运行程序后，进行代码翻译并开始训练。

### 执行结果

执行脚本[trainval_imdb_binary.py](https://gitee.com/mindspore/graphlearning/blob/master/model_zoo/gin/trainval_imdb_binary.py)启动训练。

```bash
cd model_zoo/gin
python trainval_imdb_binary.py --data_path={path}
```

其中`{path}`为数据集存放路径。

可以看到训练的结果如下：

```bash
...
Epoch 52, Time 3.547 s, Train loss 0.49981827, Train acc 0.74219, Test acc 0.594
Epoch 53, Time 3.599 s, Train loss 0.5046462, Train acc 0.74219, Test acc 0.656
Epoch 54, Time 3.505 s, Train loss 0.49653444, Train acc 0.74777, Test acc 0.766
Epoch 55, Time 3.468 s, Train loss 0.49411067, Train acc 0.74219, Test acc 0.750
```

在IMDBBinary最好的验证精度为：0.766
