# 在Cora数据集上基于图卷积网络的半监督分类

<a href="https://gitee.com/mindspore/docs/blob/master/docs/graphlearning/docs/source_zh_cn/full_training_of_GCN.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>
&nbsp;&nbsp;

## 概述

图卷积网络(GCN)于2016年提出，旨在对图结构数据进行半监督学习。提出了一种基于卷积神经网络的有效变体的可扩展方法，该方法直接在图上操作。该模型在图边的数量上线性缩放，并学习编码本地图结构和节点特征的隐藏层表示。

Cora数据集包括2708份科学出版物，分为七类之一。引文网络由10556个链接组成。数据集中的每个发布都由0/1值单词向量描述，指示词典中相应单词的不存在/存在。该词典由1433个独特的单词组成。

将Cora的文献的分类作为标签，文献的单词向量作为GCN的节点特征，文献的引用作为边，构图后利用GCN进行训练，判断文献应该属于哪个类。

> 下载完整的样例[GCN](https://gitee.com/mindspore/graphlearning/blob/master/examples/)代码。

## GCN原理

论文链接：[Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

## 定义网络结构

mindspore_gl.nn实现了GCNConv，可以直接导入使用， 用户也可以自己定义卷积层。使用GCNConv实现一个两层的GCN网络代码如下：

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

其中定义的GCNNet继承于GNNCell。GNNCell中construct函数的最后一项输入必须为Graph或者BatchedGraph，也就是MindSpore Graph Learning内置支持的图结构类。此外必须在文件的头部导入 mindspore便于代码翻译时识别执行后端。

GCNConv的参数data_feat_size为输入节点特征维度，hidden_dim_size为隐层特征维度，n_classes为输出分类的维度，in_deg和out_deg分别为图数据中节点的入度和出度。

具体GCN的实现可以参考mindspore_gl.nn.GCNConv的接口代码：<https://gitee.com/mindspore/graphlearning/blob/master/mindspore_gl/nn/conv/gcnconv.py>。

## 定义loss函数

接下来定义LossNet，包含了网络主干net和loss function两部分，这里利用mindspore.nn.SoftmaxCrossEntropyWithLogits实现交叉熵loss。

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

其中net可以通过构建一个LossNet的实例传入GCNNet。predict为net输出的预测值，target为预测真实值，由于是整图训练，通过train_mask从整图中获取一部分作为训练数据，仅这部分节点参与loss计算。

LossNet和GCNNet一样继承自GNNCell。

## 构造数据集

在mindspore_gl.dataset目录下提供了一些dataset类定义的参考。可以直接读入一些研究常用数据集，这里用cora数据集为例， 输入数据路径data_path即可构建数据类。

```python
from mindspore_gl.dataset import CoraV2

ds = CoraV2(args.data_path)
```

其中[Cora](https://data.dgl.ai/dataset/cora_v2.zip)数据下载后，解压路径即为args.data_path。

## 网络训练和验证

### 设置环境变量

环境变量的设置同MindSpore其他网络训练，特别的是设置enable_graph_kernel=True可以启动图算编译优化，加速图模型的训练。

```python
import mindspore.context as context

if train_args.fuse:
    context.set_context(device_target="GPU", save_graphs=True, save_graphs_path="./computational_graph/",
                        mode=context.GRAPH_MODE, enable_graph_kernel=True,
                        graph_kernel_flags="--enable_expand_ops=Gather --enable_cluster_ops=TensorScatterAdd,"
                                           "UnsortedSegmentSum, GatherNd --enable_recompute_fusion=false "
                                           "--enable_parallel_fusion=true ")
else:
    context.set_context(device_target="GPU", mode=context.PYNATIVE_MODE)
```

### 定义训练网络

图神经网络的训练如同其他监督学习模型，除了实例化模型主体GCNNet以及LossNet，还需定义优化器，这里用的mindspore.nn.Adam。

将LossNet实例和optimizer传入mindspore.nn.TrainOneStepCell构建一个单步训练网络train_net。

```python
import mindspore.nn as nn

net = GCNNet(data_feat_size=feature_size,
             hidden_dim_size=train_args.num_hidden,
             n_classes=ds.n_classes,
             dropout=train_args.dropout,
             activation=ms.nn.ELU)
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=train_args.lr, weight_decay=train_args.weight_decay)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)
```

### 网络训练及验证

由于是整图训练，一步训练就覆盖了整个数据集，每个epoch即为一步训练，同样验证节点通过test_mask获取，验证准确率的计算只需取出整图中的验证节点与真实值label进行比较计算： 预测值与真实值一致即为正确，正确节点数count与验证节点总数的比值即为验证准确率。

```python
for e in range(train_args.epochs):
    beg = time.time()
    train_net.set_train()
    train_loss = train_net()
    end = time.time()
    dur = end - beg
    if e >= warm_up:
        total = total + dur

    test_mask = ds.test_mask
    if test_mask is not None:
        net.set_train(False)
        out = net(ds.x, ds.in_deg, ds.out_deg, ds.g.src_idx, ds.g.dst_idx, ds.g.n_nodes, ds.g.n_edges).asnumpy()
        labels = ds.y.asnumpy()
        predict = np.argmax(out[test_mask], axis=1)
        label = labels[test_mask]
        count = np.equal(predict, label)
        print('Epoch time:{} ms Train loss {} Test acc:{}'.format(dur * 1000, train_loss,
                                                                  np.sum(count) / label.shape[0]))
```

## 执行并查看结果

### 运行过程

运行程序后，首先可以看到所有被翻译后的函数的对比图（默认GNNCell中的construct函数会被翻译）。此处展示出GCNConv的翻译对比图，左边为GCNConv的源代码；右边为翻译后的代码。

可以看到graph的API被mindspore_gl替换后的代码实现。比如调用的graph aggregate函数g.sum将被替换为Gather-Scatter的实现。可以看出以节点为中心的编程范式大大降低了图模型实现的代码量。

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

### 开启/关闭翻译界面

代码执行时默认显示翻译对比图，如果需要关闭对比视图，可以进行如下操作：

```python
from mindspore_gl.nn import GNNCell
GNNCell.disable_display()
```

如果需要修改对比视图展示宽度时（默认为200），可以进行如下操作：

```python
from mindspore_gl.nn import GNNCell
GNNCell.enable_display(screen_width=350)
```

### 执行结果

执行脚本[vc_gcn_datanet.py](https://gitee.com/mindspore/graphlearning/blob/master/examples/vc_gcn_datanet.py)启动训练。

```bash
cd examples
python vc_gcn_datanet.py --data-path={path} --fuse=True
```

其中{path}为数据集存放路径。

可以看到训练的结果（截取最后五个epoch）如下：

```bash
...
Epoch 196, Train loss 0.30630863, Test acc 0.822
Epoch 197, Train loss 0.30918056, Test acc 0.819
Epoch 198, Train loss 0.3299482, Test acc 0.819
Epoch 199, Train loss 0.2945389, Test acc 0.821
Epoch 200, Train loss 0.27628058, Test acc 0.819
```

在cora上验证精度：0.82 (论文：0.815)

以上就是整图训练的使用指南。更多样例可参考[examples directory](https://gitee.com/mindspore/graphlearning/blob/master/examples/).
