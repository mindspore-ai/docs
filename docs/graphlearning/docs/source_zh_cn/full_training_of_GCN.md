# 整图训练GCN网络

<a href="https://gitee.com/mindspore/docs/blob/master/docs/ Graph Learning/docs/source_zh_cn/full_training_of_GCN.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>
&nbsp;&nbsp;

## 概述

MindSpore  Graph Learning为用户提供了丰富的数据读入、图操作和网络结构模块接口，用户使用MindSpore  Graph Learning实现训练图神经网络只需要以下几步：

1. 定义网络结构，用户可以直接调用mindspore_gl.nn提供的接口，也可以参考这里的实现自定义图学习模块。
2. 定义loss函数。
3. 构造数据集，mindspore_gl.dataset提供了一些研究用的公开数据集的读入和构造。
4. 网络训练和验证。

此外MindSpore  Graph Learning提供了以点为中心的GNN网络编程范式，内置将以点为中心的计算表达翻译为图数据的计算操作的代码解析函数，为了方便用户调试解析过程将打印出用户输入代码与计算代码的翻译对比图。

本文档展示了使用MindSpore  Graph Learning训练GCN网络以及验证。当用户的图节点和边特征都能存入GPU时，可以不用采样进行整图训练。
具体代码参见<https://gitee.com/mindspore/graphlearning/blob/master/examples/vc_gcn_datanet.py>。

下面为GCN整图训练的示范：

## GCN原理

论文链接：<https://arxiv.org/abs/1609.02907>

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

其中定义的GCNNet继承于GNNCell。GNNCell中construct函数的最后一项输入必须为Graph或者BatchedGraph，也就是MindSpore Graph Learning内置支持的图结构类。此外必须在文件的头部导入
mindspore便于代码翻译时识别执行后端。
GCNConv的参数data_feat_size为输入节点特征维度，hidden_dim_size为隐层特征维度，n_classes为输出分类的维度，in_deg和out_deg分别为图数据中节点的入度和出度。
具体GCN的实现可以参考mindspore_gl.nn.GCNConv的接口代码<https://gitee.com/mindspore/graphlearning/blob/master/mindspore_gl/nn/conv/gcnconv.py>。

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
也可以调用mindspore_gl.temp.GraphDataset读入通用数据。

```python
from mindspore_gl.dataset import CoraV2

ds = CoraV2(args.data_path)
```

其中cora数据可通过<https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz>链接下载,解压路径即为args.data_path。

## 网络训练和验证

### 设置环境变量

环境变量的设置同MindSpore其他网络训练，特别的是设置enable_graph_kernel=True可以启动图算编译优化，加速图模型的训练。

```python
import mindspore.context as context

context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True)
```

### 定义训练网络

图神经网络的训练如同其他监督学习模型，除了实例化模型主体GCNNet以及LossNet，还需定义优化器，这里用的mindspore.nn.Adam。
将LossNet实例和optimizer传入mindspore.nn.TrainOneStepCell构建一个单步训练网络train_net。

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

### 网络训练及验证

由于是整图训练，一步训练就覆盖了整个数据集，每个epoch即为一步训练，同样验证节点通过test_mask获取，验证准确率的计算只需取出整图中的验证节点与真实值label进行比较计算：
预测值与真实值一致即为正确，正确节点数count与验证节点总数的比值即为验证准确率。

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

## 执行并查看结果

### 运行过程

运行程序后，首先可以看到所有被翻译后的函数的对比图（默认GNNCell中的construct函数会被翻译）。此处展示出GCNConv的翻译对比图，左边为GCNConv的源代码；右边为翻译后的代码。
可以看到graph的API被mindspore_gl替换后的代码实现。比如调用的graph aggregate函数g.sum将被替换为Gather-Scatter的实现。
可以看出以节点为中心的编程范式大大降低了图模型实现的代码量。

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

### 执行结果

执行脚本vc_gcn_datanet.py启动训练。

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

以上就是整图训练的使用指南。更多样例可参考[examples目录](<https://gitee.com/mindspore/graphlearning/tree/master/examples>)。