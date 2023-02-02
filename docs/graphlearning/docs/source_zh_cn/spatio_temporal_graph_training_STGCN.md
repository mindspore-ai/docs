# 利用时空图卷积网络进行交通预测

<a href="https://gitee.com/mindspore/docs/blob/master/docs/graphlearning/docs/source_zh_cn/spatio_temporal_grph_training_STGCN.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>
&nbsp;&nbsp;

## 概述

时空图卷积网络（STGCN）可以解决交通域的时间序列预测问题。实验表明，STGCN通过建模多尺度交通网络，有效地捕获了综合的时空相关性。

METR-LA是一个大规模数据集，从洛杉矶乡村公路网的1500个交通环路探测器收集。此数据集包括速度、道路容量和占用数据，覆盖约3,420英里。将路网构建成图，输入到STGCN网络中，根据历史数据来预测下个时间段的路网信息。

一般图的节点特征形状为`(节点数量, 特征维度)`，时空图中输入的特征形状通常至少为三维`(节点数量, 特征维度, 时间步)`，邻居节点的特征融合处理会更加复杂。并且由于时间维度上进行卷积，`时间步`也会发生变化，计算loss时，需要提前计算好输出时间长度。

> 下载完整的样例[STGCN](https://gitee.com/mindspore/graphlearning/tree/master/model_zoo/stgcn)代码。

## STGCN原理

论文链接： [A deep learning framework for traffic forecasting](https://arxiv.org/pdf/1709.04875.pdf)

## 图拉普拉斯归一化

将图的自环删除，对图进行归一化，得到新的边索引与边权重。
mindspore_gl.graph提供norm的API可以被用于拉普拉斯归一化。边缘索引和边缘权重归一化的代码如下所示:

```python
mask = edge_index[0] != edge_index[1]
edge_index = edge_index[:, mask]
edge_attr = edge_attr[mask]

edge_index = ms.Tensor(edge_index, ms.int32)
edge_attr = ms.Tensor(edge_attr, ms.float32)
edge_index, edge_weight = norm(edge_index, node_num, edge_attr, args.normalization)
```

关于拉普拉斯归一化的更多细节，可以看mindspore_gl.graph.norm的[API](https://gitee.com/mindspore/graphlearning/blob/master/mindspore_gl/graph/norm.py).

## 定义网络结构

mindspore_gl.nn提供了STConv的API可以直接调用。与一般的图卷积层不同，STConv的输入特征为四维，即`(批次内图数量, 时间步, 节点数量, 特征维度)`。输出特征的`时间步`需要根据1D卷积核尺寸、卷积次数来计算。

使用STConv实现一个两层的STGCN网络代码如下：

```python
class STGcnNet(GNNCell):
    """ STGCN Net """
    def __init__(self,
                 num_nodes: int,
                 in_channels: int,
                 hidden_channels_1st: int,
                 out_channels_1st: int,
                 hidden_channels_2nd: int,
                 out_channels_2nd: int,
                 out_channels: int,
                 kernel_size: int,
                 k: int,
                 bias: bool = True):
        super().__init__()
        self.layer0 = STConv(num_nodes, in_channels,
                             hidden_channels_1st,
                             out_channels_1st,
                             kernel_size,
                             k, bias)
        self.layer1 = STConv(num_nodes, out_channels_1st,
                             hidden_channels_2nd,
                             out_channels_2nd,
                             kernel_size,
                             k, bias)
        self.relu = ms.nn.ReLU()
        self.fc = ms.nn.Dense(out_channels_2nd, out_channels)

    def construct(self, x, edge_weight, g: Graph):
        x = self.layer0(x, edge_weight, g)
        x = self.layer1(x, edge_weight, g)
        x = self.relu(x)
        x = self.fc(x)
        return x
```

STConv执行的更多细节可以看mindspore_gl.nn.temporal.STConv的[API](https://gitee.com/mindspore/graphlearning/blob/master/mindspore_gl/nn/temporal/stconv.py)代码。

## 定义loss函数

由于本次任务为回归任务，可以采用最小均方差来作为损失函数。这里调用mindspore.nn.MSELoss实现最小均方差loss。

```python
class LossNet(GNNCell):
    """ LossNet definition """
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.MSELoss()

    def construct(self, feat, edges, target, g: Graph):
        """STGCN Net with loss function"""
        predict = self.net(feat, edges, g)
        predict = ops.Squeeze()(predict)
        loss = self.loss_fn(predict, target)
        return ms.ops.ReduceMean()(loss)
```

## 构造数据集

输入特征为`(批次内图数量, 时间步, 节点数量, 特征维度)`。在时序卷积上时间序列的长度将会发生变化。因此，从数据集获取特征和标签时，输入和输出时间步有相应规范，否则会出现预测值与标签值形状不一致。

限制规范可以参考代码注释。

```python
from mindspore_gl.dataset import MetrLa
metr = MetrLa(args.data_path)
# out_timestep setting
# out_timestep = in_timestep - ((kernel_size - 1) * 2 * layer_nums)
# such as: layer_nums = 2, kernel_size = 3, in_timestep = 12,
# out_timestep = 4
features, labels = metr.get_data(args.in_timestep, args.out_timestep)
```

其中[MetrLa](https://graphmining.ai/temporal_datasets/METR-LA.zip)数据下载后，解压路径即为args.data_path。

## 网络训练和验证

### 设置环境变量

环境变量设置方法可以参考[GCN](https://www.mindspore.cn/graphlearning/docs/zh-CN/master/full_training_of_GCN.html#%E8%AE%BE%E7%BD%AE%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F)。

### 定义训练网络

实例化模型主体以及LossNet和优化器。
实现方法可以参考[GCN](https://www.mindspore.cn/graphlearning/docs/zh-CN/master/full_training_of_GCN.html#%E5%AE%9A%E4%B9%89%E8%AE%AD%E7%BB%83%E7%BD%91%E7%BB%9C)。

### 网络训练及验证

实现方法可以参考[GCN](https://www.mindspore.cn/graphlearning/docs/zh-CN/master/full_training_of_GCN.html#%E7%BD%91%E7%BB%9C%E8%AE%AD%E7%BB%83%E5%8F%8A%E9%AA%8C%E8%AF%81)。

## 执行并查看结果

### 运行过程

运行程序后，翻译代码并开始训练。

### 执行结果

执行脚本[trainval_metr.py](https://gitee.com/mindspore/graphlearning/blob/master/model_zoo/stgcn/trainval_metr.py)启动训练。

```bash
cd model_zoo/stgcn
python trainval_metr.py --data-path={path} --fuse=True
```

其中{path}为数据集存放路径。

可以看到训练的结果如下：

```bash
...
Iteration/Epoch: 600:199 loss: 0.21488506
Iteration/Epoch: 700:199 loss: 0.21441595
Iteration/Epoch: 800:199 loss: 0.21243602
Time 13.162885904312134 Epoch loss 0.21053028
eval MSE: 0.2060675
```

MetrLa的MSE: 0.206
