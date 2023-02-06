# Spatio-temporal Graph Convolutional Networks for Traffic Forecasting

<a href="https://gitee.com/mindspore/docs/blob/master/docs/graphlearning/docs/source_en/spatio_temporal_grph_training_STGCN.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>
&nbsp;&nbsp;

## Overview

Spatio-Temporal Graph Convolutional Networks (STGCN) can tackle the time series prediction problem in traffic domain. Experiments show that STGCN effectively captures comprehensive spatio-temporal correlations through modeling multi-scale traffic networks.

METR-LA is a large-scale data set collected from 1,500 traffic loop detectors in the Los Angeles rural road network. This data set includes speed, road capacity, and occupancy data and covers approximately 3,420 miles. The road network is constructed into a graph and input to the STGCN network. The road network information in the next time phase is predicted based on the historical data.

The node feature shape of a general graph is `(nodes number, feature dimension)`, but the feature shape of a spatio-temporal graph is usually at least 3-dimensional `(nodes number, feature dimension, time step)`, and the feature fusion processing of neighbor nodes will be more complicated. And due to the convolution in the time dimension, the `time step` will also change. When calculating the loss, it is necessary to calculate the output time length in advance.

> Download the complete sample code here: [STGCN](https://gitee.com/mindspore/graphlearning/tree/master/model_zoo/stgcn).

## STGCN Principles

Paper: [A deep learning framework for traffic forecasting](https://arxiv.org/pdf/1709.04875.pdf)

## Graph Laplacian Normalization

The self-loop of the graph is deleted, and the graph is normalized to obtain the new edge index and edge weight.
mindspore_gl.graph implements norm, which can be used for laplacian normalization. The code for normalization of edge index and edge weight is as follows:

```python
mask = edge_index[0] != edge_index[1]
edge_index = edge_index[:, mask]
edge_attr = edge_attr[mask]

edge_index = ms.Tensor(edge_index, ms.int32)
edge_attr = ms.Tensor(edge_attr, ms.float32)
edge_index, edge_weight = norm(edge_index, node_num, edge_attr, args.normalization)
```

For details about laplacian normalization, see the [API](https://gitee.com/mindspore/graphlearning/blob/master/mindspore_gl/graph/norm.py) code of mindspore_gl.graph.norm.

## Defining a Network Model

mindspore_gl.nn implements STConv, which can be directly imported for use. Different from the general graph convolution layer, the input features of STConv are 4-dimensional, that is, `(batch graphs number, time step, nodes number, feature dimension)`.
The `time step` of the output feature needs to be calculated according to the size of the 1D convolution kernel and the times of convolutions.

The code for implementing a two-layer STGCN network using STConv is as follows:

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

For details about STConv implementation, see the [API](https://gitee.com/mindspore/graphlearning/blob/master/mindspore_gl/nn/temporal/stconv.py) code of mindspore_gl.nn.temporal.STConv.

## Defining a Loss Function

Since this task is a regression task, the minimum mean square error can be used as the loss function. In this example, mindspore.nn.MSELoss is used to implement a mean square error loss.

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

## Constructing a Dataset

Input feature is `(batch graphs number, time step, nodes number, feature dimension)`. The length of the time series changed after time convolution. Therefore, the input and output timestamps must be specified when features and tags are obtained from datasets. Otherwise, the shape of the predicted value is inconsistent with that of the label value.

For details about the restriction specifications, see the code comments.

```python
from mindspore_gl.dataset import MetrLa
metr = MetrLa(args.data_path)
# out_timestep setting
# out_timestep = in_timestep - ((kernel_size - 1) * 2 * layer_nums)
# such as: layer_nums = 2, kernel_size = 3, in_timestep = 12,
# out_timestep = 4
features, labels = metr.get_data(args.in_timestep, args.out_timestep)
```

The [MetrLa](https://graphmining.ai/temporal_datasets/METR-LA.zip) data can be downloaded and decompressed to args.data_path.

## Network Training and Validation

### Setting Environment Variables

The method of setting environment variables is similar to that of setting [GCN](https://www.mindspore.cn/graphlearning/docs/en/master/full_training_of_GCN.html#setting-environment-variables).

### Defining a Training Network

Instantiation of the model body STGcnNet and LossNet and optimizer.
The implementation method is similar to that of the [GCN](https://www.mindspore.cn/graphlearning/docs/en/master/full_training_of_GCN.html#defining-a-training-network).

### Network Training and Validation

The implementation method is similar to that of the [GCN](https://www.mindspore.cn/graphlearning/docs/en/master/full_training_of_GCN.html#network-training-and-validation-1).

## Executing Jobs and Viewing Results

### Running Process

After running the program, translate the code and start training.

### Execution Results

Run the [trainval_metr.py](https://gitee.com/mindspore/graphlearning/blob/master/model_zoo/stgcn/trainval_metr.py) script to start training.

```bash
cd model_zoo/stgcn
python trainval_metr.py --data-path={path} --fuse=True
```

{path} indicates the dataset storage path.

The training result is as follows:

```bash
...
Iteration/Epoch: 600:199 loss: 0.21488506
Iteration/Epoch: 700:199 loss: 0.21441595
Iteration/Epoch: 800:199 loss: 0.21243602
Time 13.162885904312134 Epoch loss 0.21053028
eval MSE: 0.2060675
```

MSE on MetrLa: 0.206
