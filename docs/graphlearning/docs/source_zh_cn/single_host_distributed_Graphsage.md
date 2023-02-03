# 单机多卡分布式训练

<a href="https://gitee.com/mindspore/docs/blob/master/docs/graphlearning/docs/source_zh_cn/single_host_distributed_Graphsage.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>
&nbsp;&nbsp;

## 概述

在本例中将展示如何利用Graphsage在大尺寸图上进行单机多卡训练。

GraphSAGE是一个通用的归纳框架，它利用节点特征信息（例如，文本属性）为以前看不见的数据有效地生成节点嵌入。GraphSAGE不是为每个节点训练单个嵌入，而是学习一个函数，该函数通过从节点的本地邻居中采样和聚合特征来生成嵌入。

在Reddit数据集中，作者对50个大型社区进行了抽样调查，并构建了一个帖子到帖子的图，如果同一用户对这两个帖子都发表了评论，则连接帖子。每个帖子的标签为所属的社区。该数据集总共包含232965个帖子，平均度为492。

由于Reddit数据集较大，为了减少GraphSAGE训练时间，本例中在单机上执行分布式模型训练，以加快模型训练。

> 下载完整的样例[GraphSAGE](https://gitee.com/mindspore/graphlearning/tree/master/model_zoo/graphsage)代码。

## GraphSAGE原理

论文链接：[Inductive representation learning on large graphs](https://proceedings.neurips.cc/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf)

## 设置运行脚本

在不同设备上，分布式训练的方式也不相同。

在GPU硬件平台上，MindSpore分布式并行训练中的通信使用的是英伟达集合通信库NVIDIA Collective Communication Library(简称为NCCL)。

更多关于在[GPU](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_gpu.html)上进行分布式训练的细节。

```bash
# GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_NUM=8
rm -rf device
mkdir device
cp -r src ./device
cp distributed_trainval_reddit.py ./device
cd ./device
echo "start training"
mpirun --allow-run-as-root -n ${CUDA_NUM} python3 ./distributed_trainval_reddit.py --data-path ${DATA_PATH} --epochs 5 > train.log 2>&1 &
```

MindSpore分布式并行训练的通信使用了华为集合通信库Huawei Collective Communication Library（以下简称HCCL），可以在Ascend AI处理器配套的软件包中找到。同时mindspore.communication.management中封装了HCCL提供的集合通信接口，方便用户配置分布式信息。

更多关于在[Ascend](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_ascend.html)上进行分布式训练的细节。

```bash
# Ascend
RANK_TABLE_FILE=$3
  export RANK_TABLE_FILE=${RANK_TABLE_FILE}
  for((i=0;i<8;i++));
  do
    export RANK_ID=$[i+RANK_START]
    export DEVICE_ID=$[i+RANK_START]
    echo ${DEVICE_ID}
    rm -rf ${execute_path}/device_$RANK_ID
    mkdir ${execute_path}/device_$RANK_ID
    cd ${execute_path}/device_$RANK_ID || exit
    echo "start training"
    python3 ${self_path}/distributed_trainval_reddit.py --data-path ${DATA_PATH} --epochs 2 > train$RANK_ID.log 2>&1 &
  done
```

## 定义网络结构

mindspore_gl.nn提供了SAGEConv的API可以直接调用。使用SAGEConv实现一个两层的GraphSAGE网络代码如下：

```python
class SAGENet(Cell):
    """graphsage net"""
    def __init__(self, in_feat_size, hidden_feat_size, appr_feat_size, out_feat_size):
        super().__init__()
        self.num_layers = 2
        self.layer1 = SAGEConv(in_feat_size, hidden_feat_size, aggregator_type='mean')
        self.layer2 = SAGEConv(hidden_feat_size, appr_feat_size, aggregator_type='mean')
        self.dense_out = ms.nn.Dense(appr_feat_size, out_feat_size, has_bias=False,
                                     weight_init=XavierUniform(math.sqrt(2)))
        self.activation = ms.nn.ReLU()
        self.dropout = ms.nn.Dropout(0.5)

    def construct(self, node_feat, edges, n_nodes, n_edges):
        """graphsage net forward"""
        node_feat = self.layer1(node_feat, None, edges[0], edges[1], n_nodes, n_edges)
        node_feat = self.activation(node_feat)
        node_feat = self.dropout(node_feat)
        ret = self.layer2(node_feat, None, edges[0], edges[1], n_nodes, n_edges)
        ret = self.dense_out(ret)
        return ret
```

SAGEConv执行的更多细节可以看mindspore_gl.nn.SAGEConv的[API](https://gitee.com/mindspore/graphlearning/blob/master/mindspore_gl/nn/conv/sageconv.py)代码。

## 定义loss函数

由于本次任务为分类任务，可以采用交叉熵来作为损失函数，实现方法与[GCN](https://www.mindspore.cn/graphlearning/docs/zh-CN/master/full_training_of_GCN.html#%E5%AE%9A%E4%B9%89loss%E5%87%BD%E6%95%B0)类似。

## 构造数据集

下面以[Reddit](https://data.dgl.ai/dataset/reddit.zip)数据集为例。输入数据路径，构造数据类。

get_group_size用于获取分布式训练的进程总数，get_rank用于获取当前进程的ID。数据加载器的构建方法可以参考[GIN](https://www.mindspore.cn/graphlearning/docs/zh-CN/master/batched_graph_training_GIN.html#%E6%9E%84%E9%80%A0%E6%95%B0%E6%8D%AE%E9%9B%86)。

与GIN不同的时，在本例中采样器调用的是mindspore_gl.dataloader.DistributeRandomBatchSampler。DistributeRandomBatchSampler可以根据进程ID拆分数据集索引，确保每个进程获取的数据集批次的不同部分。

```python
from mindspore_gl.dataset import Reddit
from mindspore.communication import get_rank, get_group_size

rank_id = get_rank()
world_size = get_group_size()
graph_dataset = Reddit(args.data_path)
train_sampler = DistributeRandomBatchSampler(rank_id, world_size, data_source=graph_dataset.train_nodes,
                                             batch_size=args.batch_size)
test_sampler = RandomBatchSampler(data_source=graph_dataset.test_nodes, batch_size=args.batch_size)
train_dataset = GraphSAGEDataset(graph_dataset, [25, 10], args.batch_size, len(list(train_sampler)), single_size)
test_dataset = GraphSAGEDataset(graph_dataset, [25, 10], args.batch_size, len(list(test_sampler)), single_size)
train_dataloader = ds.GeneratorDataset(train_dataset, ['seeds_idx', 'label', 'nid_feat', 'edges'],
                                       sampler=train_sampler, python_multiprocessing=True)
test_dataloader = ds.GeneratorDataset(test_dataset, ['seeds_idx', 'label', 'nid_feat', 'edges'],
                                      sampler=test_sampler, python_multiprocessing=True)
```

mindspore_gl.sampling.sage_sampler_on_homo提供了k-hop的采样方法。在`self.neighbor_nums`为list的形式，设定了每次从中心节点往外的采样点个数。
由于每个点的度数不一样，经过k-hop采样后的数组的尺寸也不一样。通过接口mindspore_gl.graph.PadArray2d将采样得到的结果离散化成5个固定的值。

```python
from mindspore_gl.dataloader.dataset import Dataset
from mindspore_gl.sampling.neighbor import sage_sampler_on_homo

class GraphSAGEDataset(Dataset):
    """Do sampling from neighbour nodes"""
    def __init__(self, graph_dataset, neighbor_nums, batch_size, length, single_size=False):
        self.graph_dataset = graph_dataset
        self.graph = graph_dataset[0]
        self.neighbor_nums = neighbor_nums
        self.x = graph_dataset.node_feat
        self.y = graph_dataset.node_label
        self.batch_size = batch_size
        self.max_sampled_nodes_num = neighbor_nums[0] * neighbor_nums[1] * batch_size
        self.single_size = single_size
        self.length = length

    def __getitem__(self, batch_nodes):
        batch_nodes = np.array(batch_nodes, np.int32)
        res = sage_sampler_on_homo(self.graph, batch_nodes, self.neighbor_nums)
        label = array_kernel.int_1d_array_slicing(self.y, batch_nodes)
        layered_edges_0 = res['layered_edges_0']
        layered_edges_1 = res['layered_edges_1']
        sample_edges = np.concatenate((layered_edges_0, layered_edges_1), axis=1)
        sample_edges = sample_edges[[1, 0], :]
        num_sample_edges = sample_edges.shape[1]
        num_sample_nodes = len(res['all_nodes'])
        max_sampled_nodes_num = self.max_sampled_nodes_num
        if self.single_size is False:
            if num_sample_nodes < floor(0.2*max_sampled_nodes_num):
                pad_node_num = floor(0.2*max_sampled_nodes_num)
            elif num_sample_nodes < floor(0.4*max_sampled_nodes_num):
                pad_node_num = floor(0.4 * max_sampled_nodes_num)
            elif num_sample_nodes < floor(0.6*max_sampled_nodes_num):
                pad_node_num = floor(0.6 * max_sampled_nodes_num)
            elif num_sample_nodes < floor(0.8*max_sampled_nodes_num):
                pad_node_num = floor(0.8 * max_sampled_nodes_num)
            else:
                pad_node_num = max_sampled_nodes_num

            if num_sample_edges < floor(0.2*max_sampled_nodes_num):
                pad_edge_num = floor(0.2*max_sampled_nodes_num)
            elif num_sample_edges < floor(0.4*max_sampled_nodes_num):
                pad_edge_num = floor(0.4 * max_sampled_nodes_num)
            elif num_sample_edges < floor(0.6*max_sampled_nodes_num):
                pad_edge_num = floor(0.6 * max_sampled_nodes_num)
            elif num_sample_edges < floor(0.8*max_sampled_nodes_num):
                pad_edge_num = floor(0.8 * max_sampled_nodes_num)
            else:
                pad_edge_num = max_sampled_nodes_num

        else:
            pad_node_num = max_sampled_nodes_num
            pad_edge_num = max_sampled_nodes_num

        layered_edges_pad_op = PadArray2d(mode=PadMode.CONST, size=[2, pad_edge_num],
                                          dtype=np.int32, direction=PadDirection.ROW,
                                          fill_value=pad_node_num - 1,
                                          )
        nid_feat_pad_op = PadArray2d(mode=PadMode.CONST,
                                     size=[pad_node_num, self.graph_dataset.node_feat_size],
                                     dtype=self.graph_dataset.node_feat.dtype,
                                     direction=PadDirection.COL,
                                     fill_value=0,
                                     reset_with_fill_value=False,
                                     use_shared_numpy=True
                                     )
        sample_edges = sample_edges[:, :pad_edge_num]
        pad_sample_edges = layered_edges_pad_op(sample_edges)
        feat = nid_feat_pad_op.lazy([num_sample_nodes, self.graph_dataset.node_feat_size])
        array_kernel.float_2d_gather_with_dst(feat, self.graph_dataset.node_feat, res['all_nodes'])
        return res['seeds_idx'], label, feat, pad_sample_edges
```

## 网络训练和验证

### 设置环境变量

分布式训练时，采用数据并行方式导入数据。在每个训练步骤结束时，各个进程会统一模型参数，在Ascend上，必须确保每个进程中的数据shape相同。

```python
device_target = str(os.getenv('DEVICE_TARGET'))
if device_target == 'Ascend':
    device_id = int(os.getenv('DEVICE_ID'))
    ms.set_context(device_id=device_id)
    single_size = True
    init()
else:
    init("nccl")
    single_size = False
```

图算编译优化设置可以参考[GCN](https://www.mindspore.cn/graphlearning/docs/zh-CN/master/full_training_of_GCN.html#%E8%AE%BE%E7%BD%AE%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F)。

### 定义训练网络

实例化模型主体以及LossNet和优化器。
实现方法与GCN类似，可以参考[GCN](https://www.mindspore.cn/graphlearning/docs/zh-CN/master/full_training_of_GCN.html#%E5%AE%9A%E4%B9%89%E8%AE%AD%E7%BB%83%E7%BD%91%E7%BB%9C)。

### 网络训练及验证

训练与验证方法可以参考[GCN](https://www.mindspore.cn/graphlearning/docs/zh-CN/master/full_training_of_GCN.html#%E7%BD%91%E7%BB%9C%E8%AE%AD%E7%BB%83%E5%8F%8A%E9%AA%8C%E8%AF%81)。

## 执行并查看结果

### 运行过程

运行程序后，翻译代码并开始训练。

### 执行结果

执行脚本[distributed_run.sh](https://gitee.com/mindspore/graphlearning/blob/master/model_zoo/graphsage/distributed_run.sh)启动训练。

- GPU

    ```bash
    cd model_zoo/graphsage
    bash distributed_run.sh GPU DATA_PATH
    ```

    `{DATA_PATH}`为数据集存放路径。

- Ascend

    ```bash
    cd model_zoo/graphsage
    bash bash distributed_run.sh Ascend DATA_PATH RANK_START RANK_SIZE RANK_TABLE_FILE
    ```

    `{DATA_PATH}`为数据集存放路径。`{ANK_START}`为使用的Ascend卡的第一个ID。`{RANK_SIZE}`为使用的卡的张数。`{RANK_TABLE_FILE}`为'rank_table_*pcs.json'文件的根路径.

可以看到训练的结果如下：

```bash
...
Iteration/Epoch: 30:4 train loss: 0.41629112
Iteration/Epoch: 30:4 train loss: 0.5337528
Iteration/Epoch: 30:4 train loss: 0.42849028
Iteration/Epoch: 30:4 train loss: 0.5358513
rank_id:3 Epoch/Time: 4:76.17579555511475
rank_id:1 Epoch/Time: 4:37.79207944869995
rank_id:2 Epoch/Time: 4:76.04292225837708
rank_id:0 Epoch/Time: 4:75.64319372177124
rank_id:2 test accuracy : 0.9276439525462963
rank_id:0 test accuracy : 0.9305013020833334
rank_id:3 test accuracy : 0.9290907118055556
rank_id:1 test accuracy : 0.9279513888888888
```

在Reddit数据上的验证精度为0.92。
