# Single-host Distributed Training

<a href="https://gitee.com/mindspore/docs/blob/master/docs/graphlearning/docs/source_en/single_host_distributed_Graphsage.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>
&nbsp;&nbsp;

## Overview

In this example, it will show how to do the single-host distributed training of GraphSAGE on large size graphs.

GraphSAGE is a general inductive framework that leverages node feature information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data. Instead of training individual embeddings for each node, GraphSAGE learns a function that generates embeddings by sampling and aggregating features from a node's local neighborhood.

In the Reddit dataset, the authors sampled 50 large communities and constructed a post-to-post graph, linking posts if the same user commented on both posts. Each post is labeled as the community to which it belongs. The dataset contains a total of 232965 posts with an average degree of 492.

Since the Reddit dataset size is large, to reduce the GraphSAGE training time, in this example, distributed model training is performed on single-host to accelerate the model training.

> Download the complete sample code here: [GraphSAGE](https://gitee.com/mindspore/graphlearning/tree/master/model_zoo/graphsage).

## GraphSAGE Principles

Paper: [Inductive representation learning on large graphs](https://proceedings.neurips.cc/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf)

## Setting Running Script

The invoking method of distributed training depending on the device.

On the GPU hardware platform, communication in MindSpore distributed parallel training uses NVIDIA’s collective communication library NVIDIA Collective Communication Library (NCCL for short).

For details about distributed training implementation on [GPU](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html).

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

The Huawei Collective Communication Library (HCCL) is used for the communication of MindSpore parallel distributed training and can be found in the Ascend 310 AI processor software package. In addition, mindspore.communication.management encapsulates the collective communication API provided by the HCCL to help users configure distributed information.

For details about distributed training implementation on [Ascend](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html).

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

## Defining a Network Model

mindspore_gl.nn implements SAGEConv, which can be directly imported for use. You can also define your own convolutional layer. The code for implementing a two-layer GraphSAGE network using SAGEConv is as follows:

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

For details about SAGENet implementation, see the [API](https://gitee.com/mindspore/graphlearning/blob/master/mindspore_gl/nn/conv/sageconv.py) code of mindspore_gl.nn.SAGEConv.

## Defining a Loss Function

Because this task is a classification task, the cross entropy can be used as the loss function, and the implementation method is similar to that of [GCN](https://www.mindspore.cn/graphlearning/docs/en/master/full_training_of_GCN.html#defining-a-loss-function).

## Constructing a Dataset

The following uses the [Reddit](https://data.dgl.ai/dataset/reddit.zip) dataset as an example. Enter the data path to construct a data class.
The get_group_size is used to obtain the total number of processes for distributed training, and the get_rank is used to obtain the ID of the current process. The construction method of dataloader can refer to [GIN](https://www.mindspore.cn/graphlearning/docs/en/master/batched_graph_training_GIN.html#constructing-a-dataset).

Different from GIN, in this example, the sampler is mindpoint_gl.dataloader.DistributeRandomBatchSampler. In DistributeRandomBatchSampler, datasets can be split based on process ID to ensure that each process obtains different part of dataset batches.

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

mindspore_gl.sampling.sage_sampler_on_homo provides a k-hop sampling method. In the list of `self.neighbor_nums`, the number of sampling nodes from the central node to the outside when sampling.
Since the degree of each point is different, the size of the array after k-hop sampling is also different. Discretize the sampling results into 5 fixed values through the API of mindspore_gl.graph.PadArray2d.

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

## Network Training and Validation

### Setting Environment Variables

During distributed training, data is imported in data parallel mode. At the end of each training step, each process unifies the model parameters. On Ascend it must be ensured that the data shape is the same in each process.

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

Graph Operator compilation optimization settings is similar to that of [GCN](https://www.mindspore.cn/graphlearning/docs/en/master/full_training_of_GCN.html#setting-environment-variables).

### Defining a Training Network

Instantiation of the model body SAGENet and LossNet and optimizer.
The implementation method is similar to that of the [GCN](https://www.mindspore.cn/graphlearning/docs/en/master/full_training_of_GCN.html#defining-a-training-network).

### Network Training and Validation

Instantiation of the model body STGcnNet and LossNet and optimizer.
The implementation method is similar to that of the [GCN](https://www.mindspore.cn/graphlearning/docs/en/master/full_training_of_GCN.html#network-training-and-validation-1).

## Executing Jobs and Viewing Results

### Running Process

After running the program, translate the code and start training.

### Execution Results

Run the [distributed_run.sh](https://gitee.com/mindspore/graphlearning/blob/master/model_zoo/graphsage/distributed_run.sh) script to start training.

- GPU

    ```bash
    cd model_zoo/graphsage
    bash distributed_run.sh GPU DATA_PATH
    ```

    {DATA_PATH} indicates the dataset storage path.

- Ascend

    ```bash
    cd model_zoo/graphsage
    bash bash distributed_run.sh Ascend DATA_PATH RANK_START RANK_SIZE RANK_TABLE_FILE
    ```

    {DATA_PATH} indicates the dataset storage path. {ANK_START} is the first Ascend device id be used. {RANK_SIZE} is numbers of Ascend device be used. {RANK_TABLE_FILE} is root path of 'rank_table_*pcs.json' file.

The training result (of the last five epochs) is as follows:

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

Accuracy verified on Reddit: 0.92.
