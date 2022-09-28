"""Operator Parallel Example"""
import numpy as np

from mindspore import Parameter
from mindspore.nn import Cell, Momentum
from mindspore.train import Model
from mindspore.nn import MSELoss
import mindspore.dataset as ds
import mindspore.ops as ops
import mindspore as ms
from mindspore.train import LossMonitor
from mindspore.train import ModelCheckpoint
from mindspore.common.initializer import initializer
from mindspore.communication import init, get_rank

def get_dataset(batch_size, step_per_epoch, in_dim, out_dim):
    np.random.seed(1)
    input_data = np.random.rand(batch_size, in_dim).astype(np.float32)
    label_data = np.random.rand(batch_size, out_dim).astype(np.float32)
    def generate():
        for _ in range(step_per_epoch):
            yield (input_data, label_data)
    return generate


class Net(Cell):
    """define net"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.weight = Parameter(initializer("normal", [self.in_dim, self.hidden_dim]), "w")
        self.weight2 = Parameter(initializer("normal", [self.hidden_dim, self.out_dim]), "w2")

        # 对matmul算子手动设置切分策略
        # 其中(2, 4)表示matmul算子的输入数据在batch维切分为两份，在width维切分为四份
        # (4, 1)表示matmul算子的权重在height维切分为四份
        self.matmul = ops.MatMul().shard(((2, 4), (4, 1)))

        self.relu = ops.ReLU()
        self.matmul2 = ops.MatMul()

    def construct(self, x):
        out = self.matmul(x, self.weight)
        out = self.relu(out)
        out = self.matmul2(out, self.weight2)
        return out

if __name__ == "__main__":
    var_step_per_epoch = 4
    var_single_batch_size = 2
    var_in_dim = 32
    var_hidden_dim = 16
    var_out_dim = 16

    ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU", save_graphs=True, save_graphs_path="../saved_graph")
    # 单机8卡环境，并行模式为全自动并行，策略搜索设置为策略传播算法
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, \
                                 search_mode="sharding_propagation", \
                                 dataset_strategy="data_parallel")

    init("nccl")

    # 获取当前卡的逻辑序号，即rank_id
    rank_id = get_rank()

    # 随机构造数据集
    fake_dataset = get_dataset(var_single_batch_size, var_step_per_epoch, var_in_dim, var_out_dim)
    dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"])

    # 定义网络结构
    net = Net(var_in_dim, var_hidden_dim, var_out_dim)

    # 定义损失函数、callback
    loss = MSELoss()
    callback = [LossMonitor(), ModelCheckpoint(directory="{}".format(rank_id))]

    # 定义优化器
    learning_rate = 0.4
    momentum = 0.1
    epoch_size = 5
    opt = Momentum(net.trainable_params(), learning_rate, momentum)

    # 模型训练
    model = Model(net, loss_fn=loss, optimizer=opt)
    model.train(epoch_size, dataset, callbacks=callback, dataset_sink_mode=False)
