"""Operator Parallel Example"""
import numpy as np

from mindspore import context, Parameter
from mindspore.nn import Cell, Momentum
from mindspore.ops import operations as ops
from mindspore import Model
from mindspore.nn import SoftmaxCrossEntropyWithLogits
import mindspore.dataset as ds
import mindspore.communication as D
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import ModelCheckpoint
from mindspore.common.initializer import initializer


step_per_epoch = 4

def get_dataset(*inputs):
    def generate():
        for _ in range(step_per_epoch):
            yield inputs
    return generate


class Net(Cell):
    """define net"""
    def __init__(self):
        super().__init__()
        self.matmul = ops.MatMul().shard(((2, 4), (4, 1)))
        self.weight = Parameter(initializer("normal", [32, 16]), "w1")
        self.relu = ops.ReLU().shard(((8, 1),))

    def construct(self, x):
        out = self.matmul(x, self.weight)
        out = self.relu(out)
        return out


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=True)
    D.init()
    rank = D.get_rank()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)

    np.random.seed(1)
    input_data = np.random.rand(16, 32).astype(np.float32)
    label_data = np.random.rand(16, 16).astype(np.float32)
    fake_dataset = get_dataset(input_data, label_data)

    net = Net()

    callback = [LossMonitor(), ModelCheckpoint(directory="{}".format(rank))]
    dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"])
    loss = SoftmaxCrossEntropyWithLogits()

    learning_rate = 0.001
    momentum = 0.1
    epoch_size = 1
    opt = Momentum(net.trainable_params(), learning_rate, momentum)

    model = Model(net, loss_fn=loss, optimizer=opt)
    model.train(epoch_size, dataset, callbacks=callback, dataset_sink_mode=False)
