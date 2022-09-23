"""PyNative Shard Function Parallel Example"""
import numpy as np
import mindspore as ms
from mindspore.communication import init
from mindspore import nn

ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL,
                             search_mode="sharding_propagation", device_num=8)
init()

class BasicBlock(nn.Cell):
    """A base layer with two dense layer"""
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.dense1 = nn.Dense(32, 32)
        self.gelu = nn.GELU()
        self.dense2 = nn.Dense(32, 32)
    def construct(self, x):
        # two dimensional input x
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dense2(x)
        return x

class Net(nn.Cell):
    """A network with three basicblock"""
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = BasicBlock()
        self.block2 = BasicBlock()
        self.block3 = BasicBlock()
    def construct(self, x):
        # All three blocks are executed as PyNative mode.
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class Net1(Net):
    """A shard function example"""
    def __init__(self):
        super(Net1, self).__init__()
        # slice input along the second axis and make output as data-parallel layout
        self.block1.shard(in_strategy=((1, 8),),
                          parameter_plan={'self.block1.dense2.weight': (8, 1)})

    def construct(self, x):
        # block1 is executed as GRAPH.
        x = self.block1(x)
        # block2 and block3 are executed as PyNative mode.
        x = self.block2(x)
        x = self.block3(x)
        return x

net = Net1()
model = ms.Model(net)
input_data = ms.Tensor(np.random.normal(size=(16, 32)), ms.float32)
output = model.train_network(input_data)
print("=========Finish=========")
