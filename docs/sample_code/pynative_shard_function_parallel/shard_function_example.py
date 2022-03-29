"""PyNative Shard Function Parallel Example"""
import numpy as np
import mindspore as ms
from mindspore import Tensor, Model
from mindspore.communication import init
from mindspore import nn
from mindspore import context

context.set_context(mode=context.PYNATIVE_MODE)
context.set_auto_parallel_context(parallel_mode=context.ParallelMode.AUTO_PARALLEL,
                                  search_mode="sharding_propagation", device_num=8)
init()

class BasicBlock(nn.Cell):
    """A base layer with two dense layer"""
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.dense1 = nn.Dense(10, 10)
        self.gelu = nn.GELU()
        self.dense2 = nn.Dense(10, 10)
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
        self.block1.shard(in_strategy=((1, 8),), out_strategy=(None,))

    def construct(self, x):
        # block1 is executed as GRAPH. The inputs/outputs layouts follow the user definition and the slice strategy for inner ops are obtained by auto search
        x = self.block1(x)
        # block2 and block3 are executed as PyNative mode.
        x = self.block2(x)
        x = self.block3(x)
        return x

net = Net1()
model = Model(net)
input_data = Tensor(np.random.normal(size=(16, 10)), ms.float32)
output = model.train_network(input_data)
