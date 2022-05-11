"""Linear Regression Tutorial
This sample code is applicable to CPU, GPU and Ascend.
"""
import numpy as np
from mindspore import dataset as ds
from mindspore.common.initializer import Normal
from mindspore import nn, Model, set_context, GRAPH_MODE
from mindspore import LossMonitor

set_context(mode=GRAPH_MODE, device_target="CPU")


def get_data(num, w=2.0, b=3.0):
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = x * w + b + noise
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)


def create_dataset(num_data, batch_size=16, repeat_size=1):
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    input_data = input_data.batch(batch_size)
    input_data = input_data.repeat(repeat_size)
    return input_data


class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        x = self.fc(x)
        return x


if __name__ == "__main__":

    data_number = 1600
    batch_number = 16
    repeat_number = 1
    lr = 0.005
    momentum = 0.9
    net = LinearNet()
    net_loss = nn.loss.MSELoss()
    opt = nn.Momentum(net.trainable_params(), lr, momentum)
    model = Model(net, net_loss, opt)
    ds_train = create_dataset(data_number, batch_size=batch_number, repeat_size=repeat_number)
    model.train(1, ds_train, callbacks=LossMonitor(), dataset_sink_mode=False)
    for param in net.trainable_params():
        print(param, param.asnumpy())
