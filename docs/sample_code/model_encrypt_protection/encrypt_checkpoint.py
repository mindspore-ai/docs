# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tutorial of safely exporting and loading CheckPoint files.
This sample code is applicable to CPU, GPU and Ascend in the Linux platform.
"""
import mindspore.nn as nn
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset as ds
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype, set_context, GRAPH_MODE

from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import Accuracy
from mindspore import Model
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore.common.initializer import Normal


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """
    create dataset for train or test

    Args:
        data_path (str): Data path
        batch_size (int): The number of data records in each group
        repeat_size (int): The number of replicated data records
        num_parallel_workers (int): The number of parallel workers
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    # define some parameters needed for data enhancement and rough justification
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # according to the parameters, generate the corresponding data enhancement method
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # using map to apply operations to a dataset
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # process the generated dataset
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


class LeNet5(nn.Cell):
    """Lenet network structure."""
    # define the operator required
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    # use the preceding operators to construct networks
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    set_context(mode=GRAPH_MODE, device_target="CPU")
    lr = 0.01
    momentum = 0.9

    # create the network
    network = LeNet5()

    # define the optimizer
    net_opt = nn.Momentum(network.trainable_params(), lr, momentum)

    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define the model
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    epoch_size = 1
    mnist_path = "./datasets/MNIST_Data"

    train_dataset = create_dataset("./datasets/MNIST_Data/train")
    eval_dataset = create_dataset("./datasets/MNIST_Data/test")

    print("========== The Training Model is Defined. ==========")

    # train the model and export the encrypted CheckPoint file through Callback
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10, enc_key=b'0123456789ABCDEF',
                                 enc_mode='AES-GCM')
    ckpoint_cb = ModelCheckpoint(prefix='lenet_enc', directory=None, config=config_ck)
    model.train(10, train_dataset, dataset_sink_mode=False, callbacks=[ckpoint_cb, LossMonitor(1875)])
    acc = model.eval(eval_dataset, dataset_sink_mode=False)
    print("Accuracy: {}".format(acc["Accuracy"]))

    # export the encrypted CheckPoint file through save_checkpoint
    save_checkpoint(network, 'lenet_enc.ckpt', enc_key=b'0123456789ABCDEF', enc_mode='AES-GCM')

    # load encrypted CheckPoint file and eval
    param_dict = load_checkpoint('lenet_enc-10_1875.ckpt', dec_key=b'0123456789ABCDEF', dec_mode='AES-GCM')
    load_param_into_net(network, param_dict)
    acc = model.eval(eval_dataset, dataset_sink_mode=False)
    print("Accuracy loading encrypted CheckPoint: {}".format(acc["Accuracy"]))
