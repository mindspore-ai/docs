# Copyright 2020 Huawei Technologies Co., Ltd
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
# pylint: disable=C0301
"""Evaluate the model during training tutorial
This sample code is applicable to CPU, GPU and Ascend.
"""
import os
import requests
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype
from mindspore import nn, Model, context
from mindspore.common.initializer import Normal
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, Callback
from mindspore.nn import Accuracy
from mindspore.nn import SoftmaxCrossEntropyWithLogits

requests.packages.urllib3.disable_warnings()

def create_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=1):
    """ create dataset for train or test
    Args:
        data_path (str): Data path
        batch_size (int): The number of data records in each group
        repeat_size (int): The number of replicated data records
        num_parallel_workers (int): The number of parallel workers
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    # define operation parameters
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    type_cast_op = C.TypeCast(mstype.int32)
    c_trans = [
        CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR),
        CV.Rescale(rescale_nml, shift_nml),
        CV.Rescale(rescale, shift),
        CV.HWC2CHW()
    ]

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=c_trans, input_columns="image", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
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


class EvalCallBack(Callback):
    """Precision verification using callback function."""
    # define the operator required
    def __init__(self, models, eval_dataset, eval_per_epochs, epochs_per_eval):
        super(EvalCallBack, self).__init__()
        self.models = models
        self.eval_dataset = eval_dataset
        self.eval_per_epochs = eval_per_epochs
        self.epochs_per_eval = epochs_per_eval

    # define operator function in epoch end
    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epochs == 0:
            acc = self.models.eval(self.eval_dataset, dataset_sink_mode=False)
            self.epochs_per_eval["epoch"].append(cur_epoch)
            self.epochs_per_eval["acc"].append(acc["Accuracy"])
            print(acc)



def download_dataset(dataset_url, path):
    """download dataset"""
    filename = dataset_url.split("/")[-1]
    save_path = os.path.join(path, filename)
    if os.path.exists(save_path):
        return
    os.makedirs(path, exist_ok=True)
    res = requests.get(dataset_url, stream=True, verify=False)
    with open(save_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=512):
            if chunk:
                f.write(chunk)
    print("The {} file is downloaded and saved in the path {} after processing".format(os.path.basename(dataset_url), path))

if __name__ == "__main__":
    # set args, train it
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    train_data_path = "./datasets/MNIST_Data/train"
    eval_data_path = "./datasets/MNIST_Data/test"
    download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-labels-idx1-ubyte", train_data_path)
    download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-images-idx3-ubyte", train_data_path)
    download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-labels-idx1-ubyte", eval_data_path)
    download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-images-idx3-ubyte", eval_data_path)

    ckpt_save_dir = "./lenet_ckpt"
    epoch_size = 10
    eval_per_epoch = 2
    repeat = 1
    train_data = create_dataset(train_data_path, repeat_size=repeat)
    eval_data = create_dataset(eval_data_path, repeat_size=repeat)
    # define the net
    network = LeNet5()
    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # define the optimizer
    net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
    config_ck = CheckpointConfig(save_checkpoint_steps=eval_per_epoch*1875, keep_checkpoint_max=15)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", directory=ckpt_save_dir, config=config_ck)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    epoch_per_eval = {"epoch": [], "acc": []}
    eval_cb = EvalCallBack(model, eval_data, eval_per_epoch, epoch_per_eval)
    model.train(epoch_size, train_data, callbacks=[ckpoint_cb, LossMonitor(375), eval_cb], dataset_sink_mode=False)
