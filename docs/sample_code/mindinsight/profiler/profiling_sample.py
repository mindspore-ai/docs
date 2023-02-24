# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Profiler functional programming use case training."""
import os
import requests
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
from mindspore import Profiler

requests.packages.urllib3.disable_warnings()


def download_dataset(dataset_url, path):
    """Download Dataset."""
    filename = dataset_url.split("/")[-1]
    save_path = os.path.join(path, filename)
    if os.path.exists(save_path):
        return
    if not os.path.exists(path):
        os.makedirs(path)
    res = requests.get(dataset_url, stream=True, verify=False)
    with open(save_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=512):
            if chunk:
                f.write(chunk)
    print("The {} file is downloaded and saved in the path {} after processing".format(os.path.basename(dataset_url),
                                                                                       path))


def datapipe(path, batch_size):
    """Dataset pipeline."""
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(ms.int32)

    dataset = MnistDataset(path)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset


class Network(nn.Cell):
    """Define model."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28 * 28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        """Construct."""
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits


def train_loop(dataset):
    """Train the net."""
    size = dataset.get_dataset_size()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)
        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
        if batch > 900:
            break


if __name__ == "__main__":
    # Download data from open datasets
    ds_train_path = "./datasets/MNIST_Data/train/"
    download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-labels-idx1-ubyte",
                     ds_train_path)
    download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-images-idx3-ubyte",
                     ds_train_path)

    ms.set_context(device_target="Ascend")
    profiler = Profiler(output_path="profiler_data")

    epochs, learning_rate = 2, 1e-2
    train_dataset = datapipe(ds_train_path, 64)
    model = Network()
    model.set_train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.SGD(model.trainable_params(), learning_rate=learning_rate)


    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits


    # Get gradient function
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)


    @ms.jit
    def train_step(data, label):
        """Define function of one-step training"""
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss


    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataset)

    profiler.analyse()
