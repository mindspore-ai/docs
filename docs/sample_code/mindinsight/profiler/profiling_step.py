# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Start profiling on step in callback mode."""
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
from mindspore import Profiler


class Network(nn.Cell):
    """The test net"""
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
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits


def datapipe(dataset, batch_size):
    """Get the dataset."""
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset


def train(epochs, model, dataset, loss_fn, optimizer):
    """Train the net."""
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    size = dataset.get_dataset_size()
    model.set_train()
    step = 0
    profiler = Profiler(start_profile=False, output_path='./data_step')

    for e in range(epochs):
        print(f"Epoch {e + 1}\n-------------------------------")
        for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
            step = step + 1
            step_loss = train_step(data, label)

            if batch % 100 == 0:
                step_loss, current = step_loss.asnumpy(), batch
                print(f"loss: {step_loss:>7f}  [{current:>3d}/{size:>3d}]")

            if step == 10:
                profiler.start()
            if step == 20:
                profiler.stop()
    profiler.analyse()


if __name__ == '__main__':
    net = Network()
    cross_loss = nn.CrossEntropyLoss()
    opt = nn.SGD(net.trainable_params(), 1e-2)
    train_dataset = MnistDataset('/dataset/MNIST_Data/train')
    train_dataset = datapipe(train_dataset, 64)

    train(3, net, train_dataset, cross_loss, opt)

    print("Done!")
