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

"""Save Model Tutorial
This sample code is applicable to CPU, GPU and Ascend.
"""
import os
import mindspore.nn as nn
from mindspore.nn import SoftmaxCrossEntropyWithLogits, Accuracy
import mindspore as ms
from src.lenet import LeNet5
from src.datasets import create_dataset
import numpy as np

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")


def output_file_formats(ckpt_path, net_work, batch_size, output_file_name, output_format):
    ms.load_checkpoint(ckpt_path, net=net_work)
    input_data = np.random.uniform(0.0, 1.0, size=batch_size).astype(np.float32)
    ms.export(net_work, ms.Tensor(input_data), file_name=output_file_name, file_format=output_format)


if __name__ == "__main__":
    lr = 0.01
    momentum = 0.9
    epoch_size = 1
    model_path = "./models/ckpt/save_model/"
    model_file = model_path + "checkpoint_lenet-1_1875.ckpt"

    os.system('rm -f {0}*.ckpt {0}*.meta {0}*.pb'.format(model_path))

    network = LeNet5()
    net_opt = nn.Momentum(network.trainable_params(), lr, momentum)
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model = ms.Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    config_ck = ms.CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=16)
    ckpoint_cb = ms.ModelCheckpoint(prefix="checkpoint_lenet", directory=model_path, config=config_ck)

    train_dataset = create_dataset("./datasets/MNIST_Data/train")
    eval_dataset = create_dataset("./datasets/MNIST_Data/test")
    print("===============start training===============")
    model.train(epoch_size, train_dataset, callbacks=[ckpoint_cb, ms.LossMonitor()], dataset_sink_mode=False)
    print("===============saving models in mindir and onnx formats===============")

    output_file_formats(model_file, network, [32, 1, 32, 32], "checkpoint_lenet", "MINDIR")
    output_file_formats(model_file, network, [32, 1, 32, 32], "checkpoint_lenet", "ONNX")
