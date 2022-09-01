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
"""Resnet train script."""

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.profiler import Profiler
from mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from src.dataset import create_dataset
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config
from src.utils import init_env
from src.resnet import resnet50


def train_epoch(epoch, model, data_loader):
    """Single train one epoch"""
    model.set_train()
    dataset_size = data_loader.get_dataset_size()
    for batch_idx, (data, target) in enumerate(data_loader):
        loss = float(model(data, target)[0].asnumpy())
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, dataset_size,
                100. * batch_idx / dataset_size, loss))


def test_epoch(model, data_loader, loss_func):
    """Evaluation once"""
    model.set_train(False)
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        output = model(data)
        test_loss += float(loss_func(output, target).asnumpy())
        pred = np.argmax(output.asnumpy(), axis=1)
        correct += (pred == target.asnumpy()).sum()
    dataset_size = data_loader.get_dataset_size()
    test_loss /= dataset_size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / dataset_size))


@moxing_wrapper()
def train_net():
    """Training processing"""
    init_env(config)
    device_num = config.device_num
    if config.use_profilor:
        profiler = Profiler()
        device_num = 40
    train_dataset = create_dataset(config.dataset_name, config.data_path, True, batch_size=config.batch_size,
                                   image_size=(int(config.image_height), int(config.image_width)),
                                   rank_size=device_num, rank_id=config.rank_id)
    eval_dataset = create_dataset(config.dataset_name, config.data_path, False, batch_size=1,
                                  image_size=(int(config.image_height), int(config.image_width)))
    config.steps_per_epoch = train_dataset.get_dataset_size()
    resnet = resnet50(num_classes=config.class_num)
    optimizer = nn.Adam(resnet.trainable_params(), config.lr, weight_decay=config.weight_decay)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_scale = ms.FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    model = ms.Model(resnet, loss_fn=loss, optimizer=optimizer, loss_scale_manager=loss_scale,
                     amp_level=config.amp_level)
    if config.use_profilor:
        model.train(3, train_dataset, callbacks=[LossMonitor(), TimeMonitor()],
                    dataset_sink_mode=config.dataset_sink_mode)
        profiler.analyse()
    else:
        config_ck = CheckpointConfig(save_checkpoint_steps=train_dataset.get_dataset_size(),
                                     keep_checkpoint_max=5)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory="./checkpoint", config=config_ck)
        model.train(config.epoch_size, train_dataset, eval_dataset, callbacks=[LossMonitor(), TimeMonitor(), ckpt_cb],
                    dataset_sink_mode=config.dataset_sink_mode)
    # train_net = nn.TrainOneStepWithLossScaleCell(
    #     nn.WithLossCell(resnet, loss), optimizer, ms.Tensor(config.loss_scale, ms.float32))
    # for epoch in range(config.epoch_size):
    #     train_epoch(epoch, train_net, train_dataset)
    #     test_epoch(resnet, eval_dataset, loss)
    #
    # print('Finished Training')
    # save_path = './resnet.ckpt'
    # ms.save_checkpoint(resnet, save_path)


if __name__ == '__main__':
    train_net()
