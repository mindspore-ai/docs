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
"""use metrics
This sample code is applicable to Ascend, CPU and GPU.
"""
import mindspore.nn as nn
from mindspore.nn import Momentum, SoftmaxCrossEntropyWithLogits
from mindspore import Model, set_context, GRAPH_MODE
from mindspore import ModelCheckpoint, CheckpointConfig, LossMonitor

from src.dataset import create_train_dataset, create_eval_dataset
from src.net import Net


if __name__ == "__main__":
    set_context(mode=GRAPH_MODE)
    ds_train = create_train_dataset()
    ds_eval = create_eval_dataset()
    net = Net()
    net_opt = Momentum(net.trainable_params(), 0.01, 0.9)
    net_loss = SoftmaxCrossEntropyWithLogits(reduction='mean')
    metrics = {
        'Accuracy': nn.Accuracy(),
        'Loss': nn.Loss(),
        'Precision': nn.Precision(),
        'Recall': nn.Recall(),
        'F1_score': nn.F1()
    }
    config_ck = CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=10)
    ckpoint = ModelCheckpoint(prefix="CKPT", config=config_ck)
    model = Model(network=net, loss_fn=net_loss, optimizer=net_opt, metrics=metrics)
    model.train(epoch=2, train_dataset=ds_train, callbacks=[ckpoint, LossMonitor()])
    result = model.eval(ds_eval)
    print(result)
