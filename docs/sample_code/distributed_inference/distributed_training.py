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
"""
Distributed training
"""
import os
from dataset import FakeData
from net import Net
from mindspore import Model, ParallelMode, set_context, GRAPH_MODE, set_auto_parallel_context, \
    reset_auto_parallel_context
from mindspore import CheckpointConfig, ModelCheckpoint
from mindspore.nn import Momentum, SoftmaxCrossEntropyWithLogits


def test_train():
    """distributed training"""
    set_context(mode=GRAPH_MODE)
    parallel_dataset = FakeData()
    strategy = ((2, 1), (1, 4))
    set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                              device_num=8,
                              strategy_ckpt_save_file="./train_strategy.ckpt")
    network = Net(matmul_size=(96, 16), strategy=strategy)
    net_opt = Momentum(network.trainable_params(), 0.01, 0.9)
    net_loss = SoftmaxCrossEntropyWithLogits(reduction='mean')
    model = Model(network=network, loss_fn=net_loss, optimizer=net_opt)
    ckpt_config = CheckpointConfig(keep_checkpoint_max=1, integrated_save=False)
    global_rank_id = int(os.getenv("RANK_ID"))
    ckpt_path = './rank_{}_ckpt'.format(global_rank_id)
    ckpt_callback = ModelCheckpoint(prefix='parallel', directory=ckpt_path, config=ckpt_config)
    model.train(epoch=2, train_dataset=parallel_dataset, callbacks=[ckpt_callback], dataset_sink_mode=False)
    reset_auto_parallel_context()
