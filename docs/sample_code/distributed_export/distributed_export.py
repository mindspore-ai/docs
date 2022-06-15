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
"""
Distributed inference
"""
import numpy as np
from net import Net
import mindspore as ms
from mindspore.communication import init


def test_export():
    """distributed inference after distributed training"""
    ms.set_context(mode=ms.GRAPH_MODE)
    init(backend_name="hccl")
    ms.set_auto_parallel_context(full_batch=True, parallel_mode="semi_auto_parallel",
                                 strategy_ckpt_load_file="./train_strategy.ckpt", device_num=8)

    predict_data = create_predict_data()
    network = Net(matmul_size=(96, 16))
    model = ms.Model(network)
    predict_layout = model.infer_predict_layout(ms.Tensor(predict_data))
    ckpt_file_list = create_ckpt_file_list()
    ms.load_distributed_checkpoint(network, ckpt_file_list, predict_layout)
    ms.export(network, ms.Tensor(predict_data), file_name='net', file_format='MINDIR')


def create_predict_data():
    """user-defined predict data"""
    inputs_np = np.random.randn(128, 96).astype(np.float32)
    return ms.Tensor(inputs_np)


def create_ckpt_file_list():
    """user-defined ckpt file list"""
    ckpt_file_list = []
    for i in range(8):
        path = "../device" + str(i) + "/" + "rank_" + str(i) + "_ckpt/" + "parallel-2_2.ckpt"
        ckpt_file_list.append(path)
    return ckpt_file_list
