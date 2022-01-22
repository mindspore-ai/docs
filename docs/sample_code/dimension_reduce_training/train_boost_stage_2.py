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
"""train resnet."""
import argparse
import os

from mindspore import context
from mindspore.nn import SGD
from mindspore import Model
from mindspore import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication import init
from mindspore import set_seed
from mindspore.parallel import set_algo_parameters
from mindspore import load_checkpoint, load_param_into_net

from models.official.cv.resnet.src.CrossEntropySmooth import CrossEntropySmooth
from models.official.cv.resnet.src.dataset import create_dataset2 as create_dataset
from models.official.cv.resnet.src.model_utils.local_adapter import get_rank_id

from resnet import resnet50 as resnet


set_seed(1)


def init_group_params(network):
    """init_group_params"""
    decayed_params = []
    no_decayed_params = []
    for param in network.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_parameters = [{'params': decayed_params, 'weight_decay': 0.0001},
                        {'params': no_decayed_params},
                        {'order_params': network.trainable_params()}]
    return group_parameters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MindSpore Dimension Reduce Training Example Stage 2')
    parser.add_argument('--data_path', type=str, default="./data", help='path where the dataset is saved')
    parser.add_argument('--pretrained_weight_path', type=str, default="", help='path to load pretrained weight')
    parser.add_argument('--pca_mat_path', type=str, default="", help='path to load pca_mat')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)
    context.set_auto_parallel_context(device_num=8, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    set_algo_parameters(elementwise_op_strategy_follow=True)
    init()

    # define train dataset
    train_data_path = os.path.join(args.data_path, "train")
    ds_train = create_dataset(dataset_path=train_data_path, do_train=True, batch_size=256, train_image_size=224,
                              eval_image_size=224, target="Ascend", distribute=True)
    step_size = ds_train.get_dataset_size()

    # define net
    net = resnet(num_classes=1001)
    if os.path.isfile(args.pretrained_weight_path):
        weight_dict = load_checkpoint(args.pretrained_weight_path)
    load_param_into_net(net, weight_dict)

    # define loss
    loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.1, num_classes=1001)

    # define optimizer
    group_params = init_group_params(net)
    opt = SGD(group_params, learning_rate=1)

    # define metrics
    metrics = {"acc"}

    # define boost config dictionary
    boost_dict = {
        "boost": {
            "mode": "manual",
            "dim_reduce": True
        },
        "common": {
            "device_num": 8
        },
        "dim_reduce": {
            "rho": 0.55,
            "gamma": 0.9,
            "alpha": 0.001,
            "sigma": 0.4,
            "n_component": 32,
            "pca_mat_path": args.pca_mat_path,
            "weight_load_dir": "../device0_stage_1/checkpoint_stage_1/checkpoint_pca",
            "timeout": 1200
        }
    }

    # define model
    model = Model(net, loss_fn=loss, optimizer=opt, metrics=metrics, boost_level="O1", boost_config_dict=boost_dict)

    # define callback
    cb = [TimeMonitor(data_size=step_size), LossMonitor()]
    if get_rank_id() == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=step_size, keep_checkpoint_max=2)
        ck_cb = ModelCheckpoint(prefix="resnet", directory="./checkpoint_stage_2", config=config_ck)
        cb += [ck_cb]

    print("============== Starting Training ==============")
    model.train(2, ds_train, callbacks=cb, sink_size=step_size, dataset_sink_mode=True)

    if get_rank_id() == 0:
        print("============== Starting Testing ==============")
        eval_data_path = os.path.join(args.data_path, "val")
        ds_eval = create_dataset(dataset_path=eval_data_path, do_train=False, batch_size=256, target="Ascend")
        if ds_eval.get_dataset_size() == 0:
            raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

        acc = model.eval(ds_eval)
        print("============== {} ==============".format(acc))
