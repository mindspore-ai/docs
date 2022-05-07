# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import os
import argparse
from mindspore import set_seed, set_context, GRAPH_MODE
from mindspore import Model
from mindspore import load_checkpoint, load_param_into_net

from src.config import config
from src.dataset import create_dataset
from src.resnet import resnet50
from src.cross_entropy_smooth import CrossEntropySmooth

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
args_opt = parser.parse_args()

set_seed(1)



if __name__ == '__main__':
    device_id = int(os.getenv('DEVICE_ID', '0'))
    # init context
    set_context(mode=GRAPH_MODE, device_target='Ascend', device_id=device_id)

    # create dataset
    dataset = create_dataset(args_opt.dataset_path, config.batch_size, do_train=False)
    step_size = dataset.get_dataset_size()

    # define net
    net = resnet50(class_num=config.class_num)

    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model
    loss = CrossEntropySmooth(sparse=True, reduction='mean', smooth_factor=config.label_smooth_factor,
                              num_classes=config.class_num)

    # define model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
