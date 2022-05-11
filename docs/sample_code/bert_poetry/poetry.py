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

'''Bert finetune script
This sample code is applicable to Ascend.
'''
import os
import re
import time
import argparse
from src.utils import BertPoetry, BertPoetryCell, BertLearningRate, BertPoetryModel
from src.finetune_config import cfg, bert_net_cfg
from src.poetry_dataset import create_poetry_dataset, create_tokenizer
from mindspore import load_checkpoint, load_param_into_net, GRAPH_MODE, set_context
from mindspore.nn import DynamicLossScaleUpdateCell
from mindspore.nn import AdamWeightDecay
from mindspore import Model
from mindspore import Callback, TimeMonitor
from mindspore import CheckpointConfig, ModelCheckpoint
from mindspore import Tensor, Parameter, export
from mindspore import dtype as mstype
from generator import generate_random_poetry, generate_hidden
import numpy as np

class LossCallBack(Callback):
    '''
    Monitor the loss in training.
    If the loss is NAN or INF, terminate training.
    Note:
        If per_print_times is 0, do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    '''
    def __init__(self, model, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be in and >= 0.")
        self._per_print_times = per_print_times
        self.model = model

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        with open("./loss.log", "a+") as f:
            f.write("epoch: {}, step: {}, loss: {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                           cb_params.net_outputs[0]))
            f.write("\n")
        print("epoch: {}, step: {}, loss: {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                     cb_params.net_outputs[0]))


def test_train():
    '''
    finetune function
    '''
    target = args_opt.device_target
    if target == "Ascend":
        try:
            devid = int(os.getenv('DEVICE_ID'))
        except TypeError:
            devid = 0
        set_context(mode=GRAPH_MODE, device_target="Ascend", device_id=devid)

    poetry, tokenizer, keep_words = create_tokenizer()
    print("total vocab_size after filtering is ", len(keep_words))

    dataset = create_poetry_dataset(bert_net_cfg.batch_size, poetry, tokenizer)

    num_tokens = 3191
    poetrymodel = BertPoetryModel(bert_net_cfg, True, num_tokens, dropout_prob=0.1)
    netwithloss = BertPoetry(poetrymodel, bert_net_cfg, True, dropout_prob=0.1)
    callback = LossCallBack(poetrymodel)

    # optimizer
    steps_per_epoch = dataset.get_dataset_size()
    print("============ steps_per_epoch is {}".format(steps_per_epoch))
    lr_schedule = BertLearningRate(learning_rate=cfg.AdamWeightDecay.learning_rate,
                                   end_learning_rate=cfg.AdamWeightDecay.end_learning_rate,
                                   warmup_steps=1000,
                                   decay_steps=cfg.epoch_num*steps_per_epoch,
                                   power=cfg.AdamWeightDecay.power)
    optimizer = AdamWeightDecay(netwithloss.trainable_params(), lr_schedule)
    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix=cfg.ckpt_prefix, directory=cfg.ckpt_dir, config=ckpt_config)
    time_cb = TimeMonitor(dataset.get_dataset_size())

    param_dict = load_checkpoint(cfg.pre_training_ckpt)
    new_dict = {}



    # load corresponding rows of embedding_lookup
    for key in param_dict:
        if "bert_embedding_lookup" not in key:
            new_dict[key] = param_dict[key]
        else:
            value = param_dict[key]
            np_value = value.data.asnumpy()
            np_value = np_value[keep_words]
            tensor_value = Tensor(np_value, mstype.float32)
            parameter_value = Parameter(tensor_value, name=key)
            new_dict[key] = parameter_value

    load_param_into_net(netwithloss, new_dict)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
    netwithgrads = BertPoetryCell(netwithloss, optimizer=optimizer, scale_update_cell=update_cell)

    model = Model(netwithgrads)
    model.train(cfg.epoch_num, dataset, callbacks=[callback, ckpoint_cb, time_cb], dataset_sink_mode=True)

def test_eval(model_ckpt_path):
    '''eval model'''
    target = args_opt.device_target
    if target == "Ascend":
        try:
            devid = int(os.getenv('DEVICE_ID'))
        except TypeError:
            devid = 0
        set_context(mode=GRAPH_MODE, device_target="Ascend", device_id=devid)
    bert_net_cfg.batch_size = 1
    poetrymodel = BertPoetryModel(bert_net_cfg, False, 3191, dropout_prob=0.0)
    poetrymodel.set_train(False)
    param_dict = load_checkpoint(model_ckpt_path)
    load_param_into_net(poetrymodel, param_dict)

    # random generation/continue
    start_time = time.time()
    output = generate_random_poetry(poetrymodel, s='')
    end_to_end_delay = (time.time()-start_time)*1000
    a = re.findall(r'[\u4e00-\u9fa5]*[\uff0c\u3002]', output)

    print("\n**********************************")
    print("随机生成: \n")
    for poem in a:
        print(poem)
    print("\ncost time: {:.1f} ms".format(end_to_end_delay))
    print("\n")

    start = "天下为公"
    start_time = time.time()
    output = generate_random_poetry(poetrymodel, s=start)
    end_to_end_delay = (time.time()-start_time)*1000
    a = re.findall(r'[\u4e00-\u9fa5]*[\uff0c\u3002]', output)

    print("\n**********************************")
    print("续写 【{}】: \n".format(start))
    for poem in a:
        print(poem)
    print("\ncost time: {:.1f} ms".format(end_to_end_delay))
    print("\n")



    # hidden poetry
    s = "人工智能"
    start_time = time.time()
    output = generate_hidden(poetrymodel, head=s)
    if "'" in output:
        print(output)
    else:
        end_to_end_delay = (time.time()-start_time)*1000
        a = re.findall(r'[\u4e00-\u9fa5]*[\uff0c\u3002]', output)
        print("\n**********************************")
        print("藏头诗 【{}】: \n".format(s))
        for poem in a:
            print(poem)
        print("\ncost time: {:.1f} ms".format(end_to_end_delay))
        print("\n")


def export_net(model_ckpt_path):
    bert_net_cfg.batch_size = 1
    poetrymodel = BertPoetryModel(bert_net_cfg, False, 3191, dropout_prob=0.0)
    poetrymodel.set_train(False)
    param_dict = load_checkpoint(model_ckpt_path)
    load_param_into_net(poetrymodel, param_dict)
    input_id = np.ones(shape=(1, 128))
    token_type_id = np.ones(shape=(1, 128))
    pad_mask = np.ones(shape=(1, 128))
    export(poetrymodel, Tensor(input_id, mstype.int32),\
            Tensor(token_type_id, mstype.int32),\
            Tensor(pad_mask, mstype.float32),\
            file_name='./serving/bert/1/poetry', file_format='MINDIR')

parser = argparse.ArgumentParser(description='Bert finetune')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--train', type=str, default="True", help='train or inference')
parser.add_argument('--ckpt_path', type=str, help='path of your ckpt')
parser.add_argument('--export', type=str, default="False", help="whether export MINDIF")
args_opt = parser.parse_args()
if __name__ == "__main__":
    if args_opt.export in ["true", "True", "TRUE"]:
        ckpt_path = args_opt.ckpt_path
        export_net(ckpt_path)
        exit()

    if args_opt.train in ["true", "True", "TRUE"]:
        test_train()
    else:
        ckpt_path = args_opt.ckpt_path
        test_eval(ckpt_path)
