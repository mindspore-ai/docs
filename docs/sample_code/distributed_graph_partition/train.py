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
'''Distributed graph partition'''
import os
import mindspore.context as context
from mindspore.train import Accuracy
from mindspore.train import Model
from mindspore.train import LossMonitor, TimeMonitor
from mindspore.communication import init, get_rank
from lenet import LeNet, get_optimizer, get_loss, create_dataset


context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
init()
net = LeNet()
opt = get_optimizer(net)
criterion = get_loss()
model = Model(net, criterion, opt, metrics={"Accuracy": Accuracy()})

print("================= Start training =================", flush=True)
ds_train = create_dataset(os.path.join(os.getenv("DATA_PATH"), 'train'))
model.train(10, ds_train, callbacks=[LossMonitor(), TimeMonitor()], dataset_sink_mode=False)

print("================= Start testing =================", flush=True)
ds_eval = create_dataset(os.path.join(os.getenv("DATA_PATH"), 'test'))
acc = model.eval(ds_eval, dataset_sink_mode=False)

if get_rank() == 0:
    print("Accuracy is:", acc)
