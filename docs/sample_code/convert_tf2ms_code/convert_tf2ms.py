# Copyright 2023 Huawei Technologies Co., Ltd
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
"""convert tensorflow checkpoint to mindspore checkpoint"""
import os
import numpy as np
from mindspore import Tensor
from mindspore import save_checkpoint
import tensorflow.compat.v1 as tf
from ms_lenet import LeNet5, mindspore_running
from tf_lenet import tf_running
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

params_mapping = {
    "conv1/weight": "conv1.weight",
    "conv2/weight": "conv2.weight",
    "fc1/dense/kernel": "fc1.weight",
    "fc1/dense/bias": "fc1.bias",
    "fc2/dense/kernel": "fc2.weight",
    "fc2/dense/bias": "fc2.bias",
    "fc3/dense/kernel": "fc3.weight",
    "fc3/dense/bias": "fc3.bias",
}


def tensorflow_param(ckpt_path):
    """Get TensorFlow parameter and shape"""
    tf_param = {}
    reader = tf.train.load_checkpoint(ckpt_path)
    for name in reader.get_variable_to_shape_map():
        try:
            print(name, reader.get_tensor(name).shape)
            tf_param[name] = reader.get_tensor(name)
        except AttributeError as e:
            print(e)
    return tf_param


def mindspore_params(net):
    """Get MindSpore parameter and shape"""
    ms_param = {}
    for param in net.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape)
        ms_param[name] = value
    return ms_param


def tensorflow2mindspore(tf_ckpt_dir, param_mapping_dict, ms_ckpt_path):
    """convert tensorflow ckpt to mindspore ckpt"""
    reader = tf.train.load_checkpoint(tf_ckpt_dir)
    new_params_list = []
    for name in param_mapping_dict:
        param_dict = {}
        parameter = reader.get_tensor(name)
        if 'conv' in name and 'weight' in name:
            # 对卷积权重进行转置
            parameter = np.transpose(parameter, axes=[3, 2, 0, 1])
        if 'fc' in name and 'kernel' in name:
            parameter = np.transpose(parameter, axes=[1, 0])
        param_dict['name'] = param_mapping_dict[name]
        param_dict['data'] = Tensor(parameter)
        new_params_list.append(param_dict)
    save_checkpoint(new_params_list, os.path.join(ms_ckpt_path, 'tf2mindspore.ckpt'))


def mean_relative_error(y_expect, y_pred):
    """mean relative error"""
    if y_expect.dtype == np.bool:
        y_expect = y_expect.astype(np.int32)
        y_pred = y_pred.astype(np.int32)

    rerror = np.abs(y_expect - y_pred)/np.maximum(np.abs(y_expect), np.abs(y_pred))
    rerror = rerror[~np.isnan(rerror)]
    rerror = rerror[~np.isinf(rerror)]
    relative_error_out = np.mean(rerror)
    return relative_error_out


if __name__ == '__main__':
    tf_model_path = './model'
    tf_outputs = tf_running(tf_model_path)
    tf_params = tensorflow_param(tf_model_path)
    print("*" * 30)
    network = LeNet5()
    ms_params = mindspore_params(network)
    tensorflow2mindspore(tf_model_path, params_mapping, './')
    ms_outputs = mindspore_running('./tf2mindspore.ckpt')
    print("************tensorflow outputs**************")
    print(tf_outputs)
    print("************mindspore outputs**************")
    print(ms_outputs)
    relative_error = mean_relative_error(tf_outputs, ms_outputs)
    print("Diff: ", relative_error)
