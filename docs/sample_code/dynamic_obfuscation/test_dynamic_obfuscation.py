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
"""Test dynamic obfuscation"""
import os
import numpy as np
import mindspore as ms

import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal

ms.context.set_context(mode=ms.context.GRAPH_MODE)


def weight_variable():
    return TruncatedNormal(0.02)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias, has_bias=False)


class ObfuscateNet(nn.Cell):
    """ construct network for obfuscation """
    def __init__(self):
        super(ObfuscateNet, self).__init__()
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.matmul = ops.MatMul()
        self.matmul_weight1 = ms.Tensor(np.random.random((16 * 5 * 5, 120)).astype(np.float32))
        self.matmul_weight2 = ms.Tensor(np.random.random((120, 84)).astype(np.float32))
        self.matmul_weight3 = ms.Tensor(np.random.random((84, 10)).astype(np.float32))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        """ construct function """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.matmul(x, self.matmul_weight1)
        x = self.relu(x)
        x = self.matmul(x, self.matmul_weight2)
        x = self.relu(x)
        x = self.matmul(x, self.matmul_weight3)
        return x


def test_obfuscate_model_password_mode():
    """
    Feature: Obfuscate MindIR format model with dynamic obfuscation (password mode).
    Description: Test obfuscate a MindIR format model and then load it for prediction.
    Expectation: Success.
    """
    net = ObfuscateNet()
    input_tensor = ms.Tensor(np.ones((1, 1, 32, 32)).astype(np.float32))
    ms.export(net, input_tensor, file_name="net", file_format="MINDIR")
    original_result = net(input_tensor).asnumpy()

    # obfuscate model
    obf_config = {"original_model_path": "net.mindir", "save_model_path": "./obf_net",
                  "model_inputs": [input_tensor], "obf_ratio": 0.8, "obf_password": 3423}
    ms.obfuscate_model(obf_config)

    # load obfuscated model, predict with right password
    obf_graph = ms.load("obf_net.mindir")
    obf_net = nn.GraphCell(obf_graph, obf_password=3423)
    right_password_result = obf_net(input_tensor).asnumpy()

    # load obfuscated model, predict with wrong password
    obf_graph = ms.load("obf_net.mindir")
    obf_net = nn.GraphCell(obf_graph, obf_password=5344)
    wrong_password_result = obf_net(input_tensor).asnumpy()

    os.remove("net.mindir")
    os.remove("obf_net.mindir")

    assert np.all(original_result == right_password_result)
    assert np.any(original_result != wrong_password_result)


def test_obfuscate_model_customized_func_mode():
    """
    Feature: Obfuscate MindIR format model with dynamic obfuscation (cusomized_func mode).
    Description: Test obfuscate a MindIR format model and then load it for prediction.
    Expectation: Success.
    """
    net = ObfuscateNet()
    input_tensor = ms.Tensor(np.ones((1, 1, 32, 32)).astype(np.float32))
    ms.export(net, input_tensor, file_name="net", file_format="MINDIR")
    original_result = net(input_tensor).asnumpy()

    # obfuscate model
    def my_func(x1, x2):
        if x1 + x2 > 1000000000:
            return True
        return False

    obf_config = {"original_model_path": "net.mindir", "save_model_path": "./obf_net",
                  "model_inputs": [input_tensor], "obf_ratio": 0.8, "customized_func": my_func}
    ms.obfuscate_model(obf_config)

    # load obfuscated model, predict with right customized function
    obf_graph = ms.load("obf_net.mindir", obf_func=my_func)
    obf_net = nn.GraphCell(obf_graph)
    right_func_result = obf_net(input_tensor).asnumpy()

    # load obfuscated model, predict with wrong customized function
    def wrong_func(x1, x2):
        if x1 + x2 > 1000000000:
            return False
        return True

    obf_graph = ms.load("obf_net.mindir", obf_func=wrong_func)
    obf_net = nn.GraphCell(obf_graph)
    wrong_func_result = obf_net(input_tensor).asnumpy()

    os.remove("net.mindir")
    os.remove("obf_net.mindir")

    assert np.all(original_result == right_func_result)
    assert np.any(original_result != wrong_func_result)


def test_export_password_mode():
    """
    Feature: Obfuscate MindIR format model with dynamic obfuscation (password mode) in ms.export().
    Description: Test obfuscate a MindIR format model and then load it for prediction.
    Expectation: Success.
    """
    net = ObfuscateNet()
    input_tensor = ms.Tensor(np.ones((1, 1, 32, 32)).astype(np.float32))
    ms.export(net, input_tensor, file_name="net", file_format="MINDIR")
    original_result = net(input_tensor).asnumpy()

    # obfuscate model
    obf_config = {"obf_ratio": 0.8, "obf_password": 3423}
    ms.export(net, input_tensor, file_name="obf_net", file_format="MINDIR", obf_config=obf_config)

    # load obfuscated model, predict with right password
    obf_graph = ms.load("obf_net.mindir")
    obf_net = nn.GraphCell(obf_graph, obf_password=3423)
    right_password_result = obf_net(input_tensor).asnumpy()

    # load obfuscated model, predict with wrong password
    obf_graph = ms.load("obf_net.mindir")
    obf_net = nn.GraphCell(obf_graph, obf_password=5344)
    wrong_password_result = obf_net(input_tensor).asnumpy()

    os.remove("net.mindir")
    os.remove("obf_net.mindir")

    assert np.all(original_result == right_password_result)
    assert np.any(original_result != wrong_password_result)


def test_export_customized_func_mode():
    """
    Feature: Obfuscate MindIR format model with dynamic obfuscation (customized_func mode) in ms.export().
    Description: Test obfuscate a MindIR format model and then load it for prediction.
    Expectation: Success.
    """
    net = ObfuscateNet()
    input_tensor = ms.Tensor(np.ones((1, 1, 32, 32)).astype(np.float32))
    ms.export(net, input_tensor, file_name="net", file_format="MINDIR")
    original_result = net(input_tensor).asnumpy()

    # obfuscate model
    def my_func(x1, x2):
        if x1 + x2 > 1000000000:
            return True
        return False

    obf_config = {"obf_ratio": 0.8, "customized_func": my_func}
    ms.export(net, input_tensor, file_name="obf_net", file_format="MINDIR", obf_config=obf_config)

    # load obfuscated model, predict with customized function
    obf_graph = ms.load("obf_net.mindir", obf_func=my_func)
    obf_net = nn.GraphCell(obf_graph)
    right_func_result = obf_net(input_tensor).asnumpy()

    # load obfuscated model, predict with customized function
    def wrong_func(x1, x2):
        if x1 + x2 > 1000000000:
            return False
        return True

    obf_graph = ms.load("obf_net.mindir", obf_func=wrong_func)
    obf_net = nn.GraphCell(obf_graph)
    wrong_func_result = obf_net(input_tensor).asnumpy()

    os.remove("net.mindir")
    os.remove("obf_net.mindir")

    assert np.all(original_result == right_func_result)
    assert np.any(original_result != wrong_func_result)
