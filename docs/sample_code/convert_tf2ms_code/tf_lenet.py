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
"""tensorflow network"""
import os
import numpy as np
import tensorflow.compat.v1 as tf


def get_variable(name, shape=None, dtpye=tf.float32, initializer=tf.ones_initializer()):
    """initializer"""
    return tf.get_variable(name, shape, dtpye, initializer)


def lenet(inputs):
    """lenet model definition"""
    with tf.variable_scope('conv1'):
        net = tf.nn.conv2d(input=inputs, filter=get_variable('weight', [5, 5, 1, 6]),
                           strides=[1, 1, 1, 1], padding='VALID')
        net = tf.nn.relu(net)

    with tf.variable_scope('pool1'):
        net = tf.nn.max_pool2d(input=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('conv2'):
        net = tf.nn.conv2d(input=net, filter=get_variable('weight', [5, 5, 6, 16]),
                           strides=[1, 1, 1, 1], padding='VALID')
        net = tf.nn.relu(net)

    with tf.variable_scope('pool2'):
        net = tf.nn.max_pool2d(input=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('fc1'):
        size = 5 * 5 * 16
        net = tf.reshape(net, shape=[-1, size])
        net = tf.layers.dense(inputs=net, units=120)
        net = tf.nn.relu(net)

    with tf.variable_scope('fc2'):
        net = tf.layers.dense(inputs=net, units=84)
        net = tf.nn.relu(net)

    with tf.variable_scope('fc3'):
        net = tf.layers.dense(inputs=net, units=1)

    return net


def tf_running(model_savel_path):
    """tensorflow running"""
    np_in = np.ones([8, 32, 32, 1]).astype(np.float32)
    inputs = tf.convert_to_tensor(np_in)

    tf_network = lenet(inputs)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        res = sess.run(tf_network)
        saver.save(sess, os.path.join(model_savel_path, 'lenet'))
        tf_outputs = res
    return tf_outputs

if __name__ == '__main__':
    model_dir = './'
    outs = tf_running(model_dir)
