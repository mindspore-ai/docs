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
"""MNIST."""
import os
import numpy as np
from download import download

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

def create_dataset(with_preprocess=True, need_download=True):
    '''
    Download, load and preprocess MNIST dataset.
    '''
    if need_download:
        # Download the opensource dataset, MNIST.
        mnist_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip"
        download(mnist_url, "./", kind="zip", replace=True)

    if not os.path.exists("MNIST_Data/train"):
        raise RuntimeError("MNIST dataset file was ruined, set download to True for a new one.")

    # Load MNIST dataset
    mnist_dataset = ds.MnistDataset("MNIST_Data/train")

    if with_preprocess:
        resize_height, resize_width = 32, 32
        rescale = 1.0 / 255.0
        rescale_nml = 1 / 0.3081
        shift_nml = -1 * 0.1307 / 0.3081

        # define map operations
        resize_op = vision.Resize((resize_height, resize_width))
        rescale_nml_op = vision.Rescale(rescale_nml * rescale, shift_nml)
        type_cast_op = transforms.TypeCast(np.int32)
        hwc2chw_op = vision.HWC2CHW()

        # apply transforms on images
        mnist_dataset = mnist_dataset.map(operations=type_cast_op, input_columns="label")
        mnist_dataset = mnist_dataset.map(operations=resize_op, input_columns="image")
        mnist_dataset = mnist_dataset.map(operations=rescale_nml_op, input_columns="image")
        mnist_dataset = mnist_dataset.map(operations=hwc2chw_op, input_columns="image")

        # batch operation
        mnist_dataset = mnist_dataset.batch(128, drop_remainder=True)

    return mnist_dataset
