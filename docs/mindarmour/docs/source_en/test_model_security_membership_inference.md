# Using Membership Inference to Test Model Security

<!-- TOC -->

- [Using Membership Inference to Test Model Security](#using-membership-inference-to-test-model-security)
    - [Overview](#overview)
    - [Implementation](#implementation)
        - [Importing Library Files](#importing-library-files)
            - [Importing Related Packages](#importing-related-packages)
        - [Loading the Dataset](#loading-the-dataset)
        - [Creating the Model](#creating-the-model)
        - [Using Membership Inference for Privacy Security Evaluation](#using-membership-inference-for-privacy-security-evaluation)
    - [References](#references)

<!-- /TOC -->
<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindarmour/docs/source_en/test_model_security_membership_inference.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>&nbsp;&nbsp;

## Overview

Membership inference is a method of inferring user privacy data. Privacy refers to some attributes of a single user. Once the privacy is disclosed, personal injury and reputation damage may occur. Although user privacy data is confidential, it can be inferred by using non-sensitive information. If members of a private club like to wear purple sunglasses and red shoes, then a person who wears purple sunglasses and red shoes (non-sensitive information) may be inferred as a member of this private club (sensitive information). This is membership inference.

In machine learning and deep learning, if an attacker has some access permissions (black box, gray box, or white box) of a model to obtain some or all information about the model output, structure, or parameters, they can determine whether a sample belongs to a training set of a model. In this case, we can use membership inference to evaluate the privacy data security of machine learning and deep learning models. If more than 60% samples can be correctly inferred using membership inference, the model has privacy data leakage risks.

The following uses a VGG16 model and CIFAR-100 dataset as an example to describe how to use membership inference to perform model privacy security evaluation. This tutorial uses pre-trained model parameters for demonstration. This following describes only the model structure, parameter settings, and dataset preprocessing method.

> This example is for the Ascend 910 AI Processor. You can download the complete sample code in the following link:
>
> <https://gitee.com/mindspore/mindarmour/blob/master/examples/privacy/membership_inference/example_vgg_cifar.py>

## Implementation

### Importing Library Files

#### Importing Related Packages

The following contains common modules, MindSpore-related modules, membership inference feature modules, and configuration log labels and log levels.

```python
import argparse
import sys
import math
import os

import numpy as np

import mindspore.nn as nn
from mindspore import Model, load_param_into_net, load_checkpoint
from mindspore import dtype as mstype
from mindspore.common import initializer as init
from mindspore.common.initializer import initializer
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as vision
from mindarmour import MembershipInference
from mindarmour.utils import LogUtil

LOGGER = LogUtil.get_instance()
TAG = "MembershipInference_test"
LOGGER.set_level("INFO")
```

### Loading the Dataset

The CIFAR-100 dataset is used as an example. You can use your own dataset. Ensure that the input data has only two attributes: `image` and `label`.

```python
# Generate CIFAR-100 data.
def vgg_create_dataset100(data_home, image_size, batch_size, rank_id=0, rank_size=1, repeat_num=1,
                          training=True, num_samples=None, shuffle=True):
    """Data operations."""
    de.config.set_seed(1)
    data_dir = os.path.join(data_home, "train")
    if not training:
        data_dir = os.path.join(data_home, "test")

    if num_samples is not None:
        data_set = de.Cifar100Dataset(data_dir, num_shards=rank_size, shard_id=rank_id,
                                      num_samples=num_samples, shuffle=shuffle)
    else:
        data_set = de.Cifar100Dataset(data_dir, num_shards=rank_size, shard_id=rank_id)

    input_columns = ["fine_label"]
    output_columns = ["label"]
    data_set = data_set.rename(input_columns=input_columns, output_columns=output_columns)
    data_set = data_set.project(["image", "label"])

    rescale = 1.0 / 255.0
    shift = 0.0

    # Define map operations.
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT.
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize(image_size)  # interpolation default BILINEAR.
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # Apply map operations on images.
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=c_trans, input_columns="image")

    # Apply repeat operations.
    data_set = data_set.repeat(repeat_num)

    # Apply batch operations.
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    return data_set
```

### Creating the Model

The VGG16 model is used as an example. You can use your own model.

```python
def _make_layer(base, args, batch_norm):
    """Make stage network of VGG."""
    layers = []
    in_channels = 3
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               padding=args.padding,
                               pad_mode=args.pad_mode,
                               has_bias=args.has_bias,
                               weight_init='XavierUniform')
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(layers)


class Vgg(nn.Cell):
    """
    VGG network definition.
    """

    def __init__(self, base, num_classes=1000, batch_norm=False, batch_size=1, args=None, phase="train"):
        super(Vgg, self).__init__()
        _ = batch_size
        self.layers = _make_layer(base, args, batch_norm=batch_norm)
        self.flatten = nn.Flatten()
        dropout_ratio = 0.5
        if not args.has_dropout or phase == "test":
            dropout_ratio = 1.0
        self.classifier = nn.SequentialCell([
            nn.Dense(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, num_classes)])

    def construct(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


base16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def vgg16(num_classes=1000, args=None, phase="train"):
    net = Vgg(base16, num_classes=num_classes, args=args, batch_norm=args.batch_norm, phase=phase)
    return net
```

### Using Membership Inference for Privacy Security Evaluation

1. Build the VGG16 model and load the parameter file.

    You can directly load the pre-trained VGG16 parameter settings or use the preceding network for training.

    ```python
    ...
    # load parameter
    parser = argparse.ArgumentParser("main case arg parser.")
    parser.add_argument("--data_path", type=str, required=True, help="Data home path for dataset")
    parser.add_argument("--pre_trained", type=str, required=True, help="Checkpoint path")
    args = parser.parse_args()
    args.batch_norm = True
    args.has_dropout = False
    args.has_bias = False
    args.padding = 0
    args.pad_mode = "same"
    args.weight_decay = 5e-4
    args.loss_scale = 1.0

    # Load the pretrained model.
    net = vgg16(num_classes=100, args=args)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9,
                      weight_decay=args.weight_decay, loss_scale=args.loss_scale)
    load_param_into_net(net, load_checkpoint(args.pre_trained))
    model = Model(network=net, loss_fn=loss, optimizer=opt)
    ```

2. Load the CIFAR-100 dataset and split it into a training set and a test set of the membership inference model at the ratio of 8:2.

    ```python
    # Load and split dataset.
    train_dataset = vgg_create_dataset100(data_home=args.data_path, image_size=(224, 224),
                                          batch_size=64, num_samples=5000, shuffle=False)
    test_dataset = vgg_create_dataset100(data_home=args.data_path, image_size=(224, 224),
                                         batch_size=64, num_samples=5000, shuffle=False, training=False)
    train_train, eval_train = train_dataset.split([0.8, 0.2])
    train_test, eval_test = test_dataset.split([0.8, 0.2])
    msg = "Data loading completed."
    LOGGER.info(TAG, msg)
    ```

3. Set the inference and evaluation parameters.

    Set the method and parameters for membership inference. Currently, the following inference methods are supported: KNN, LR, MLPClassifier, and RandomForestClassifier. The data type of inference parameters is list. Each method is represented by a dictionary whose keys are `method` and `params`.

    ```python
    config = [
            {
                "method": "lr",
                "params": {
                    "C": np.logspace(-4, 2, 10)
                }
            },
         {
                "method": "knn",
                "params": {
                    "n_neighbors": [3, 5, 7]
                }
            },
            {
                "method": "mlp",
                "params": {
                    "hidden_layer_sizes": [(64,), (32, 32)],
                    "solver": ["adam"],
                    "alpha": [0.0001, 0.001, 0.01]
                }
            },
            {
                "method": "rf",
                "params": {
                    "n_estimators": [100],
                    "max_features": ["auto", "sqrt"],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            }
        ]
    ```

    The training set is regarded as a positive class, and the test set is regarded as a negative class. You can set the following three evaluation metrics:
    - Accuracy: Percentage of samples correctly inferred to all samples.
    - Precision: Percentage of correctly inferred positive samples to all inferred positive samples.
    - Recall: Percentage of correctly inferred positive samples to all actual positive samples.
    If the number of samples is large enough and all the preceding metric values are greater than 0.6, the target model has privacy leakage risks.

    ```python
    metrics = ["precision", "accuracy", "recall"]
    ```

4. Train the membership inference model.

    ```python
    inference = MembershipInference(model)                  # Get inference model.

    inference.train(train_train, train_test, config)        # Train inference model.
    msg = "Membership inference model training completed."
    LOGGER.info(TAG, msg)

    result = inference.eval(eval_train, eval_test, metrics) # Eval metrics.
    count = len(config)
    for i in range(count):
        print("Method: {}, {}".format(config[i]["method"], result[i]))
    ```

5. Run the following command to start member inference training and evaluation to obtain the result:

    ```bash
    python example_vgg_cifar.py --data_path ./cifar-100-binary/ --pre_trained ./VGG16-100_781.ckpt
    ```

    Metric values of membership inference are accurate to four decimal places.

    Take the first row as an example. When lr (logical regression classification) is used for membership inference, the accuracy is 0.7132, the precision is 0.6596, and the recall is 0.8810, indicating that lr has a probability of 71.32% that can correctly determine whether a data sample belongs to a training set of the target model. In a binary classification task, the metrics indicate that membership inference is valid, that is, the model has privacy leakage risks.

    ```text
    Method: lr, {'recall': 0.8810,'precision': 0.6596,'accuracy': 0.7132}
    Method: knn, {'recall': 0.7082,'precision': 0.5613,'accuracy': 0.5774}
    Method: mlp, {'recall': 0.6729,'precision': 0.6462,'accuracy': 0.6522}
    Method: rf, {'recall': 0.8513, 'precision': 0.6655, 'accuracy': 0.7117}
    ```

## References

[1] [Shokri R , Stronati M , Song C , et al. Membership Inference Attacks against Machine Learning Models[J].](https://arxiv.org/abs/1610.05820v2)
