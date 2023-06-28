# Differential Privacy in Machine Learning

<a href="https://gitee.com/mindspore/docs/blob/r0.5/tutorials/source_en/advanced_use/differential_privacy.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

Differential privacy is a mechanism for protecting user data privacy. What is privacy? Privacy refers to the attributes of individual users. Common attributes shared by a group of users may not be considered as privacy. For example, if we say "smoking people have a higher probability of getting lung cancer", it does not disclose privacy. However, if we say "Zhang San smokes and gets lung cancer", it discloses the privacy of Zhang San. Assume that there are 100 patients in a hospital and 10 of them have lung cancer. If the information of any 99 patients are known, we can infer whether the remaining one has lung cancer. This behavior of stealing privacy is called differential attack. Differential privacy is a method for preventing differential attacks. By adding noise, the query results of two datasets with only one different record are nearly indistinguishable. In the above example, after differential privacy is used, the statistic information of the 100 patients achieved by the attacker is almost the same as that of the 99 patients. Therefore, the attacker can hardly infer the information of the remaining one patient.

**Differential privacy in machine learning**

Machine learning algorithms usually update model parameters and learn data features based on a large amount of data. Ideally, these models can learn the common features of a class of entities and achieve good generalization, such as "smoking patients are more likely to get lung cancer" rather than models with individual features, such as "Zhang San is a smoker who gets lung cancer." However, machine learning algorithms do not distinguish between general and individual features. The published machine learning models, especially the deep neural networks,  may unintentionally memorize and expose the features of individual entities in training data. This can be exploited by malicious attackers to reveal Zhang San's privacy information from the published model. Therefore, it is necessary to use differential privacy to protect machine learning models from privacy leakage.

**Differential privacy definition** [1]

$Pr[\mathcal{K}(D)\in S] \le e^{\epsilon} Pr[\mathcal{K}(D') \in S]+\delta$

For datasets $D$ and $D'$ that differ on only one record, the probability of obtaining the same result from $\mathcal{K}(D)$ and $\mathcal{K}(D')$ by using a randomized algorithm $\mathcal{K}$ must meet the preceding formula. $\epsilon$ indicates the differential privacy budget and $\delta$ indicates the perturbation. The smaller the values of $\epsilon$ and $\delta$, the closer the data distribution output by $\mathcal{K}$ on $D$ and $D'$.

**Differential privacy measurement**

Differential privacy can be measured using $\epsilon$ and $\delta$.

- $\epsilon$: specifies the upper limit of the output probability that can be changed when a record is added to or deleted from the dataset. We usually hope that $\epsilon$ is a small constant. A smaller value indicates stricter differential privacy conditions.
- $\delta$: limits the probability of arbitrary model behavior change. Generally, this parameter is set to a small constant. You are advised to set this parameter to a value less than the reciprocal of the size of a training dataset.

**Differential privacy implemented by MindArmour**

MindArmour differential privacy module Differential-Privacy implements the differential privacy optimizer. Currently, SGD, Momentum, and Adam are supported. They are differential privacy optimizers based on the Gaussian mechanism. Gaussian noise mechanism supports both non-adaptive policy and adaptive policy  The non-adaptive policy use a fixed noise parameter for each step while the adaptive policy changes the noise parameter along time or iteration step. An advantage of using the non-adaptive Gaussian noise is that a differential privacy budget $\epsilon$ can be strictly controlled. However, a disadvantage is that in a model training process, the noise amount added in each step is fixed. In the later training stage, large noise makes the model convergence difficult, and even causes the performance to decrease greatly and the model usability to be poor. Adaptive noise can solve this problem. In the initial model training stage, the amount of added noise is large. As the model converges, the amount of noise decreases gradually, and the impact of noise on model availability decreases. The disadvantage is that the differential privacy budget cannot be strictly controlled. Under the same initial value, the $\epsilon$ of the adaptive differential privacy is greater than that of the non-adaptive differential privacy. Rényi differential privacy (RDP) [2] is also provided to monitor differential privacy budgets.

The LeNet model and MNIST dataset are used as an example to describe how to use the differential privacy optimizer to train a neural network model on MindSpore.

> This example is for the Ascend 910 AI processor and supports PYNATIVE_MODE. You can download the complete sample code from <https://gitee.com/mindspore/mindarmour/blob/r0.5/example/mnist_demo/lenet5_dp_model_train.py>.

## Implementation

### Importing Library Files

The following are the required public modules, MindSpore modules, and differential privacy feature modules.

```python
import os
from easydict import EasyDict as edict

import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.nn.metrics import Accuracy
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.transforms.vision import Inter
import mindspore.common.dtype as mstype

from mindarmour.diff_privacy import DPModel
from mindarmour.diff_privacy import DPOptimizerClassFactory
from mindarmour.diff_privacy import PrivacyMonitorFactory
from mindarmour.utils.logger import LogUtil
from lenet5_net import LeNet5
from lenet5_config import mnist_cfg as cfg

LOGGER = LogUtil.get_instances()
LOGGER.set_level('INFO')
TAG = 'Lenet5_train'
```

### Configuring Parameters

1. Set the running environment, dataset path, model training parameters, checkpoint storage parameters, and differential privacy parameters.
   
   ```python
   cfg = edict({
        'num_classes': 10,  # the number of classes of model's output
        'lr': 0.1,  # the learning rate of model's optimizer
        'momentum': 0.9,  # the momentum value of model's optimizer
        'epoch_size': 10,  # training epochs
        'batch_size': 256,  # batch size for training
        'image_height': 32,  # the height of training samples
        'image_width': 32,  # the width of training samples
        'save_checkpoint_steps': 234,  # the interval steps for saving checkpoint file of the model
        'keep_checkpoint_max': 10,  # the maximum number of checkpoint files would be saved
        'device_target': 'Ascend',  # device used
        'data_path': './MNIST_unzip',  # the path of training and testing data set
        'dataset_sink_mode': False,  # whether deliver all training data to device one time
        'micro_batches': 16,  # the number of small batches split from an original batch
        'norm_clip': 1.0,  # the clip bound of the gradients of model's training parameters
        'initial_noise_multiplier': 1.5,  # the initial multiplication coefficient of the noise added to training
        # parameters' gradients
        'mechanisms': 'AdaGaussian',  # the method of adding noise in gradients while training
        'optimizer': 'Momentum'  # the base optimizer used for Differential privacy training
   })
   ```

2. Configure necessary information, including the environment information and execution mode.

   ```python
   context.set_context(mode=context.PYNATIVE_MODE, device_target=cfg.device_target)
   ```

   For details about the API configuration, see the `context.set_context`.

### Preprocessing the Dataset

Load the dataset and convert the dataset format to a MindSpore data format.

```python
def generate_mnist_dataset(data_path, batch_size=32, repeat_size=1,
                           num_parallel_workers=1, sparse=True):
    """
    create dataset for training or testing
    """
    # define dataset
    ds1 = ds.Dataset(data_path)

    # define operation parameters
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width),
                          interpolation=Inter.LINEAR)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    if not sparse:
        one_hot_enco = C.OneHot(10)
        ds1 = ds1.map(input_columns="label", operations=one_hot_enco,
                      num_parallel_workers=num_parallel_workers)
        type_cast_op = C.TypeCast(mstype.float32)
    ds1 = ds1.map(input_columns="label", operations=type_cast_op,
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=resize_op,
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=rescale_op,
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=hwc2chw_op,
                  num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    ds1 = ds1.shuffle(buffer_size=buffer_size)
    ds1 = ds1.batch(batch_size, drop_remainder=True)
    ds1 = ds1.repeat(repeat_size)

    return ds1
```

### Creating the Model

The LeNet model is used as an example. You can also create and train your own model.

```python
from mindspore import nn
from mindspore.common.initializer import TruncatedNormal


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    return TruncatedNormal(0.05)


class LeNet5(nn.Cell):
    """
    LeNet network
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16*5*5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
```

Load the LeNet network, define the loss function, configure the checkpoint parameters, and load data by using the `generate_mnist_dataset` function defined in the preceding information.

```python
network = LeNet5()
net_loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                             keep_checkpoint_max=cfg.keep_checkpoint_max)
ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet",
                             directory='./trained_ckpt_file/',
                             config=config_ck)

# get training dataset
ds_train = generate_mnist_dataset(os.path.join(cfg.data_path, "train"),
                                  cfg.batch_size,
                                  cfg.epoch_size)
```

### Introducing the Differential Privacy

1. Set parameters of a differential privacy optimizer.

   - Determine whether values of the **micro_batches** and **batch_size** parameters meet the requirements. The value of **batch_size** must be an integer multiple of **micro_batches**.
   - Instantiate a differential privacy factory class.
   - Set a noise mechanism for the differential privacy. Currently, the Gaussian noise mechanism with a fixed standard deviation (`Gaussian`) and the Gaussian noise mechanism with an adaptive standard deviation (`AdaGaussian`) are supported.
   - Set an optimizer type. Currently, `SGD`, `Momentum`, and `Adam` are supported.
   - Set up a differential privacy budget monitor RDP to observe changes in the differential privacy budget $\epsilon$ in each step.

   ```python
    if cfg.micro_batches and cfg.batch_size % cfg.micro_batches != 0:
        raise ValueError("Number of micro_batches should divide evenly batch_size")
    
    # Create a factory class of DP optimizer
    gaussian_mech = DPOptimizerClassFactory(cfg.micro_batches)

    # Set the method of adding noise in gradients while training. Initial_noise_multiplier is suggested to be greater
    # than 1.0, otherwise the privacy budget would be huge, which means that the privacy protection effect is weak.
    # mechanisms can be 'Gaussian' or 'AdaGaussian', in which noise would be decayed with 'AdaGaussian' mechanism while
    # be constant with 'Gaussian' mechanism.
    gaussian_mech.set_mechanisms(cfg.mechanisms,
                                 norm_bound=cfg.l2_norm_bound,
                                 initial_noise_multiplier=cfg.initial_noise_multiplier)

    # Wrap the base optimizer for DP training. Momentum optimizer is suggested for LenNet5.
    net_opt = gaussian_mech.create(cfg.optimizer)(params=network.trainable_params(),
                                                  learning_rate=cfg.lr,
                                                  momentum=cfg.momentum)

    # Create a monitor for DP training. The function of the monitor is to compute and print the privacy budget(eps
    # and delta) while training.
    rdp_monitor = PrivacyMonitorFactory.create('rdp',
                                               num_samples=60000,
                                               batch_size=cfg.batch_size,
                                               initial_noise_multiplier=cfg.initial_noise_multiplier*
                                               cfg.l2_norm_bound,
                                               per_print_times=50)

   ```

2. Package the LeNet model as a differential privacy model by transferring the network to `DPModel`.

   ```python
   # Create the DP model for training.
   model = DPModel(micro_batches=cfg.micro_batches,
                   norm_clip=cfg.l2_norm_bound,
                   dp_mech=gaussian_mech.mech,
                   network=network,
                   loss_fn=net_loss,
                   optimizer=net_opt,
                   metrics={"Accuracy": Accuracy()})
   ```

3. Train and test the model.

   ```python
   LOGGER.info(TAG, "============== Starting Training ==============")
   model.train(cfg['epoch_size'], ds_train, callbacks=[ckpoint_cb, LossMonitor(), rdp_monitor],
               dataset_sink_mode=cfg.dataset_sink_mode)

   LOGGER.info(TAG, "============== Starting Testing ==============")
   ckpt_file_name = 'trained_ckpt_file/checkpoint_lenet-10_234.ckpt'
   param_dict = load_checkpoint(ckpt_file_name)
   load_param_into_net(network, param_dict)
   ds_eval = generate_mnist_dataset(os.path.join(cfg.data_path, 'test'), batch_size=cfg.batch_size)
   acc = model.eval(ds_eval, dataset_sink_mode=False)
   LOGGER.info(TAG, "============== Accuracy: %s  ==============", acc)
   ```
   
4. Run the following command.
   
   Execute the script:
   
   ```bash
   python lenet5_dp_model_train.py
   ```
   
   In the preceding command, replace `lenet5_dp_model_train.py` with the name of your script.
   
5. Display the result.

   The accuracy of the LeNet model without differential privacy is 99%, and the accuracy of the LeNet model with adaptive differential privacy AdaDP is 98%.
   ```
   ============== Starting Training ==============
   ...
   ============== Starting Testing ==============
   ...
   ============== Accuracy: 0.9879  ==============
   ```
   
### References

[1] C. Dwork and J. Lei. Differential privacy and robust statistics. In STOC, pages 371–380. ACM, 2009.

[2] Ilya Mironov. Rényi diﬀerential privacy. In IEEE Computer Security Foundations Symposium, 2017.

[3] Abadi, M. e. a., 2016. *Deep learning with differential privacy.* s.l.:Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security.



