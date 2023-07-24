# Migrating Training Scripts from Third Party Frameworks

`Linux` `Ascend` `GPU` `CPU` `Whole Process` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/tutorials/training/source_en/advanced_use/migrate_3rd_scripts.md)

## Overview

You've probably written scripts for frameworks such as TensorFlow and PyTorch. This tutorial describes how to migrate existing TensorFlow and PyTorch networks to MindSpore, including key steps and operation recommendations which help you quickly migrate your network.

## Preparations

Before you start working on your scripts, prepare your operator assessment and hardware and software environments to make sure that MindSpore can support the network you want to migrate.

### Operator Assessment

Analyze the operators contained in the network to be migrated and figure out how does MindSpore support these operators based on the [Operator List](https://www.mindspore.cn/doc/note/en/r1.1/operator_list.html).

Take ResNet-50 as an example. The two major operators [Conv](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/nn/mindspore.nn.Conv2d.html) and [BatchNorm](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/nn/mindspore.nn.BatchNorm2d.html) exist in the MindSpore Operator List.

If any operator does not exist, you are advised to perform the following operations:

- Operator replacement: Analyze the operator implementation formula and check whether a combination of existing operators of MindSpore can be used to achieve the expected objective.
- Substitution solution: For example, if a loss operator is not supported, check whether it can be replaced with a loss operator of the same type supported by MindSpore; alternatively, check whether the current network structure can be replaced by another mainstream network of the same type.

If the operators used for replacement are not able to fulfill complete function, you are advised to perform the following operations:

- Delete unnecessary functions.
- Find a substitution solution for necessary functions.

If the preceding requirements cannot be met, you can raise requirements in the [MindSpore code repository](https://gitee.com/mindspore/mindspore).

### Software and Hardware Environments

Prepare the hardware environment, find a platform corresponding to your environment by referring to the [installation guide](https://www.mindspore.cn/install/en), and install MindSpore.

## E2E Network Migration

### Training Phase

#### Script Migration

MindSpore differs from TensorFlow and PyTorch in the network structure. Before migration, you need to clearly understand the original script and information of each layer, such as shape.

> You can also use [MindConverter Tool](https://gitee.com/mindspore/mindinsight/tree/r1.1/mindinsight/mindconverter) to automatically convert the PyTorch network definition script to MindSpore network definition script.

The ResNet-50 network migration and training on the Ascend 910 is used as an example.

1. Import MindSpore modules.

   Import the corresponding MindSpore modules based on the required APIs. For details about the module list, see <https://www.mindspore.cn/doc/api_python/en/r1.1/index.html>.

2. Load and preprocess a dataset.

   Use MindSpore to build the required dataset. Currently, MindSpore supports common datasets. You can call APIs in the original format, `MindRecord`, and `TFRecord`. In addition, MindSpore supports data processing and data augmentation. For details, see the [Data Preparation](https://www.mindspore.cn/tutorial/training/en/r1.1/use/data_preparation.html).

   In this example, the CIFAR-10 dataset is loaded, which supports both single-GPU and multi-GPU scenarios.

   ```python
   if device_num == 1:
       ds = de.Cifar10Dataset(dataset_path, num_parallel_workers=4, shuffle=True)
   else:
       ds = de.Cifar10Dataset(dataset_path, num_parallel_workers=4, shuffle=True,
                              num_shards=device_num, shard_id=rank_id)
   ```

   Then, perform data augmentation, data cleaning, and batch processing. For details about the code, see <https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/resnet/src/dataset.py>.

3. Build a network.

   The biggest difference between MindSpore and TensorFlow in convolution is the data format. `NCHW` is used in MindSpore by default, while `NHWC` is used in TensorFlow.

   The following uses the first convolutional layer on the ResNet-50 network whose batch\_size is set to 32 as an example:

    - In TensorFlow, the format of the input feature is \[32, 224, 224, 3], and the size of the convolution kernel is \[7, 7, 3, 64].

    - In MindSpore, the format of the input feature is \[32, 3, 224, 224], and the size of the convolution kernel is \[64, 3, 7, 7].

      ```python
      def _conv7x7(in_channel, out_channel, stride=1):
          weight_shape = (out_channel, in_channel, 7, 7)
          weight = _weight_variable(weight_shape)
          return nn.Conv2d(in_channel, out_channel,
                          kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init=weight)


       def _bn(channel):
           return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                               gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
       ```

4. Build a subnet.

   In MindSpore, `nn.Cell` is used to build a subnet structure. The network structure must be defined before being used in a subnet. Define each operator to be used in the `__init__` function of the Cell, connect the defined operators in the `construct` function, and then return the output of the subnet through `return`.

   ```python
   class ResidualBlock(nn.Cell):
       """
       ResNet V1 residual block definition.

       Args:
           in_channel (int): Input channel.
           out_channel (int): Output channel.
           stride (int): Stride size for the first convolutional layer. Default: 1.

       Returns:
           Tensor, output tensor.

       Examples:
           >>> ResidualBlock(3, 256, stride=2)
       """
       expansion = 4

       def __init__(self,
                   in_channel,
                   out_channel,
                   stride=1):
           super(ResidualBlock, self).__init__()

           channel = out_channel
           self.conv1 = _conv1x1(in_channel, channel, stride=1)
           self.bn1 = _bn(channel)

           self.conv2 = _conv3x3(channel, channel, stride=stride)
           self.bn2 = _bn(channel)

           self.conv3 = _conv1x1(channel, out_channel, stride=1)
           self.bn3 = _bn_last(out_channel)

           self.relu = nn.ReLU()

           self.down_sample = False

           if stride != 1 or in_channel != out_channel:
               self.down_sample = True
           self.down_sample_layer = None

           if self.down_sample:
               self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride),
                                                           _bn(out_channel)])
           self.add = ops.Add()

       def construct(self, x):
           identity = x

           out = self.conv1(x)
           out = self.bn1(out)
           out = self.relu(out)

           out = self.conv2(out)
           out = self.bn2(out)
           out = self.relu(out)

           out = self.conv3(out)
           out = self.bn3(out)

           if self.down_sample:
               identity = self.down_sample_layer(identity)

           out = self.add(out, identity)
           out = self.relu(out)

           return out
   ```

5. Define a concatenated structure.

   The ResNet-50 network has a large number of repeated structures. In TensorFlow, you can use the for loop function to reduce repeated code. In MindSpore, each defined Cell object is independent. Especially for subnets with weight parameters, the defined Cell cannot be used repeatedly. If a large number of repeated concatenated structures exist, you can construct multiple Cell instances using the for loop function and concatenate them by using `SequentialCell`.

   ```python
   def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
       """
       Make stage network of ResNet.

       Args:
           block (Cell): Resnet block.
           layer_num (int): Layer number.
           in_channel (int): Input channel.
           out_channel (int): Output channel.
           stride (int): Stride size for the first convolutional layer.

       Returns:
           SequentialCell, the output layer.

       Examples:
           >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
       """
       layers = []

       resnet_block = block(in_channel, out_channel, stride=stride)
       layers.append(resnet_block)

       for _ in range(1, layer_num):
           resnet_block = block(out_channel, out_channel, stride=1)
           layers.append(resnet_block)

       return nn.SequentialCell(layers)
   ```

6. Build the entire network.

   The [ResNet-50](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/resnet/src/resnet.py) network structure is formed by connecting multiple defined subnets. Follow the rule of defining subnets before using them and define all the subnets used in the `__init__` and connect subnets in the `construct`.

7. Define a loss function and an optimizer.

   After the network is defined, the loss function and optimizer need to be defined accordingly.

   ```python
   loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
   opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum, config.weight_decay, config.loss_scale)
   ```

8. Build a model.

   Similar to the `Estimator` API of TensorFlow, the defined network prototype, loss function, and optimizer are transferred to the `Model` API of MindSpore and automatically combined into a network that can be used for training.

   To use loss scale in training, define a `loss_scale_manager` and transfer it to the `Model` API.

   ```python
   loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
   ```

   You can use a built-in assessment method of `Model` by setting the [metrics](https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/custom_debugging_info.html#mindspore-metrics) attribute.

   ```python
   model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'})
   ```

   Similar to `estimator.train` of TensorFlow, you can call the `model.train` API to perform training. Functions such as CheckPoint and intermediate result printing can be defined on the `model.train` API in Callback mode.

   ```python
   time_cb = TimeMonitor(data_size=step_size)
   loss_cb = LossMonitor()
   cb = [time_cb, loss_cb]
   if config.save_checkpoint:
       config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                       keep_checkpoint_max=config.keep_checkpoint_max)
       ckpt_cb = ModelCheckpoint(prefix="resnet", directory=config.save_checkpoint_path, config=config_ck)
       cb += [ckpt_cb]
   model.train(epoch_size, dataset, callbacks=cb)
   ```

#### Accuracy Debugging

The accuracy optimization process is as follows:

1. When validating the single-GPU accuracy, you are advised to use a small dataset for training. After the validation is successful, use the full dataset for multi-GPU accuracy validation. This helps improve the debugging efficiency.
2. Delete unnecessary skills (such as augmentation configuration and dynamic loss scale in an optimizer) from the script. After the validation is successful, add functions one by one. After a new function is confirmed to be normal, add the next function. In this way, you can quickly locate the fault.

#### On-Cloud Integration

Run your scripts on ModelArts. For details, see [Using MindSpore on Cloud](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/use_on_the_cloud.html).

### Inference Phase

Models trained on the Ascend 910 AI processor can be used for inference on different hardware platforms. Refer to the [Multi-platform Inference Tutorial](https://www.mindspore.cn/tutorial/inference/en/r1.1/multi_platform_inference.html) for detailed steps.

## Examples

- [Model Zoo](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo)
