# 迁移第三方框架训练脚本

`Linux` `Ascend` `GPU` `CPU` `全流程` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.1/tutorials/training/source_zh_cn/advanced_use/migrate_3rd_scripts.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

你可能已经编写过TensorFlow、PyTorch等框架的脚本，本教程介绍如何将已有的TensorFlow、PyTorch等的网络迁移到MindSpore，包括主要步骤和操作建议，帮助你快速进行网络迁移。

## 准备环节

在动手改造你的脚本前，请先做好算子评估和软硬件环境准备，确保MindSpore可以支持你希望迁移的网络。

### 算子评估

分析待迁移的网络中所包含的算子，结合[MindSpore算子支持列表](https://www.mindspore.cn/doc/note/zh-CN/r1.1/operator_list_ms.html)，梳理出MindSpore对这些算子的支持程度。

以ResNet-50为例，[Conv](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/nn/mindspore.nn.Conv2d.html)和[BatchNorm](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/nn/mindspore.nn.BatchNorm2d.html)是其中最主要的两个算子，它们已在MindSpore支持的算子列表中。

如果发现没有对应算子，建议：

- 使用其他算子替换：分析算子实现公式，审视是否可以采用MindSpore现有算子叠加达到预期目标。
- 临时替代方案：比如不支持某个Loss，是否可以替换为同类已支持的Loss算子；又比如当前的网络结构，是否可以替换为其他同类主流网络等。

如果发现支持的算子存在功能不全，建议：

- 非必要功能：可删除。
- 必要功能：寻找替代方案。

如果上述仍不能满足你的要求，你可以在[MindSpore代码仓](https://gitee.com/mindspore/mindspore)提出诉求。

### 软硬件环境准备

准备好硬件环境，查看与你环境对应平台的[安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装。

## E2E迁移网络

### 训练阶段

#### 脚本迁移

MindSpore与TensorFlow、PyTorch在网络结构组织方式上，存在一定差别，迁移前需要对原脚本有较为清晰的了解，明确地知道每一层的shape等信息。

> 你也可以使用[MindConverter工具](https://gitee.com/mindspore/mindinsight/tree/r1.1/mindinsight/mindconverter)实现PyTorch网络定义脚本到MindSpore网络定义脚本的自动转换。

下面，我们以ResNet-50的迁移，并在Ascend 910上训练为例：

1. 导入MindSpore模块。

    根据所需使用的接口，导入相应的MindSpore模块，模块列表详见<https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/index.html>。

2. 加载数据集和预处理。

    使用MindSpore构造你需要使用的数据集。目前MindSpore已支持常见数据集，你可以通过原始格式、`MindRecord`、`TFRecord`等多种接口调用，同时还支持数据处理以及数据增强等相关功能，具体用法可参考[准备数据教程](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/use/data_preparation.html)。

    本例中加载了Cifar-10数据集，可同时支持单卡和多卡的场景。

    ```python
    if device_num == 1:
        ds = de.Cifar10Dataset(dataset_path, num_parallel_workers=4, shuffle=True)
    else:
        ds = de.Cifar10Dataset(dataset_path, num_parallel_workers=4, shuffle=True,
                               num_shards=device_num, shard_id=rank_id)
    ```

    然后对数据进行了数据增强、数据清洗和批处理等操作。代码详见<https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/resnet/src/dataset.py>。

3. 构建网络。

    与TensorFlow相比，MindSpore对于卷积的最大差异在于数据格式。MindSpore整网默认使用`NCHW`的格式，与常见的TensorFlow所使用的`NHWC`不同。

    以batch_size=32的ResNet-50网络中第一层卷积为例:

    - 在TensorFlow中，输入feature的格式为[32, 224, 224, 3]，卷积核大小为[7, 7, 3, 64]。
    - 在MindSpore中，输入feature的格式为[32, 3, 224, 224]，卷积核大小为[64, 3, 7, 7]。

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

4. 构造子网。

    MindSpore中使用`nn.Cell`来构造一个子网结构。子网内遵循先定义后使用的原则来搭建网络结构。每一个需要使用的算子需先定义在Cell的`__init__`函数内，然后在`construct`函数内将定义好的算子连接起来，最后将子网的输出通过`return`返回。

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

5. 定义串联结构。

    ResNet-50网络中有大量的重复结构，TensorFlow中可以使用for循环调用函数的方式来减少重复代码。MindSpore中，我们定义的每一个Cell对象都是独立的，尤其对于内部存在权重参数的子网，定义的Cell是不能重复使用的，如果出现大量重复串联结构，可以使用循环构造多个Cell实例并通过`SequentialCell`来串联。

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

6. 构造整网。

    将定义好的多个子网连接起来就是整个[ResNet-50](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/resnet/src/resnet.py)网络的结构了。同样遵循先定义后使用的原则，在`__init__`中定义所有用到的子网，在`construct`中连接子网。

7. 定义损失函数和优化器。

    定义好网络后，还需要相应地定义损失函数和优化器。

    ```python
    loss = SoftmaxCrossEntropyWithLogits(sparse=True)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum, config.weight_decay, config.loss_scale)
    ```

8. 构造模型。

    类似于TensorFlow的`Estimator`接口，将定义好的网络原型、损失函数、优化器传入MindSpore的`Model`接口，内部会自动将其组合成一个可用于训练的网络。

    如果需要在训练中使用Loss Scale，则可以单独定义一个`loss_scale_manager`，一同传入`Model`接口。

    ```python
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    ```

    如果希望使用`Model`内置的评估方法，则可以使用[metrics](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/custom_debugging_info.html#mindspore-metrics)属性设置希望使用的评估方法。

    ```python
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'})
    ```

    类似于TensorFlow的`estimator.train`，可以通过调用`model.train`接口来进行训练。CheckPoint和中间结果打印等功能，可通过`Callback`的方式定义到`model.train`接口上。

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

#### 精度调试

精度调优过程建议如下两点：  

1. 单卡精度验证时，建议先采用小数据集进行训练。验证达标后，多卡精度验证时，再采用全量数据集。这样可以帮助提升调试效率。
2. 首先删减脚本中的不必要技巧（如优化器中的增强配置、动态Loss Scale等），验证达标后，在此基础上逐个叠加新增功能，待当前新增功能确认正常后，再叠加下一个功能。这样可以帮助快速定位问题。

#### 云上集成

请参考[在云上使用MindSpore](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/use_on_the_cloud.html)，将你的脚本运行在ModelArts。

### 推理阶段

在Ascend 910 AI处理器上训练后的模型，支持在不同的硬件平台上执行推理。详细步骤可参考[多平台推理教程](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.1/multi_platform_inference.html)。

## 样例参考

1. [常用数据集读取样例](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.1/dataset_loading.html)

2. [Model Zoo](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo)
