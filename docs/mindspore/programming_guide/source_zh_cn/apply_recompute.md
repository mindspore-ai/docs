# 重计算

 `Ascend` `GPU` `CPU` `模型训练` `中级` `高级`

<!-- TOC -->

- [重计算](#重计算)
    - [概述](#概述)
    - [准备工作](#准备工作)
    - [配置重计算](#配置重计算)
    - [训练模型](#训练模型)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/apply_recompute.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

MindSpore采用反向模式的自动微分，根据正向图计算流程来自动推导出反向图，正向图和反向图一起构成了完整的计算图。在计算某些反向算子时，需要用到一些正向算子的计算结果，导致这些正向算子的计算结果需要驻留在内存中，直到依赖它们的反向算子计算完，这些正向算子的计算结果占用的内存才会被复用。这一现象推高了训练的内存峰值，在大规模网络模型中尤为显著。

为了解决这个问题，MindSpore提供了重计算的功能，可以不保存正向算子的计算结果，让这些内存可以被复用，然后在计算反向算子时，如果需要正向的结果，再重新计算正向算子。此教程以模型ResNet-50为例，讲解MindSpore如何配置重计算功能去训练模型。

## 准备工作

1. 准备模型代码。ResNet-50模型的代码可参见：<https://gitee.com/mindspore/models/tree/master/official/cv/resnet>，其中，`train.py`为训练的主函数所在，`src/`目录中包含ResNet-50模型的定义和配置信息等，`script/`目录中包含一些训练和推理脚本。
2. 准备数据集。本样例采用`CIFAR-10`数据集，数据集的下载和加载方式可参考：<https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_ascend.html>。

## 配置重计算

我们可以通过调用两种接口去配置重计算，以`src/resnet.py`为例：

1. 调用`Primitive`的[recompute接口](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Primitive.html#mindspore.ops.Primitive.recompute)，调用该接口之后，在计算反向部分时，该算子会被重新计算。

   ```python
   class ResNet(nn.Cell):
       ...
       def __init__(self,
                    block,
                    layer_nums,
                    in_channels,
                    out_channels,
                    strides,
                    num_classes,
                    use_se=False,
                    res_base=False):
           super(ResNet, self).__init__()
           ...
           self.relu = ops.ReLU()
           self.relu.recompute()
           ...
   ```

2. 调用`Cell`的[recompute接口](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.recompute)，调用该接口之后，在计算反向部分时，除了该Cell的输出算子，Cell里面其他的所有算子以及子Cell里面的所有算子都会被重新计算。

   ```python
   class ResNet(nn.Cell):
       def __init__(self,
                    block,
                    layer_nums,
                    in_channels,
                    out_channels,
                    strides,
                    num_classes,
                    use_se=False,
                    res_base=False):
           super(ResNet, self).__init__()
           ...
           self.layer1 = self._make_layer(block,
                                          layer_nums[0],
                                          in_channel=in_channels[0],
                                          out_channel=out_channels[0],
                                          stride=strides[0],
                                          use_se=self.use_se)

       def _make_layer(self, block, layer_num, in_channel, out_channel, stride, use_se=False, se_block=False):
           ...
           if se_block:
               for _ in range(1, layer_num - 1):
                   resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
                   resnet_block.recompute()
           else:
               for _ in range(1, layer_num):
                   resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
                   resnet_block.recompute()
           ...

   class ResidualBlock(nn.Cell):
       def __init__(self,
                    in_channel,
                    out_channel,
                    stride=1,
                    use_se=False, se_block=False):
           super(ResidualBlock, self).__init__()
           ...

       def construct(self, x):
           ...

   def resnet50(class_num=10):
       return ResNet(ResidualBlock,
                     [3, 4, 6, 3],
                     [64, 256, 512, 1024],
                     [256, 512, 1024, 2048],
                     [1, 2, 2, 2],
                     class_num)
   ```

## 训练模型

以GPU环境为例，使用训练脚本`scripts/run_standalone_train_gpu.sh`。执行命令：`bash scripts/run_standalone_train_gpu.sh $数据集路径 config/resnet50_cifar10_config.yaml`。
通过在`src/train.py`中设置context：`save_graph=True`，可以打印出计算图结构进行对比。

设置重计算前：

```text
...
%56(equivoutput) = Conv2D(%53, %55) {instance name: conv2d} primitive_attrs: {pad_list: (0, 0, 0, 0), stride: (1, 1, 1, 1), pad: (0, 0, 0, 0), pad_mode: 1, out_channel: 64, kernel_size: (1, 1), input_names: [x, w], format: NCHW, groups: 1, mode: 1, group: 1, dilation: (1, 1, 1, 1), output_names: [output]}
      : (<Tensor[Float16], (32, 256, 56, 56)>, <Tensor[Float16], (64, 256, 1, 1)>) -> (<Tensor[Float16], (32, 64, 56, 56)>)
...
%61(equiv[CNode]707) = BatchNorm(%56, %57, %58, %59, %60) {instance name: bn_train} primitive_attrs: {epsilon: 0.000100, is_training: true, momentum: 0.100000, format: NCHW, output_names: [y, batch_mean, batch_variance, reserve_space_1, reserve_space_2], input_names: [x, scale, offset, mean, variance]}
      : (<Tensor[Float16], (32, 64, 56, 56)>, <Tensor[Float32], (64)>, <Tensor[Float32], (64)>, <Tensor[Float32], (64)>, <Tensor[Float32], (64)>) -> (<Tuple[Tensor[Float16],Tensor[Float32]*4]>)
...
%927(out) = BatchNormGrad(%923, %56, %57, %924, %925, %926) primitive_attrs: {epsilon: 0.000100, format: NCHW, is_training: true} cnode_primal_attrs: {forward_node_name: BatchNorm_102499}
      : (<Tensor[Float16], (32, 64, 56, 56)>, <Tensor[Float16], (32, 64, 56, 56)>, <Tensor[Float32], (64)>, <Tensor[Float32], (64)>, <Tensor[Float32], (64)>, <Tensor[Float32], (64)>) -> (<Tuple[Tensor[Float16],Tensor[Float32]*2]>)
...
```

设置重计算后：

```text
...
%56(equivoutput) = Conv2D(%53, %55) {instance name: conv2d} primitive_attrs: {pad_list: (0, 0, 0, 0), stride: (1, 1, 1, 1), pad: (0, 0, 0, 0), pad_mode: 1, out_channel: 64, kernel_size: (1, 1), input_names: [x, w], format: NCHW, groups: 1, mode: 1, group: 1, dilation: (1, 1, 1, 1), output_names: [output]} cnode_attrs: {need_cse_after_recompute: true, recompute: true}
      : (<Tensor[Float16], (32, 256, 56, 56)>, <Tensor[Float16], (64, 256, 1, 1)>) -> (<Tensor[Float16], (32, 64, 56, 56)>)
...
%61(equiv[CNode]707) = BatchNorm(%56, %57, %58, %59, %60) {instance name: bn_train} primitive_attrs: {epsilon: 0.000100, is_training: true, momentum: 0.100000, format: NCHW, output_names: [y, batch_mean, batch_variance, reserve_space_1, reserve_space_2], input_names: [x, scale, offset, mean, variance]}
      : (<Tensor[Float16], (32, 64, 56, 56)>, <Tensor[Float32], (64)>, <Tensor[Float32], (64)>, <Tensor[Float32], (64)>, <Tensor[Float32], (64)>) -> (<Tuple[Tensor[Float16],Tensor[Float32]*4]>)
...
%1094([CNode]15682) = Conv2D(%1091, %1093) {instance name: conv2d} primitive_attrs: {pad_list: (1, 1, 1, 1), stride: (1, 1, 1, 1), pad: (0, 0, 0, 0), pad_mode: 1, out_channel: 64, kernel_size: (3, 3), input_names: [x, w], format: NCHW, groups: 1, mode: 1, group: 1, dilation: (1, 1, 1, 1), output_names: [output]} cnode_attrs: {need_cse_after_recompute: true, duplicated: true}
      : (<Tensor[Float16], (32, 64, 56, 56)>, <Tensor[Float16], (64, 64, 3, 3)>) -> (<Tensor[Float16], (32, 64, 56, 56)>)
...
%1095([CNode]15681) = BatchNormGrad(%1085, %1094, %98, %1086, %1087, %1088) primitive_attrs: {epsilon: 0.000100, format: NCHW, is_training: true} cnode_attrs: {target_grad: true} cnode_primal_attrs: {forward_node_name: BatchNorm_102499}
      : (<Tensor[Float16], (32, 64, 56, 56)>, <Tensor[Float16], (32, 64, 56, 56)>, <Tensor[Float32], (64)>, <Tensor[Float32], (64)>, <Tensor[Float32], (64)>, <Tensor[Float32], (64)>) -> (<Tuple[Tensor[Float16],Tensor[Float32]*2]>)
...
```

可见，`Conv2D`算子被复制出来了一份，作为反向算子`BatchNormGrad`的输入。
