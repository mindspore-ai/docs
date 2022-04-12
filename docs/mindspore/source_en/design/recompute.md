# Recomputation

 `Ascend` `GPU` `CPU` `Model Running` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/design/recompute.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

The automatic differential of MindSpore is in reverse-mode, which derives the backward pass according to the forward pass. Before some backward operators are computed, the results of some forward operators should be ready. It leads to the problem that the memory occupied by these results of the forward operators, can not be reused until the computation of the backward operators are completed. This problem can  drive up the peak of memory, which is particularly significant in the large model.

In order to solve this problem, Mindspore provides the recomputation function. It will recompute the forward operators before computing the backward operators rather than storing the results of forward operators, which can help the memory be reused. This tutorial takes the model ResNet-50 for example to explain how to configure recomputation to train your model in MindSpore.

## Preliminaries

1. Prepare the model. The ResNet-50 code can be found at: <https://gitee.com/mindspore/models/tree/master/official/cv/resnet>, in which `train.py` is the main function for training, `src/` directory contains the model definition and configuration files of ResNet-50, `script/` directory contains the training and evaluation scripts.

2. Prepare the dataset. This example uses the `CIFAR-10` dataset. For details about how to download and load the dataset, visit <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html>.

## Configuring for Recomputation

We can call two kinds of interface to configure the recomputation. Take `src/resnet.py` for example:

1. The [recompute api](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Primitive.html#mindspore.ops.Primitive.recompute) of `Primitive`. It can set an operator to be recomputed. After setting, the operator will be recomputed in the backward pass.

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

2. Call the [recompute api](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.recompute) of `Cell`. It can set the whole `Cell` to be recomputed. After setting, except the output of the `Cell`, all the operators in this cell and its sub `Cell` will be recomputed in the backward pass.

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

## Training the Model

We take the GPU environment for example, use the script `script/run_standalone_train_gpu.sh`. Run the command `bash scripts/run_standalone_train_gpu.sh $date_set_path config/resnet50_cifar10_config.yaml`.
We can set the context: `save_graph=True` in `src/train.py` to print the construction of the computation graph to do comparison.

The graph before setting recomputation is as follow:

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

The graph after setting recomputation is as follow:

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

We can see that `Conv2D` is replicated to become the input to `BatchNormGrad`.
