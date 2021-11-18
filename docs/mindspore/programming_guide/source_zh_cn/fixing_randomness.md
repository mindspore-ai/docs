# 固定随机性以复现脚本运行结果

`Linux` `Ascend` `模型运行` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/fixing_randomness.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

固定随机性的目的是复现脚本运行结果，以便定位问题。例如，固定随机性后，相同条件下的两次训练所产生的loss曲线应基本一致，您可以多次执行调试，方便地查找loss曲线异常的原因而无需担心上次调试的问题现象在本次运行时不再出现。请注意，即使在所有可固定的随机性都固定后，也未必可以在MindSpore上精确复现运行结果。特别是当使用的MindSpore版本（commit id）不同，或者是执行脚本的机器不是同一台机器时，即使使用相同的种子也不一定能够复现运行结果。

固定随机性后，有可能会出现运行性能下降的情况，因此建议在问题修复后，取消固定随机性，删除相关的脚本改动，以免影响脚本正常的运行性能。

固定MindSpore脚本随机性的步骤如下：

1. 在脚本开始处固定全局随机数种子。

   包括MindSpore全局随机数种子，[mindspore.set_seed(1)](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore/mindspore.set_seed.html#mindspore.set_seed)，[mindspore.dataset.config.set_seed(1)](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.dataset.config.html#mindspore.dataset.config.set_seed)；numpy等三方库的全局随机数种子`numpy.random.seed(1)`；Python随机数种子`random.seed(1)`等。样例代码如下：

    ```python
    import random

    import numpy

    import mindspore

    mindspore.set_seed(1)
    mindspore.dataset.config.set_seed(1)
    numpy.random.seed(1)
    random.seed(1)
    ```

2. 固定超参。

   建议以明确的数值指定各个超参，涉及到动态学习率的，请确保生成动态学习率的各个参数都是确定的。避免使用带有随机性的超参。

3. 固定初始化权重。

   建议通过加载固定checkpoint文件的形式固定初始化权重。加载checkpoint时要确保文件被完全加载，不能pop出某些key后再加载。

4. 固定数据处理方法和数据顺序。

   删除或替换所有随机数据处理算子（例如 删除[RandomHorizontalFlip](https://mindspore.cn/docs/api/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.RandomHorizontalFlip.html#mindspore.dataset.vision.c_transforms.RandomHorizontalFlip)、将[RandomCrop](https://mindspore.cn/docs/api/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.RandomCrop.html#mindspore.dataset.vision.c_transforms.RandomCrop)替换为[Crop](https://mindspore.cn/docs/api/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.Crop.html#mindspore.dataset.vision.c_transforms.Crop)等）。关闭shuffle功能。不要使用数据集的sampler。将`num_parallel_workers`参数设置为1以避免并行数据处理对数据顺序的影响。如果需要从某个迭代开始训练，可以使用`dataset.skip()`接口跳过之前迭代的数据。目前已知的随机算子包括：所有名称中带有Random的算子。

5. 固定网络。

   删除网络中带有随机性的算子，例如DropOut算子和名称中带有Random的算子。若有的随机算子确实不能删除，则应该设置固定的随机数种子（随机数种子建议选择0以外的数字）。DropOut算子随机性在部分场景下难以固定，建议始终删除。目前已知的随机算子包括：[Random Operators](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.ops.html#random-operators)，所有名称中带有DropOut的算子。此外，Ascend后端上使用atomic_write特性的算子也有微小的随机性，该随机性不会引起计算结果的错误，只是会导致算子在输入相同的两次计算之间产生微小的差异，算子列表请见本文末尾。

进行上述操作后，在相同环境下两次运行训练脚本，检查loss曲线。若loss曲线基本一致（至少前两个loss值的相对差异和绝对差异均在千分之一以内，参考[numpy.allclose()](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html)），则说明成功固定了随机性。若loss曲线不一致，应检查上述固定随机性的步骤是否都做到位了。如果固定随机性的操作均做到了，但是前两个loss值还是不一致，请[新建issue向MindSpore求助](https://gitee.com/mindspore/mindspore/issues/new)。

## 说明

1. 本文档主要适用于Ascend后端上`GRAPH_MODE`的训练脚本。
2. Ascend后端上使用atomic_write特性的算子列表：

    - [ReduceSum](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ReduceSum.html#mindspore.ops.ReduceSum)
    - [DynamicGRUV2](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.DynamicGRUV2.html#mindspore.ops.DynamicGRUV2)
    - [DynamicRNN](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.DynamicRNN.html#mindspore.ops.DynamicRNN)
    - [LayerNorm](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.LayerNorm.html#mindspore.ops.LayerNorm)
    - [NLLLoss](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.NLLLoss.html#mindspore.ops.NLLLoss)
