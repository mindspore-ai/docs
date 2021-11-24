# 固定随机性以复现脚本运行结果

`Linux` `Ascend` `模型运行` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/fixing_randomness.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

固定随机性的目的是复现脚本运行结果，辅助定位问题。固定随机性后，相同条件下的两次训练所产生的loss曲线应基本一致，您可以多次执行调试，方便地查找loss曲线异常的原因而无需担心上次调试的问题现象在本次运行时不再出现。

请注意，即使在所有可固定的随机性都固定后，也未必可以在MindSpore上精确复现运行结果。特别是当使用的MindSpore版本（commit id）不同时，或者是执行脚本的机器不是同一台机器时，或者执行脚本的AI训练加速器不是同一个物理设备时，即使使用相同的种子也不一定能够复现运行结果。

固定随机性后，有可能会出现运行性能下降的情况，因此建议在问题修复后，取消固定随机性，删除相关的脚本改动，以免影响脚本正常的运行性能。

本文适用于Ascend上的静态图模式。

固定MindSpore脚本随机性的步骤如下：

1. 在要执行的脚本的开始处插入代码，固定全局随机数种子。

   需要固定的随机数种子包括MindSpore全局随机数种子[mindspore.set_seed(1)](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore/mindspore.set_seed.html#mindspore.set_seed)；numpy等三方库的全局随机数种子`numpy.random.seed(1)`；Python随机数种子`random.seed(1)`等。样例代码如下：

    ```python
    import random

    import numpy

    import mindspore

    mindspore.set_seed(1)
    numpy.random.seed(1)
    random.seed(1)
    ```

2. 固定超参。

   建议以明确的数值指定各个超参，涉及到动态学习率的，请确保生成动态学习率的各个参数都是确定的。避免使用带有随机性的超参。

3. 固定初始化权重。

   建议通过加载固定checkpoint文件的形式固定初始化权重。加载checkpoint时要确保文件被完全加载，不能pop出某些key后再加载。

4. 固定数据处理方法和数据顺序。

   （1）删除或替换所有随机数据处理算子（例如 删除[RandomHorizontalFlip](https://mindspore.cn/docs/api/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.RandomHorizontalFlip.html#mindspore.dataset.vision.c_transforms.RandomHorizontalFlip)、将[RandomCrop](https://mindspore.cn/docs/api/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.RandomCrop.html#mindspore.dataset.vision.c_transforms.RandomCrop)替换为[Crop](https://mindspore.cn/docs/api/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.Crop.html#mindspore.dataset.vision.c_transforms.Crop)等）。随机算子指所有名称中带有Random的数据处理算子。

   （2）设置`shuffle=False`以关闭shuffle功能。不要使用数据集的sampler。

   （3）将`num_parallel_workers`参数设置为1以避免并行数据处理对数据顺序的影响。

   （4）如果需要从某个迭代开始训练，可以使用`dataset.skip()`接口跳过之前迭代的数据。
   样例代码如下：

    ```python
    import mindspore.dataset as ds

    data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=1, shuffle=False)
    data_set.map(operations=trans, input_columns="image", num_parallel_workers=1)
    ```

5. 固定网络。

   删除网络中带有随机性的算子，例如DropOut算子和名称中带有Random的算子。若有的随机算子确实不能删除，则应该设置固定的随机数种子（随机数种子建议选择0以外的数字）。DropOut算子随机性在部分场景下难以固定，建议始终删除。目前已知的随机算子包括：[Random Operators](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.ops.html#random-operators)，所有名称中带有DropOut的算子。此外，Ascend后端上使用atomic_write特性的算子也有微小的随机性，该随机性不会引起计算结果的错误，只是会导致算子在输入相同的两次计算之间产生微小的差异。使用atomic_write特性的算子列表请见本文末尾。

进行上述操作后，在相同环境下两次运行训练脚本，检查loss曲线。若loss曲线基本一致（至少前两个迭代的loss值均满足atol=1e-3，rtol=1e-3的条件下[numpy.allclose()](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html)为True，则说明成功固定了随机性。若loss曲线不一致，应检查上述固定随机性的步骤是否都做到位了。如果固定随机性的操作均做到了，但是前两个loss值还是不一致，请[新建issue向MindSpore求助](https://gitee.com/mindspore/mindspore/issues/new)。

建议使用非下沉模式运行脚本，以得到脚本每个迭代的loss值，然后可以对前两个迭代的loss值进行对比。原因是下沉模式下一般只能得到每个epoch的loss值，由于一个epoch中经历的迭代数一般较多，随机性累积可能会使得两次运行的epoch粒度的loss值存在明显差距，无法作为随机性是否固定完毕的依据。

## 说明

1. 本文档主要适用于Ascend后端上`GRAPH_MODE`的训练脚本。
2. Ascend后端上使用atomic_write特性的算子列表：

    - [ReduceSum](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ReduceSum.html#mindspore.ops.ReduceSum)
    - [DynamicGRUV2](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.DynamicGRUV2.html#mindspore.ops.DynamicGRUV2)
    - [DynamicRNN](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.DynamicRNN.html#mindspore.ops.DynamicRNN)
    - [LayerNorm](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.LayerNorm.html#mindspore.ops.LayerNorm)
    - [NLLLoss](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.NLLLoss.html#mindspore.ops.NLLLoss)
