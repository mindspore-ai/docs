# 自动数据增强

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/programming_guide/source_zh_cn/auto_augmentation.md)

## 概述

MindSpore除了可以让用户自定义数据增强的使用，还提供了一种自动数据增强方式，可以基于特定策略自动对图像进行数据增强处理。

自动数据增强主要分为基于概率的自动数据增强和基于回调参数的自动数据增强。

## 基于概率的自动数据增强

MindSpore提供了一系列基于概率的自动数据增强API，用户可以对各种数据增强操作进行随机选择与组合，使数据增强更加灵活。

关于API的详细说明，可以参见[API文档](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.dataset.transforms.html)。

### RandomApply

API接收一个数据增强操作列表`transforms`，以一定的概率顺序执行列表中各数据增强操作，默认概率为0.5，否则都不执行。

在下面的代码示例中，以0.5的概率来顺序执行`RandomCrop`和`RandomColorAdjust`操作，否则都不执行。

```python
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore.dataset.transforms.c_transforms import RandomApply

rand_apply_list = RandomApply([c_vision.RandomCrop(512), c_vision.RandomColorAdjust()])
```

### RandomChoice

API接收一个数据增强操作列表`transforms`，从中随机选择一个数据增强操作执行。

在下面的代码示例中，等概率地在`CenterCrop`和`RandomCrop`中选择一个操作执行。

```python
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore.dataset.transforms.c_transforms import RandomChoice

rand_choice = RandomChoice([c_vision.CenterCrop(512), c_vision.RandomCrop(512)])
```

### RandomSelectSubpolicy

API接收一个预置策略列表，包含一系列子策略组合，每一子策略由若干个顺序执行的数据增强操作及其执行概率组成。

对各图像先等概率随机选择一种子策略，再依照子策略中的概率顺序执行各个操作。

在下面的代码示例中，预置了两条子策略，子策略1中包含`RandomRotation`、`RandomVerticalFlip`和`RandomColorAdjust`三个操作，概率分别为0.5、1.0和0.8；子策略2中包含`RandomRotation`和`RandomColorAdjust`两个操作，概率分别为1.0和0.2。

```python
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore.dataset.vision.c_transforms import RandomSelectSubpolicy

policy_list = [
      [(c_vision.RandomRotation((45, 45)), 0.5), (c_vision.RandomVerticalFlip(), 1.0), (c_vision.RandomColorAdjust(), 0.8)],
      [(c_vision.RandomRotation((90, 90)), 1.0), (c_vision.RandomColorAdjust(), 0.2)]
      ]
policy = RandomSelectSubpolicy(policy_list)
```

## 基于回调参数的自动数据增强

MindSpore的`sync_wait`接口支持按batch或epoch粒度在训练过程中动态调整数据增强策略，用户可以设定阻塞条件触发特定的数据增强操作。

`sync_wait`将阻塞整个数据处理pipeline直到`sync_update`触发用户预先定义的`callback`函数，两者需配合使用，对应说明如下：

- sync_wait(condition_name, num_batch=1, callback=None)

    该API为数据集添加一个阻塞条件`condition_name`，当`sync_update`调用时执行指定的`callback`函数。

- sync_update(condition_name, num_batch=None, data=None)

    该API用于释放对应`condition_name`的阻塞，并对`data`触发指定的`callback`函数。

下面将演示基于回调参数的自动数据增强的用法。

1. 用户预先定义`Augment`类，其中`preprocess`为自定义的数据增强函数，`update`为更新数据增强策略的回调函数。

    ```python
    import mindspore.dataset.vision.py_transforms as transforms
    import mindspore.dataset as ds
    import numpy as np

    class Augment:
        def __init__(self):
            self.ep_num = 0
            self.step_num = 0

        def preprocess(self, input_):
            return (np.array((input_ + self.step_num ** self.ep_num - 1), ))

        def update(self, data):
            self.ep_num = data['ep_num']
            self.step_num = data['step_num']
    ```

2. 数据处理pipeline先回调自定义的增强策略更新函数`update`，然后在`map`操作中按更新后的策略来执行`preprocess`中定义的数据增强操作。

    ```python
    arr = list(range(1, 4))
    dataset = ds.NumpySlicesDataset(arr, shuffle=False)
    aug = Augment()
    dataset = dataset.sync_wait(condition_name="policy", callback=aug.update)
    dataset = dataset.map(operations=[aug.preprocess])
    ```

3. 在每个step中调用`sync_update`进行数据增强策略的更新。

    ```python
    epochs = 5
    itr = dataset.create_tuple_iterator(num_epochs=epochs)
    step_num = 0
    for ep_num in range(epochs):
        for data in itr:
            print("epcoh: {}, step:{}, data :{}".format(ep_num, step_num, data))
            step_num += 1
            dataset.sync_update(condition_name="policy", data={'ep_num': ep_num, 'step_num': step_num})
    ```

    输出结果如下：

    ```text
    epcoh: 0, step:0, data :[Tensor(shape=[], dtype=Int64, value= 1)]
    epcoh: 0, step:1, data :[Tensor(shape=[], dtype=Int64, value= 2)]
    epcoh: 0, step:2, data :[Tensor(shape=[], dtype=Int64, value= 3)]
    epcoh: 1, step:3, data :[Tensor(shape=[], dtype=Int64, value= 1)]
    epcoh: 1, step:4, data :[Tensor(shape=[], dtype=Int64, value= 5)]
    epcoh: 1, step:5, data :[Tensor(shape=[], dtype=Int64, value= 7)]
    epcoh: 2, step:6, data :[Tensor(shape=[], dtype=Int64, value= 6)]
    epcoh: 2, step:7, data :[Tensor(shape=[], dtype=Int64, value= 50)]
    epcoh: 2, step:8, data :[Tensor(shape=[], dtype=Int64, value= 66)]
    epcoh: 3, step:9, data :[Tensor(shape=[], dtype=Int64, value= 81)]
    epcoh: 3, step:10, data :[Tensor(shape=[], dtype=Int64, value= 1001)]
    epcoh: 3, step:11, data :[Tensor(shape=[], dtype=Int64, value= 1333)]
    epcoh: 4, step:12, data :[Tensor(shape=[], dtype=Int64, value= 1728)]
    epcoh: 4, step:13, data :[Tensor(shape=[], dtype=Int64, value= 28562)]
    epcoh: 4, step:14, data :[Tensor(shape=[], dtype=Int64, value= 38418)]
    ```
