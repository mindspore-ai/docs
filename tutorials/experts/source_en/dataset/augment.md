# Auto Augmentation

<a href="https://gitee.com/mindspore/docs/blob/r1.7/tutorials/experts/source_en/dataset/augment.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore not only allows you to customize data augmentation, but also provides an auto augmentation method to automatically perform data augmentation on images based on specific policies.

Auto augmentation can be implemented based on probability or callback parameters.

## Probability Based Auto Augmentation

MindSpore provides a series of probability-based auto augmentation APIs. You can randomly select and combine various data augmentation operations to make data augmentation more flexible.

For details about APIs, see [MindSpore API](https://www.mindspore.cn/docs/en/r1.7/api_python/mindspore.dataset.transforms.html).

### RandomApply

The API receives a data augmentation operation list `transforms` and executes the data augmentation operations in the list in sequence at a certain probability or executes none of them. The default probability is 0.5.

In the following code example, the `RandomCrop` and `RandomColorAdjust` operations are executed in sequence with a probability of 0.5 or none of them are executed.

```python
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore.dataset.transforms.c_transforms import RandomApply

rand_apply_list = RandomApply([c_vision.RandomCrop(512), c_vision.RandomColorAdjust()])
```

### RandomChoice

The API receives a data augmentation operation list `transforms` and randomly selects a data augmentation operation to perform.

In the following code example, an operation is selected from `CenterCrop` and `RandomCrop` for execution with equal probability.

```python
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore.dataset.transforms.c_transforms import RandomChoice

rand_choice = RandomChoice([c_vision.CenterCrop(512), c_vision.RandomCrop(512)])
```

### RandomSelectSubpolicy

The API receives a preset policy list, including a series of sub-policy combinations. Each sub-policy consists of several data augmentation operations executed in sequence and their execution probabilities.

First, a sub-policy is randomly selected for each image with equal probability, and then operations are performed according to the probability sequence in the sub-policy.

In the following code example, two sub-policies are preset. Sub-policy 1 contains the `RandomRotation`, `RandomVerticalFlip`, and `RandomColorAdjust` operations, whose probabilities are 0.5, 1.0, and 0.8, respectively. Sub-policy 2 contains the `RandomRotation` and `RandomColorAdjust` operations, with the probabilities of 1.0 and 0.2, respectively.

```python
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore.dataset.vision.c_transforms import RandomSelectSubpolicy

policy_list = [
      [(c_vision.RandomRotation((45, 45)), 0.5), (c_vision.RandomVerticalFlip(), 1.0), (c_vision.RandomColorAdjust(), 0.8)],
      [(c_vision.RandomRotation((90, 90)), 1.0), (c_vision.RandomColorAdjust(), 0.2)]
      ]
policy = RandomSelectSubpolicy(policy_list)
```

## Callback Parameter based Auto Augmentation

The `sync_wait` API of MindSpore supports dynamic adjustment of the data augmentation policy by batch or epoch granularity during training. You can set blocking conditions to trigger specific data augmentation operations.

`sync_wait` blocks the entire data processing pipeline until `sync_update` triggers the customized `callback` function. The two APIs must be used together. Their descriptions are as follows:

- sync_wait(condition_name, num_batch=1, callback=None)

    This API adds a blocking condition `condition_name` to a dataset. When `sync_update` is called, the specified `callback` function is executed.

- sync_update(condition_name, num_batch=None, data=None)

    This API releases the block corresponding to `condition_name` and triggers the specified `callback` function for `data`.

The following demonstrates the use of automatic data augmentation based on callback parameters.

1. Customize the `Augment` class where `preprocess` is a custom data augmentation function and `update` is a callback function for updating the data augmentation policy.

    ```python
    import mindspore.dataset as ds
    import numpy as np

    class Augment:
        def __init__(self):
            self.ep_num = 0
            self.step_num = 0

        def preprocess(self, input_):
            return np.array((input_ + self.step_num ** self.ep_num - 1), )

        def update(self, data):
            self.ep_num = data['ep_num']
            self.step_num = data['step_num']
    ```

2. The data processing pipeline calls back the custom data augmentation policy update function `update`, and then performs the data augmentation operation defined in `preprocess` based on the updated policy in the `map` operation.

    ```python
    arr = list(range(1, 4))
    dataset = ds.NumpySlicesDataset(arr, shuffle=False)
    aug = Augment()
    dataset = dataset.sync_wait(condition_name="policy", callback=aug.update)
    dataset = dataset.map(operations=[aug.preprocess])
    ```

3. Call `sync_update` in each step to update the data augmentation policy.

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

    The output is as follows:

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
