# 自动数据增强

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [自动数据增强](#自动数据增强)
	- [基于概率动态调整数据增强策略](#基于概率动态调整数据增强策略)
	- [基于训练结果信息动态调整数据增强策略](#基于训练结果信息动态调整数据增强策略)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/api/source_zh_cn/programming_guide/auto_augment.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 基于概率动态调整数据增强策略

MindSpore提供一系列基于概率的数据增强的API，用户可以对各种图像操作进行随机选择、组合，使数据增强更灵活。

- [`RandomApply(transforms, prob=0.5)`](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.dataset.transforms.html?highlight=randomapply#mindspore.dataset.transforms.c_transforms.RandomApply)
以一定的概率指定`transforms`操作，即可能执行，也可以能不执行；`transform`可以是一个，也可以是一系列。

  ```python

  rand_apply_list = RandomApply([c_vision.RandomCrop(), c_vision.RandomColorAdjust()])
  ds = ds.map(operations=rand_apply_list)

  ```

  按50%的概率来顺序执行`RandomCrop`和`RandomColorAdjust`操作,否则都不执行。

- [`RandomChoice(transforms)`](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.dataset.transforms.html?highlight=randomchoice#mindspore.dataset.transforms.c_transforms.RandomChoice)
从`transfrom`操作中随机选择一个执行。

  ```python

  rand_choice = RandomChoice([c_vision.CenterCrop(), c_vision.RandomCrop()])
  ds = ds.map(operations=rand_choice)

  ```

  分别以50%概率来执行`CenterCrop`和`RandomCrop`操作。

- [`RandomSelectSubpolicy(policy)`](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.dataset.transforms.vision.html?highlight=randomselectsubpolicy#mindspore.dataset.transforms.vision.c_transforms.RandomSelectSubpolicy)
用户可以预置策略（Policy），每次随机选择一个子策略（SubPolicy）组合，同一子策略中由若干个顺序执行的图像增强操作组成，每个操作与两个参数关联：1） 执行操作的概率 2）执行操作的幅度；
对于一个batch中的每张图像，随机选择子策略来变换图像。

  ```python

  policy = [
          [(c_vision.RandomRotation((45, 45)), 0.5), (c_vision.RandomVerticalFlip(), 1.0), (c_vision.RandomColorAdjust(), 0.8)],
          [(c_vision.RandomRotation((90, 90)), 1), (c_vision.RandomColorAdjust(), 0.2)]
          ]
  ds = ds.map(operations=c_vision.RandomSelectSubpolicy(policy), input_columns=["image"])

  ```

  示例中包括2条子策略，其中子策略1中包含`RandomRotation`、`RandomVerticalFlip`、`RandomColorAdjust`3个操作，概率分别为0.5、1.0、0.8；子策略2中包含`RandomRotation`和`RandomColorAdjust`，概率分别为1.0、2.0。

## 基于训练结果信息动态调整数据增强策略

Mindspore的`sync_wait`接口支持按batch或epoch粒度来调整数据增强策略，实现训练过程中动态调整数据增强策略。
`sync_wait`必须和`sync_update`配合使用实现数据pipeline上的同步回调。

- [`sync_wait(condition_name, num_batch=1, callback=None)`](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.dataset.html?highlight=sync_wait#mindspore.dataset.ImageFolderDatasetV2.sync_wait)
- [`sync_update(condition_name, num_batch=None, data=None)`](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.dataset.html?highlight=sync_update#mindspore.dataset.ImageFolderDatasetV2.sync_update)

`sync_wait`将阻塞整个数据处理pipeline直到`sync_update`触发用户预先定义的`callback`函数。

1. 用户预先定义class`Augment`，其中`preprocess`为`map`操作中的自定义数据增强函数，`update`为更新数据增强策略的回调函数。

  ```python
  import mindspore.dataset.transforms.vision.py_transforms as transforms
  import mindspore.dataset as de
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

2. 数据处理pipeline先回调自定义的增强策略更新函数`auto_aug.update`，然后在`map`操作中按更新后的策略来执行`auto_aug.preprocess`中定义的数据增强。

  ```python

  arr = list(range(1, 4))
  ds = de.NumpySlicesDataset(arr, shuffle=False)
  aug = Augment()
  ds= ds.sync_wait(condition_name="policy", callback=aug.update)
  ds = ds.map(operations=[aug.preprocess])

  ```

3. 在每个step调用`sync_update`进行数据增强策略的更新。

  ```python
  epochs = 5
  itr = ds.create_tuple_iterator(num_epochs=epochs)
  step_num = 0
  for ep_num in range(epochs):
      for data in itr:
          print("epcoh: {}, step:{}, data :{}".format(ep_num, step_num, data))
          step_num += 1
          ds.sync_update(condition_name="policy", data={'ep_num': ep_num, 'step_num': step_num})

  ```
