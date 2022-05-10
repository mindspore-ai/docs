# Auto Augmentation

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/dataset/augment.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore not only allows you to customize data augmentation, but also provides an auto augmentation method to automatically perform data augmentation on images based on specific policies.

Auto augmentation can be implemented based on **probability** or **callback parameters**.

## Probability-Based Auto Augmentation

MindSpore provides a series of probability-based auto augmentation APIs. You can randomly select and combine various data augmentation operations to make data augmentation more flexible.

### RandomApply

The `RandomApply` receives a data augmentation operation list and executes the data augmentation operations in the list in sequence at a certain probability or executes none of them. The default probability is 0.5.

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

In the following code example, two sub-policies are preset.

- Sub-policy 1 contains the `RandomRotation` and `RandomVerticalFlip`operations, whose probabilities are 0.5 and 1.0, respectively.

- Sub-policy 2 contains the `RandomRotation` and `RandomColorAdjust` operations, with the probabilities of 1.0 and 0.2, respectively.

```python
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore.dataset.vision.c_transforms import RandomSelectSubpolicy

policy_list = [
      [(c_vision.RandomRotation((45, 45)), 0.5), (c_vision.RandomVerticalFlip(), 1.0), (c_vision.RandomColorAdjust(), 0.8)],
      [(c_vision.RandomRotation((90, 90)), 1.0), (c_vision.RandomColorAdjust(), 0.2)]
      ]
policy = RandomSelectSubpolicy(policy_list)
```

## Callback Parameter-based Auto Augmentation

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

## ImageNet Automatic Data Augmentation

The following is an example of implementing AutoAugment on an ImageNet dataset.

The data augmentation policy for the ImageNet dataset contains 25 sub-strategies, each of which contains two transformations. A combination of sub-strategies is randomly selected for each image in a batch, and each transformation in the sub-strategy is determined by predetermined probability.

Users can use the `RandomSelectSubpolicy` interface of the `c_transforms` module in MindSpore to implement AutoAugment, and the standard data augmentation method in ImageNet classification training is divided into the following steps:

- `RandomCropDecodeResize`: Decoding after random cropping.
- `RandomHorizontalFlip`: Flipping randomly horizontally.
- `Normalize`: Normalization.
- `HWC2CHW`: Changing picture channel.

1. Define the mapping of the MindSpore operator to the AutoAugment operator:

    ```python
    import mindspore.dataset.vision.c_transforms as c_vision
    import mindspore.dataset.transforms.c_transforms as c_transforms

    # define Auto Augmentation operators
    PARAMETER_MAX = 10

    def float_parameter(level, maxval):
        return float(level) * maxval /  PARAMETER_MAX

    def int_parameter(level, maxval):
        return int(level * maxval / PARAMETER_MAX)

    def shear_x(level):
        transforms_list = []
        v = float_parameter(level, 0.3)

        transforms_list.append(c_vision.RandomAffine(degrees=0, shear=(-v, -v)))
        transforms_list.append(c_vision.RandomAffine(degrees=0, shear=(v, v)))
        return c_transforms.RandomChoice(transforms_list)

    def shear_y(level):
        transforms_list = []
        v = float_parameter(level, 0.3)

        transforms_list.append(c_vision.RandomAffine(degrees=0, shear=(0, 0, -v, -v)))
        transforms_list.append(c_vision.RandomAffine(degrees=0, shear=(0, 0, v, v)))
        return c_transforms.RandomChoice()

    def translate_x(level):
        transforms_list = []
        v = float_parameter(level, 150 / 331)

        transforms_list.append(c_vision.RandomAffine(degrees=0, translate=(-v, -v)))
        transforms_list.append(c_vision.RandomAffine(degrees=0, translate=(v, v)))
        return c_transforms.RandomChoice()

    def translate_y(level):
        transforms_list = []
        v = float_parameter(level, 150 / 331)

        transforms_list.append(c_vision.RandomAffine(degrees=0, translate=(0, 0, -v, -v)))
        transforms_list.append(c_vision.RandomAffine(degrees=0, translate=(0, 0, v, v)))
        return c_transforms.RandomChoice()

    def color_impl(level):
        v = float_parameter(level, 1.8) + 0.1
        return c_vision.RandomColor(degrees=(v, v))

    def rotate_impl(level):
        transforms_list = []
        v = int_parameter(level, 30)

        transforms_list.append(c_vision.RandomRotation(degrees=(-v, -v)))
        transforms_list.append(c_vision.RandomRotation(degrees=(v, v)))
        return c_transforms.RandomChoice()

    def solarize_impl(level):
        level = int_parameter(level, 256)
        v = 256 - level
        return c_vision.RandomSolarize(threshold=(0, v))

    def posterize_impl(level):
        level = int_parameter(level, 4)
        v = 4 - level
        return c_vision.RandomPosterize(bits=(v, v))

    def contrast_impl(level):
        v = float_parameter(level, 1.8) + 0.1
        return c_vision.RandomColorAdjust(contrast=(v, v))

    def autocontrast_impl(level):
        return c_vision.AutoContrast()

    def sharpness_impl(level):
        v = float_parameter(level, 1.8) + 0.1
        return c_vision.RandomSharpness(degrees=(v, v))

    def brightness_impl(level):
        v = float_parameter(level, 1.8) + 0.1
        return c_vision.RandomColorAdjust(brightness=(v, v))
    ```

2. Define the AutoAugment policy for the ImageNet dataset:

    ```python
    # define the Auto Augmentation policy
    imagenet_policy = [
        [(posterize_impl(8), 0.4), (rotate_impl(9), 0.6)],
        [(solarize_impl(5), 0.6), (autocontrast_impl(5), 0.6)],
        [(c_vision.Equalize(), 0.8), (c_vision.Equalize(), 0.6)],
        [(posterize_impl(7), 0.6), (posterize_impl(6), 0.6)],

        [(c_vision.Equalize(), 0.4), (solarize_impl(4), 0.2)],
        [(c_vision.Equalize(), 0.4), (rotate_impl(8), 0.8)],
        [(solarize_impl(3), 0.6), (c_vision.Equalize(), 0.6)],
        [(posterize_impl(5), 0.8), (c_vision.Equalize(), 1.0)],
        [(rotate_impl(3), 0.2), (solarize_impl(8), 0.6)],
        [(c_vision.Equalize(), 0.6), (posterize_impl(6), 0.4)],

        [(rotate_impl(8), 0.8), (color_impl(0), 0.4)],
        [(rotate_impl(9), 0.4), (c_vision.Equalize(), 0.6)],
        [(c_vision.Equalize(), 0.0), (c_vision.Equalize(), 0.8)],
        [(c_vision.Invert(), 0.6), (c_vision.Equalize(), 1.0)],
        [(color_impl(4), 0.6), (contrast_impl(8), 1.0)],

        [(rotate_impl(8), 0.8), (color_impl(2), 1.0)],
        [(color_impl(8), 0.8), (solarize_impl(7), 0.8)],
        [(sharpness_impl(7), 0.4), (c_vision.Invert(), 0.6)],
        [(shear_x(5), 0.6), (c_vision.Equalize(), 1.0)],
        [(color_impl(0), 0.4), (c_vision.Equalize(), 0.6)],

        [(c_vision.Equalize(), 0.4), (solarize_impl(4), 0.2)],
        [(solarize_impl(5), 0.6), (autocontrast_impl(5), 0.6)],
        [(c_vision.Invert(), 0.6), (c_vision.Equalize(), 1.0)],
        [(color_impl(4), 0.6), (contrast_impl(8), 1.0)],
        [(c_vision.Equalize(), 0.8), (c_vision.Equalize(), 0.6)],
    ]
    ```

3. Insert the AutoAugment transform after the `RandomCropDecodeResize` operation.

    ```python
    import mindspore.dataset as ds
    from mindspore import dtype as mstype


    def create_dataset(dataset_path, train, repeat_num=1,
                       batch_size=32, shuffle=True, num_samples=5):
        # create a train or eval imagenet2012 dataset for ResNet-50
        dataset = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8,
                                        shuffle=shuffle, num_samples=num_samples)

        image_size = 224
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

        # define map operations
        if train:
            trans = imagenet_policy
        else:
            trans = [c_vision.Decode(),
                     c_vision.Resize(256),
                     c_vision.CenterCrop(image_size),
                     c_vision.Normalize(mean=mean, std=std),
                     c_vision.HWC2CHW()]
            type_cast_op = c_transforms.TypeCast(mstype.int32)

        # map images and labes
        dataset = dataset.map(operations=trans, input_columns="image")
        dataset = dataset.map(operations=type_cast_op, input_columns="label")

        # apply the batch and repeat operation
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.repeat(repeat_num)

        return dataset
    ```

4. Verify automatic data augmentations:

    ```python
    import matplotlib.pyplot as plt

    # Define the path to image folder directory.
    DATA_DIR = "/path/to/image_folder_directory"
    dataset = create_dataset(dataset_path=DATA_DIR,
                             train=True,
                             batch_size=5,
                             shuffle=False,
                             num_samples=5)

    epochs = 5
    columns = 5
    rows = 5
    step_num = 0
    fig = plt.figure(figsize=(8, 8))
    itr = dataset.create_dict_iterator()

    for ep_num in range(epochs):
        for data in itr:
            step_num += 1
            for index in range(rows):
                fig.add_subplot(rows, columns, ep_num * rows + index + 1)
                plt.imshow(data['image'].asnumpy()[index])
    plt.show()
    ```

> For a better demonstration of the effect, only 5 images are loaded here, and no `shuffle` operation is performed when reading, nor `Normalize` and `HWC2CHW` operations are performed when automatic data augmentation is performed.

![augment](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/experts/source_en/dataset/images/auto_augmentation.png)

The running result can be seen that the augmentation effect of each image in the batch, the horizontal direction represents 5 images of 1 batch, and the vertical direction represents 5 batches.

## References

[1] [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501).

