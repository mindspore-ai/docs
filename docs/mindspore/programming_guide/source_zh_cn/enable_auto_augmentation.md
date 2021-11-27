# 应用自动数据增强

`Ascend` `GPU` `CPU` `数据准备`

<!-- TOC -->

- [应用自动数据增强](#应用自动数据增强)
    - [概述](#概述)
    - [ImageNet自动数据增强](#imagenet自动数据增强)
    - [参考文献](#参考文献)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/enable_auto_augmentation.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>
&nbsp;&nbsp;
<a href="https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/master/notebook/mindspore_enable_auto_augmentation.ipynb"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_notebook.png"></a>
&nbsp;&nbsp;
<a href="https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL21pbmRzcG9yZV9lbmFibGVfYXV0b19hdWdtZW50YXRpb24uaXB5bmI=&imageid=65f636a0-56cf-49df-b941-7d2a07ba8c8c" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_modelarts.png"></a>
&nbsp;&nbsp;
<a href="https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/master/notebook/mindspore_enable_auto_augmentation.py"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_download_code.png"></a>

## 概述

自动数据增强（AutoAugment）[1]是在一系列图像增强子策略的搜索空间中，通过搜索算法找到适合特定数据集的图像增强方案。MindSpore的`c_transforms`模块提供了丰富的C++算子来实现AutoAugment，用户也可以自定义函数或者算子来实现。更多MindSpore算子的详细说明参见[API文档](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.dataset.vision.html)。

MindSpore算子和AutoAugment中的算子的对应关系如下：

| AutoAugment算子 | MindSpore算子 | 描述 |
| :------: | :------ | ------ |
| shearX | RandomAffine | 横向剪切 |
| shearY | RandomAffine | 纵向剪切 |
| translateX | RandomAffine | 水平平移 |
| translateY | RandomAffine | 垂直平移 |
| rotate | RandomRotation | 旋转变换 |
| color | RandomColor | 颜色变换 |
| posterize | RandomPosterize | 减少颜色通道位数 |
| solarize | RandomSolarize | 指定的阈值范围内，反转所有的像素点 |
| contrast | RandomColorAdjust | 调整对比度 |
| sharpness | RandomSharpness | 调整锐度 |
| brightness | RandomColorAdjust | 调整亮度 |
| autocontrast | AutoContrast | 最大化图像对比度 |
| equalize | Equalize | 均衡图像直方图 |
| invert | Invert | 反转图像 |

## ImageNet自动数据增强

本教程以在ImageNet数据集上实现AutoAugment作为示例。

针对ImageNet数据集的数据增强策略包含25条子策略，每条子策略中包含两种变换，针对一个batch中的每张图像随机挑选一个子策略的组合，以预定的概率来决定是否执行子策略中的每种变换。

用户可以使用MindSpore中`c_transforms`模块的`RandomSelectSubpolicy`接口来实现AutoAugment，在ImageNet分类训练中标准的数据增强方式分以下几个步骤：

- `RandomCropDecodeResize`：随机裁剪后进行解码。

- `RandomHorizontalFlip`：水平方向上随机翻转。

- `Normalize`：归一化。

- `HWC2CHW`：图片通道变化。

在`RandomCropDecodeResize`后插入AutoAugment变换，如下所示：

1. 引入MindSpore数据增强模块。

    ```python
    import matplotlib.pyplot as plt

    import mindspore.dataset as ds
    import mindspore.dataset.transforms.c_transforms as c_transforms
    import mindspore.dataset.vision.c_transforms as c_vision
    from mindspore import dtype as mstype
    ```

2. 定义MindSpore算子到AutoAugment算子的映射：

    ```python
    # define Auto Augmentation operators
    PARAMETER_MAX = 10

    def float_parameter(level, maxval):
        return float(level) * maxval /  PARAMETER_MAX

    def int_parameter(level, maxval):
        return int(level * maxval / PARAMETER_MAX)

    def shear_x(level):
        v = float_parameter(level, 0.3)
        return c_transforms.RandomChoice([c_vision.RandomAffine(degrees=0, shear=(-v,-v)), c_vision.RandomAffine(degrees=0, shear=(v, v))])

    def shear_y(level):
        v = float_parameter(level, 0.3)
        return c_transforms.RandomChoice([c_vision.RandomAffine(degrees=0, shear=(0, 0, -v,-v)), c_vision.RandomAffine(degrees=0, shear=(0, 0, v, v))])

    def translate_x(level):
        v = float_parameter(level, 150 / 331)
        return c_transforms.RandomChoice([c_vision.RandomAffine(degrees=0, translate=(-v,-v)), c_vision.RandomAffine(degrees=0, translate=(v, v))])

    def translate_y(level):
        v = float_parameter(level, 150 / 331)
        return c_transforms.RandomChoice([c_vision.RandomAffine(degrees=0, translate=(0, 0, -v,-v)), c_vision.RandomAffine(degrees=0, translate=(0, 0, v, v))])

    def color_impl(level):
        v = float_parameter(level, 1.8) + 0.1
        return c_vision.RandomColor(degrees=(v, v))

    def rotate_impl(level):
        v = int_parameter(level, 30)
        return c_transforms.RandomChoice([c_vision.RandomRotation(degrees=(-v, -v)), c_vision.RandomRotation(degrees=(v, v))])

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

3. 定义ImageNet数据集的AutoAugment策略：

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

4. 在`RandomCropDecodeResize`操作后插入AutoAugment变换。

    ```python
    def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, shuffle=True, num_samples=5, target="Ascend"):
        # create a train or eval imagenet2012 dataset for ResNet-50
        dataset = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8,
                                   shuffle=shuffle, num_samples=num_samples)

        image_size = 224
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

        # define map operations
        if do_train:
            trans = [
                c_vision.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            ]

            post_trans = [
                c_vision.RandomHorizontalFlip(prob=0.5),
            ]
        else:
            trans = [
                c_vision.Decode(),
                c_vision.Resize(256),
                c_vision.CenterCrop(image_size),
                c_vision.Normalize(mean=mean, std=std),
                c_vision.HWC2CHW()
            ]
        dataset = dataset.map(operations=trans, input_columns="image")
        if do_train:
            dataset = dataset.map(operations=c_vision.RandomSelectSubpolicy(imagenet_policy), input_columns=["image"])
            dataset = dataset.map(operations=post_trans, input_columns="image")
        type_cast_op = c_transforms.TypeCast(mstype.int32)
        dataset = dataset.map(operations=type_cast_op, input_columns="label")
        # apply the batch operation
        dataset = dataset.batch(batch_size, drop_remainder=True)
        # apply the repeat operation
        dataset = dataset.repeat(repeat_num)

        return dataset
    ```

5. 验证自动数据增强效果。

    ```python
    # Define the path to image folder directory. This directory needs to contain sub-directories which contain the images
    DATA_DIR = "/path/to/image_folder_directory"
    dataset = create_dataset(dataset_path=DATA_DIR, do_train=True, batch_size=5, shuffle=False, num_samples=5)

    epochs = 5
    itr = dataset.create_dict_iterator()
    fig=plt.figure(figsize=(8, 8))
    columns = 5
    rows = 5

    step_num = 0
    for ep_num in range(epochs):
        for data in itr:
            step_num += 1
            for index in range(rows):
                fig.add_subplot(rows, columns, ep_num * rows + index + 1)
                plt.imshow(data['image'].asnumpy()[index])
    plt.show()
    ```

    > 为了更好地演示效果，此处只加载5张图片，且读取时不进行`shuffle`操作，自动数据增强时也不进行`Normalize`和`HWC2CHW`操作。

    ![augment](./images/auto_augmentation.png)

    运行结果可以看到，batch中每张图像的增强效果，水平方向表示1个batch的5张图像，垂直方向表示5个batch。

## 参考文献

[1] [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501).
