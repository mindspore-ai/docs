# 数据集切分

`Ascend` `分布式并行`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/design/dataset_slice.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

在进行分布式训练时，以图片数据为例，当单张图片的大小过大时，如遥感卫星等大幅面图片，即使一张图片也过大，需要对图片进行切分，每张卡读取一部分图片，
进行分布式训练。处理数据集切分的场景，需要配合模型并行一起才能达到预期的降低显存的效果，因此，基于自动并行提供了该项功能。本教程使用的样例为ResNet50，不是大幅面的网络，仅作示例。真实应用到大幅面的网络时，往往需要详细设计并行策略。

## 操作实践

### 样例代码说明

>你可以在这里下载完整的样例代码：
>
>https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training

目录结构如下：

```text
└─sample_code
    ├─distributed_training
    │      rank_table_16pcs.json
    │      rank_table_8pcs.json
    │      rank_table_2pcs.json
    │      resnet.py
    │      resnet50_distributed_training_dataset_slice.py
    │      run_dataset_slice.sh
```

### 创建数据集

> 数据集切分仅支持全/半自动模式，在数据并行模式下不涉及。

使用数据集切分时，需要同时调用数据集的[SlicePatches](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.SlicePatches.html)接口去构造数据集，并且，为了保证各卡读入数据一致，需要对数据集固定随机数种子。

数据集定义部分如下。

```python
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
import mindspore.dataset.transforms.c_transforms as C
from mindspore.communication import init, get_rank, get_group_size
from mindspore import context
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
init()
ds.config.set_seed(1000) # set dataset seed to make sure that all cards read the same data
def create_dataset(data_path, repeat_num=1, batch_size=32, slice_h_num=1, slice_w_num=1):
    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    rank_id = get_rank()

    # create a full dataset before slicing
    data_set = ds.Cifar10Dataset(data_path, shuffle=True)

    # define map operations
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize((resize_height, resize_width))
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op]

    # apply map operations on images
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    # in random map function, using num_parallel_workers=1 to avoid the dataset random seed not working.
    data_set = data_set.map(operations=c_trans, input_columns="image", num_parallel_workers=1)
    # slice image
    slice_patchs_img_op = vision.SlicePatchs(slice_h_num, slice_w_num)
    img_cols = ['img' + str(x) for x in range(slice_h_num * slice_w_num)]
    data_set = data_set.map(operations=slice_patchs_img_op, input_columns="image", output_columns=img_cols,
                            column_order=[img_cols[rank_id % (slice_h_num * slice_w_num)], "label"])
    # change hwc to chw
    data_set = data_set.map(operations=changeswap_op, input_columns=img_cols[rank_id % (slice_h_num * slice_w_num)])
    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)

    return data_set
```

### 配置数据集切分策略

> 数据集切分仅支持全/半自动模式，在数据并行模式下不涉及。

在`mindspore.context.auto_parallel_context`中提供了`dataset_strategy`选项，为数据集配置切分策略。

dataset_strategy接口还有以下几点限制：

1. 每个输入至多允许在一维进行切分。如支持`context.set_auto_parallel_context(dataset_strategy=((1, 1, 1, 8), (8,))))`或者`dataset_strategy=((1, 1, 1, 8), (1,)))`，每个输入都仅仅切分了一维；但是不支持`dataset_strategy=((1, 1, 4, 2), (1,))`，其第一个输入切分了两维。

2. 维度最高的一个输入，切分的数目，一定要比其他维度的多。如支持`dataset_strategy=((1, 1, 1, 8), (8,)))`或者`dataset_strategy=((1, 1, 1, 1), (1,)))`，其维度最多的输入为第一个输入，切分份数为8，其余输入切分均不超过8；但是不支持`dataset_strategy=((1, 1, 1, 1), (8,))`，其维度最多的输入为第一维，切分份数为1，但是其第二个输入切分份数却为8，超过了第一个输入的切分份数。

```python
import os
from mindspore.context import ParallelMode
from mindspore import context
context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
slice_h_num = 1
slice_w_num = 8
batch_size = 256
context.set_auto_parallel_context(dataset_strategy=(((1, 1, slice_h_num, slice_w_num), (1,))))
data_path = os.getenv('DATA_PATH')
dataset = create_dataset(data_path, batch_size=batch_size, slice_h_num=slice_h_num, slice_w_num=slice_w_num)
```

### 运行代码

上述流程的数据，代码和执行过程，可以参考：<https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_ascend.html#单机多卡训练>。差异点在于，将执行脚本更改为run_dataset_slice.sh。