# Dataset Slicing

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/dataset_slice.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

When performing distributed training, taking image data as an example, when the size of a single image is too large, such as large-format images of remote sensing satellites, even one image is too large, it is necessary to slice the images and read a portion of each card to perform distributed training. Scenarios that deal with dataset slicing need to be combined with model parallelism to achieve the desired effect of reducing video memory, so this feature is provided based on automatic parallelism. The sample used in this tutorial is ResNet50, not a large-format network, and is intended as an example only. Real-life applications to large-format networks often require detailed design of parallel strategies.

## Operation Practices

### Sample Code Description

> You can download the full sample code here:
>
> <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training>

The directory structure is as follows:

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

### Creating the Dataset

> Dataset slicing is only supported in full/semi-automatic mode and is not involved in data parallel mode.

When using dataset slicing, you need to call the [SlicePatches](https://www.mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.SlicePatches.html) interface to construct the dataset at the same time. To ensure that the read-in data is consistent across cards, the dataset needs to be fixed with a random number seed.

The dataset definition section is as follows.

```python
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.communication import init, get_rank, get_group_size

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
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
    type_cast_op = transforms.TypeCast(ms.int32)

    c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op]

    # apply map operations on images
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    # in random map function, using num_parallel_workers=1 to avoid the dataset random seed not working.
    data_set = data_set.map(operations=c_trans, input_columns="image", num_parallel_workers=1)
    # slice image
    slice_patchs_img_op = vision.SlicePatchs(slice_h_num, slice_w_num)
    img_cols = ['img' + str(x) for x in range(slice_h_num * slice_w_num)]
    data_set = data_set.map(operations=slice_patchs_img_op, input_columns="image", output_columns=img_cols)
    data_set = data_set.project([img_cols[rank_id % (slice_h_num * slice_w_num)], "label"])
    # change hwc to chw
    data_set = data_set.map(operations=changeswap_op, input_columns=img_cols[rank_id % (slice_h_num * slice_w_num)])
    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)

    return data_set
```

### Configuring Dataset Slicing Strategy

> Dataset slicing is only supported in full/semi-automatic mode and is not involved in data parallel mode.

The `dataset_strategy` option is provided in `mindspore.auto_parallel_context` to configure the slicing strategy for the dataset.

The dataset_strategy interface also has the following limitations:

1. Each input is allowed to be sliced in at most one dimension. If support `set_auto_parallel_context(dataset_strategy=((1, 1, 1, 8), (8,))))` or `dataset_strategy=((1, 1, 1, 8), (1,)))`, each input is sliced to just one dimension, but does not support `dataset_strategy=((1, 1, 4, 2), (1,))`, whose first input is sliced to two dimensions.

2. The number of slices for one input with the highest dimension, must be more than the other dimensions. If support `dataset_strategy=((1, 1, 1, 8), (8,)))` or `dataset_strategy=((1, 1, 1, 1, 1), (1,)))` is supported, the input with the most dimensions is the first input, the number of slices is 8, and the rest of the inputs are sliced by no more than 8 parts. However, it does not support `dataset_strategy=((1, 1, 1, 1), (8,)`, whose input with the most dimensions is the first dimension and the number of slices is 1, but the number of slices of second input is 8, which exceeds the number of slices of the first input.

```python
import os
import mindspore as ms
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, gradients_mean=True)
slice_h_num = 1
slice_w_num = 8
batch_size = 256
ms.set_auto_parallel_context(dataset_strategy=(((1, 1, slice_h_num, slice_w_num), (1,))))
data_path = os.getenv('DATA_PATH')
dataset = create_dataset(data_path, batch_size=batch_size, slice_h_num=slice_h_num, slice_w_num=slice_w_num)
```

### Running the Code

The data, code and execution of the above process can be found at: <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#single-host-training>. The difference is that the execution script is changed to run_dataset_slice.sh.
