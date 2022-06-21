# Enabling Heterogeneous Acceleration for Data

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/dataset/dataset_offload.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore provides a computing load balancing technology which can distribute the MindSpore operators to different heterogeneous hardware. On one hand, it balances the computing overhead between different hardware, on the other, it uses the advantages of heterogeneous hardware to accelerate the operators.

Currently this heterogeneous hardware acceleration technology (introduced as the offload feature in the following sections) only supports moving dataset operations from the dataset pipeline to the computation graph, which balances the computation overhead between the data processing and the model training. To be specific, the dataset operators are currently executed on CPU, by using the offload feature, some dataset operators can be moved to the network in order to fully use GPU or Ascend for a better computation performance.

The offload feature will move only the supported dataset operations applied on the specific input column at the end of the pipeline to the accelerator. This includes consecutive data augmentation operators which are used in the map data processing operator, granted they come at the end of the dataset pipeline for a specific input column.

The current supported data augmentation operators which can perform heterogeneous acceleration are:

| Operator Name        | Operator Path                              | Operator Introduction                                        |
| -------------------- | -------------------------------------------| ------------------------------------------------------------ |
| HWC2CHW              | mindspore.dataset.vision.transforms.py     | Transpose a Numpy image array from shape (H, W, C) to shape (C, H, W) |
| Normalize            | mindspore.dataset.vision.transforms.py     | Normalize the image                                          |
| RandomColorAdjust    | mindspore.dataset.vision.transforms.py     | Perform a random brightness, contrast, saturation, and hue adjustment on the input PIL image |
| RandomHorizontalFlip | mindspore.dataset.vision.transforms.py     | Randomly flip the input image                                |
| RandomSharpness      | mindspore.dataset.vision.transforms.py     | Adjust the sharpness of the input PIL Image by a random degree |
| RandomVerticalFlip   | mindspore.dataset.vision.transforms.py     | Randomly flip the input image vertically with a given probability |
| Rescale              | mindspore.dataset.vision.transforms.py     | Rescale the input image with the given rescale and shift     |
| TypeCast             | mindspore.dataset.transforms.transforms.py | Cast tensor to a given MindSpore data type                   |

## Offload Process

The following figures show the typical computation process of how to use heterogeneous acceleration in the given dataset pipeline.

![offload](images/offload_process.PNG)

Heterogeneous acceleration has two new API changes to let users enable this functionality:

1. Map data operator adds offload input parameter.

2. Dataset global configuration of mindspore.dataset.config added set_auto_offload interface.

To check if the data augmentation operators are moved to the accelerator, users can save and check the computation graph IR files which will have the related operators written before the model structure. The heterogeneous acceleration is currently available for both dataset sink mode (dataset_sink_mode=True) and dataset non-sink mode (dataset_sink_mode=False).

## Enabling Heterogeneous Acceleration by Using Data

There are two options provided by MindSpore to enable heterogeneous acceleration.  

### Option 1

Use the global config to set automatic heterogeneous acceleration. In this case, the offload argument for all map data processing operators will be set to True (see Option 2). It should be noted that the offload argument is given for a specific map operator, it will have priority over the global config option.

```python
import mindspore.dataset as ds
ds.config.set_auto_offload(True)
```

### Option 2

Set the argument offload to True in the map data processing operator (by default it is set to None).  

```python
import mindspore.dataset as ds
import mindspore.common.dtype as mstype
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

dataset = ds.ImageFolder(dir)
type_cast_op = transforms.TypeCast(mstype.int32)
image_ops = [vision.RandomCropDecodeResize(train_image_size),
             vision.RandomHorizontalFlip(prob=0.5),
             vision.Normalize(mean=mean, std=std),
             vision.HWC2CHW()]
dataset = dataset.map(operations=type_cast_op, input_columns="label", offload=True)
dataset = dataset.map(operations=image_ops , input_columns="image", offload=True)
```

The heterogeneous acceleration supports being applied on multi-column dataset as the below example shows.

```python
dataset = dataset.map(operations=type_cast_op, input_columns="label")
dataset = dataset.map(operations=copy_column,
                      input_columns=["image", "label"],
                      output_columns=["image1", "image2", "label"],
                      column_order=["image1", "image2", "label"])
dataset = dataset.map(operations=image_ops, input_columns=["image1"], offload=True)
dataset = dataset.map(operations=image_ops, input_columns=["image2"], offload=True)
```

## Constraints

The heterogeneous acceleration feature is still under development. The current usage is limited under the following constraints:

1. The feature does not support concatenated or zipped datasets currently.  

2. The heterogeneous acceleration operator must be the last or more consecutive data augmentation operations acting on a particular data input column, but the data input column is processed in an unlimited order, for example

    ```python
    dataset = dataset.map(operations=type_cast_op, input_columns="label", offload=True)
    ```

    which can be shown in:

    ```python
    dataset = dataset.map(operations=image_ops, input_columns="image", offload=False)
    ```

    That is, even if the map operator acting on the "image" column is not set to offload, the map operator acting on the "label" column can also perform offload.

3. This feature does not currently support the user to specify output columns in the map data operator.
