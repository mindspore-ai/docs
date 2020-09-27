# Loading Image Dataset

`Linux` `Ascend` `GPU` `CPU` `Data Preparation` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Loading Image Dataset](#loading-image-dataset)
    - [Overview](#overview)
    - [Preparation](#preparation)
    - [Load Dataset](#load-dataset)
    - [Process Data](#process-data)
    - [Augmentation](#augmentation)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/training/source_en/use/load_dataset_image.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

In computer vision training tasks, it is often difficult to read the entire dataset directly into memory due to memory capacity. `mindspore.dataset` module provided by MindSpore enables user to customize their data fetching strategy from disk. At the same time, data processing and data augmentation operators are applied to the data. Pipelined data processing produces a continuous flow of data to the training network, improving overall performance. In addition, MindSpore supports data loading in distributed scenarios.

This tutorial uses the MNIST dataset as an example to demonstrate how to load and process image data using MindSpore.

## Preparation

1. Download and decompress the training [Image](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz) and [Label](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz) of the MNIST dataset to `./MNIST` directory. The directory structure is as follows.

    ```
    └─MNIST
        ├─train-images.idx3-ubyte
        └─train-labels.idx1-ubyte
    ```

2. Import `mindspore.dataset` module.

    ```python
    import mindspore.dataset as ds
    ```

## Load Dataset

MindSpore supports loading common datasets in the field of image processing that come in a variety of on-disk formats. User can also implement custom dataset class to load customized data.

The following shows how to load the MNIST dataset using the `MnistDataset` in the `mindspore.dataset` module.

1. Configure the dataset directory and create the `MnistDataset`.

    ```python
    DATA_DIR = "./MNIST"
    mnist_dataset = ds.MnistDataset(DATA_DIR, num_samples=6, shuffle=False)
    ```

2. Create an iterator then obtain data through the iterator.

    ```python
    import matplotlib.pyplot as plt

    mnist_it = mnist_dataset.create_dict_iterator()
    data = mnist_it.get_next()
    plt.imshow(data['image'].asnumpy().squeeze(), cmap=plt.cm.gray)
    plt.title(data['label'].asnumpy(), fontsize=20)
    plt.show()
    ```

    Image is shown below.

    ![mnist_5](./images/mnist_5.png)

In addition, user can pass in a sampler to specify the sampling process during dataset loading.
## Process Data

The following demonstrates how to construct a pipeline and perform operations such as `shuffle`, `batch` and `repeat` on MNIST dataset.

```python
for data in mnist_dataset.create_dict_iterator():
    print(data['label'])
```

The output is as follows:

```python
5
0
4
1
9
2
```

1. Shuffle the dataset.

    ```python
    ds.config.set_seed(58)
    ds1 = mnist_dataset.shuffle(buffer_size=6)

    for data in ds1.create_dict_iterator():
        print(data['label'])
    ```

    The output is as follows:

    ```python
    4
    2
    1
    0
    5
    9
    ```

2. Add `batch` after `shuffle`.

    ```python
    ds2 = ds1.batch(batch_size=2)

    for data in ds2.create_dict_iterator():
        print(data['label'])
    ```

    The output is as follows:

    ```python
    [4 2]
    [1 0]
    [5 9]
    ```

3. Add `repeat` after `batch`.

    ```python
    ds3 = ds2.repeat(count=2)

    for data in ds3.create_dict_iterator():
        print(data['label'])
    ```

    The output is as follows:

    ```python
    [4 2]
    [1 0]
    [5 9]
    [2 4]
    [0 9]
    [1 5]
    ```

    Results show the dataset is repeated, and the order of the replica is different from that of the first copy.

    > Having `repeat` in the pipeline results in the execution of repeated operations defined in the entire pipeline, instead of simply copying the current dataset.

## Augmentation

The following demonstrates how to use the `c_transforms` module to augment data in the MNIST dataset.

1. Import related modules and load the dataset.

    ```python
    from mindspore.dataset.vision import Inter
    import mindspore.dataset.vision.c_transforms as transforms

    mnist_dataset = ds.MnistDataset(DATA_DIR, num_samples=6, shuffle=False)
    ```

2. Define augmentation operators and perform `Resize` and `RandomCrop` operations on images in the dataset.

    ```python
    resize_op = transforms.Resize(size=(200,200), interpolation=Inter.LINEAR)
    crop_op = transforms.RandomCrop(150)
    transforms_list = [resize_op, crop_op]
    ds4 = mnist_dataset.map(operations=transforms_list, input_columns="image")
    ```

3. Visualize result of augmentation.

    ```python
    mnist_it = ds4.create_dict_iterator()
    data = mnist_it.get_next()
    plt.imshow(data['image'].asnumpy().squeeze(), cmap=plt.cm.gray)
    plt.title(data['label'].asnumpy(), fontsize=20)
    plt.show()
    ```

    The original image is scaled up then randomly cropped to 150 x 150.

    ![mnist_5_resize_crop](./images/mnist_5_resize_crop.png)
