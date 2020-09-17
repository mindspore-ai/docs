# Optimize Data Processing

`Linux` `Ascend` `GPU` `CPU` `Data Preparation` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Optimize Data Processing](#optimize-data-processing)
    - [Overview](#overview)
    - [Overall Process](#overall-process)
    - [Preparations](#preparations)
        - [Importing Modules](#importing-modules)
        - [Downloading the Required Dataset](#downloading-the-required-dataset)
    - [Optimizing the Data Loading Performance](#optimizing-the-data-loading-performance)
        - [Performance Optimization Solution](#performance-optimization-solution)
        - [Code Example](#code-example)
    - [Optimizing the Shuffle Performance](#optimizing-the-shuffle-performance)
        - [Performance Optimization Solution](#performance-optimization-solution-1)
        - [Code Example](#code-example-1)
    - [Optimizing the Data Augmentation Performance](#optimizing-the-data-augmentation-performance)
        - [Performance Optimization Solution](#performance-optimization-solution-2)
        - [Code Example](#code-example-2)
    - [Performance Optimization Solution Summary](#performance-optimization-solution-summary)
        - [Multi-thread Optimization Solution](#multi-thread-optimization-solution)
        - [Multi-process Optimization Solution](#multi-process-optimization-solution)
        - [Compose Optimization Solution](#compose-optimization-solution)
        - [Operator Fusion Optimization Solution](#operator-fusion-optimization-solution)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_en/advanced_use/optimize_the_performance_of_data_preparation.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

Data is the most important factor of deep learning. Data quality determines the upper limit of deep learning result, whereas model quality enables the result to approach the upper limit.Therefore, high-quality data input  is beneficial to the entire deep neural network. During the entire data processing and data augmentation process, data continuously flows through a "pipeline" to the training system, as shown in the following figure:

![title](./images/pipeline.png)

MindSpore provides data processing and data augmentation functions for users. In the pipeline process, if each step can be properly used, the data performance will be greatly improved. This section describes how to optimize performance during data loading, data processing, and data augmentation based on the CIFAR-10 dataset.

## Overall Process
- Prepare data.
- Optimize the data loading performance.
- Optimize the shuffle performance.
- Optimize the data augmentation performance.
- Summarize the performance optimization solution.

## Preparations

### Importing Modules

The `dataset` module provides APIs for loading and processing datasets.


```python
import mindspore.dataset as ds
```

The `numpy` module is used to generate ndarrays.


```python
import numpy as np
```

### Downloading the Required Dataset

1. Create the `./dataset/Cifar10Data` directory in the current working directory. The dataset used for this practice is stored in this directory.
2. Create the `./transform` directory in the current working directory. The dataset generated during the practice is stored in this directory.
3. Download [the CIFAR-10 dataset in binary format](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) and decompress the dataset file to the `./dataset/Cifar10Data/cifar-10-batches-bin` directory. The dataset will be used during data loading.
4. Download [the CIFAR-10 Python dataset in file-format](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and decompress the dataset file to the `./dataset/Cifar10Data/cifar-10-batches-py` directory. The dataset will be used for data conversion.

The directory structure is as follows:


    dataset/Cifar10Data
    ├── cifar-10-batches-bin
    │   ├── batches.meta.txt
    │   ├── data_batch_1.bin
    │   ├── data_batch_2.bin
    │   ├── data_batch_3.bin
    │   ├── data_batch_4.bin
    │   ├── data_batch_5.bin
    │   ├── readme.html
    │   └── test_batch.bin
    └── cifar-10-batches-py
        ├── batches.meta
        ├── data_batch_1
        ├── data_batch_2
        ├── data_batch_3
        ├── data_batch_4
        ├── data_batch_5
        ├── readme.html
        └── test_batch

In the preceding information:
- The `cifar-10-batches-bin` directory is the directory for storing the CIFAR-10 dataset in binary format.
- The `cifar-10-batches-py` directory is the directory for storing the CIFAR-10 dataset in Python file format.

## Optimizing the Data Loading Performance

MindSpore provides multiple data loading methods, including common dataset loading, user-defined dataset loading, and MindSpore data format loading. For details, see [Loading Datasets](https://www.mindspore.cn/tutorial/en/master/use/data_preparation/loading_the_datasets.html). The dataset loading performance varies depending on the underlying implementation method.

|      | Common Dataset | User-defined Dataset | MindRecord Dataset |
| :----: | :----: | :----: | :----: |
| Underlying implementation | C++ | Python | C++ |
| Performance | High | Medium | High |

### Performance Optimization Solution

![title](./images/data_loading_performance_scheme.png)

Suggestions on data loading performance optimization are as follows:
- Built-in loading operators are preferred for supported dataset formats. For details, see [Built-in Loading Operators](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.dataset.html). If the performance cannot meet the requirements, use the multi-thread concurrency solution. For details, see [Multi-thread Optimization Solution](#multi-thread-optimization-solution).
- For a dataset format that is not supported, convert the format to MindSpore data format and then use the `MindDataset` class to load the dataset. For details, see [Converting Datasets into MindSpore Data Format](https://www.mindspore.cn/tutorial/en/master/use/data_preparation/converting_datasets.html). If the performance cannot meet the requirements, use the multi-thread concurrency solution, for details, see [Multi-thread Optimization Solution](#multi-thread-optimization-solution).
- For dataset formats that are not supported, the user-defined `GeneratorDataset` class is preferred for implementing fast algorithm verification. If the performance cannot meet the requirements, the multi-process concurrency solution can be used. For details, see [Multi-process Optimization Solution](#multi-process-optimization-solution).

### Code Example

Based on the preceding suggestions of data loading performance optimization, the `Cifar10Dataset` class of built-in loading operators, the `MindDataset` class after data conversion, and the `GeneratorDataset` class are used to load data. The sample code is displayed as follows:

1. Use the `Cifar10Dataset` class of built-in operators to load the CIFAR-10 dataset in binary format. The multi-thread optimization solution is used for data loading. Four threads are enabled to concurrently complete the task. Finally, a dictionary iterator is created for the data and a data record is read through the iterator.


    ```python
    cifar10_path = "./dataset/Cifar10Data/cifar-10-batches-bin/"

    # create Cifar10Dataset for reading data
    cifar10_dataset = ds.Cifar10Dataset(cifar10_path, num_parallel_workers=4)
    # create a dictionary iterator and read a data record through the iterator
    print(next(cifar10_dataset.create_dict_iterator()))
    ```

    The output is as follows:
    ```
    {'image': Tensor(shape=[32, 32, 3], dtype=UInt8, value=
          [[[235, 235, 235],
            [230, 230, 230],
            [234, 234, 234],
            ...,
            [248, 248, 248],
            [248, 248, 248],
            [249, 249, 249]],
            ...,
            [120, 120, 119],
            [146, 146, 146],
            [177, 174, 190]]]), 'label': Tensor(shape=[], dtype=UInt32, value= 9)}
    ```

2. Use the `Cifar10ToMR` class to convert the CIFAR-10 dataset into MindSpore data format. In this example, the CIFAR-10 dataset in Python file format is used. Then use the `MindDataset` class to load the dataset in MindSpore data format. The multi-thread optimization solution is used for data loading. Four threads are enabled to concurrently complete the task. Finally, a dictionary iterator is created for data and a data record is read through the iterator.


    ```python
    from mindspore.mindrecord import Cifar10ToMR

    cifar10_path = './dataset/Cifar10Data/cifar-10-batches-py/'
    cifar10_mindrecord_path = './transform/cifar10.record'

    cifar10_transformer = Cifar10ToMR(cifar10_path, cifar10_mindrecord_path)
    # executes transformation from Cifar10 to MindRecord
    cifar10_transformer.transform(['label'])

    # create MindDataset for reading data
    cifar10_mind_dataset = ds.MindDataset(dataset_file=cifar10_mindrecord_path, num_parallel_workers=4)
    # create a dictionary iterator and read a data record through the iterator
    print(next(cifar10_mind_dataset.create_dict_iterator()))
    ```

    The output is as follows:
    ```
    {'data': Tensor(shape=[1431], dtype=UInt8, value= [255, 216, 255, ...,  63, 255, 217]), 'id': Tensor(shape=[], dtype=Int64, value= 30474), 'label': Tensor(shape=[], dtype=Int64, value= 2)}
    ```

3. The `GeneratorDataset` class is used to load the user-defined dataset, and the multi-process optimization solution is used. Four processes are enabled to concurrently complete the task. Finally, a dictionary iterator is created for the data, and a data record is read through the iterator.


    ```python
    def generator_func(num):
        for i in range(num):
            yield (np.array([i]),)

    # create GeneratorDataset for reading data
    dataset = ds.GeneratorDataset(source=generator_func(5), column_names=["data"], num_parallel_workers=4)
    # create a dictionary iterator and read a data record through the iterator
    print(next(dataset.create_dict_iterator()))
    ```

    The output is as follows:
    ```
    {'data': Tensor(shape=[1], dtype=Int64, value= [0])}
    ```

## Optimizing the Shuffle Performance

The shuffle operation is used to shuffle ordered datasets or repeated datasets. MindSpore provides the `shuffle` function for users.  A larger value of `buffer_size` indicates a higher shuffling degree, consuming more time and computing resources. This API allows users to shuffle the data at any time during the entire pipeline process. For details, see [Shuffle Processing](https://www.mindspore.cn/tutorial/en/master/use/data_preparation/data_processing_and_augmentation.html#shuffle). However, because the underlying implementation methods are different, the performance of this method is not as good as that of setting the `shuffle` parameter to directly shuffle data by referring to the [Built-in Loading Operators](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.dataset.html).

### Performance Optimization Solution

![title](./images/shuffle_performance_scheme.png)

Suggestions on shuffle performance optimization are as follows:
- Use the `shuffle` parameter of built-in loading operators to shuffle data.
- If the `shuffle` function is used and the performance still cannot meet the requirements, increase the value of the `buffer_size` parameter to improve the performance.

### Code Example

Based on the preceding shuffle performance optimization suggestions, the `shuffle` parameter of the `Cifar10Dataset` class of built-in loading operators and the `Shuffle` function are used to shuffle data. The sample code is displayed as follows:

1. Use the built-in operator in `Cifar10Dataset` class to load the CIFAR-10 dataset. In this example, the CIFAR-10 dataset in binary format is used, and the `shuffle` parameter is set to True to perform data shuffle. Finally, a dictionary iterator is created for the data and a data record is read through the iterator.


    ```python
    cifar10_path = "./dataset/Cifar10Data/cifar-10-batches-bin/"

    # create Cifar10Dataset for reading data
    cifar10_dataset = ds.Cifar10Dataset(cifar10_path, shuffle=True)
    # create a dictionary iterator and read a data record through the iterator
    print(next(cifar10_dataset.create_dict_iterator()))
    ```

    The output is as follows:
    ```
    {'image': Tensor(shape=[32, 32, 3], dtype=UInt8, value=
          [[[254, 254, 254],
            [255, 255, 254],
            [255, 255, 254],
            ...,
            [232, 234, 244],
            [226, 230, 242],
            [228, 232, 243]],
            ...,
            [ 64,  61,  63],
            [ 63,  58,  60],
            [ 61,  56,  58]]]), 'label': Tensor(shape=[], dtype=UInt32, value= 9)}
    ```

2. Use the `shuffle` function to shuffle data. Set `buffer_size` to 3 and use the `GeneratorDataset` class to generate data.


    ```python
    def generator_func():
        for i in range(5):
            yield (np.array([i, i+1, i+2, i+3, i+4]),)

    ds1 = ds.GeneratorDataset(source=generator_func, column_names=["data"])
    print("before shuffle:")
    for data in ds1.create_dict_iterator():
        print(data["data"])

    ds2 = ds1.shuffle(buffer_size=3)
    print("after shuffle:")
    for data in ds2.create_dict_iterator():
        print(data["data"])
    ```
    ```
    The output is as follows:

    before shuffle:
    [0 1 2 3 4]
    [1 2 3 4 5]
    [2 3 4 5 6]
    [3 4 5 6 7]
    [4 5 6 7 8]
    after shuffle:
    [2 3 4 5 6]
    [0 1 2 3 4]
    [4 5 6 7 8]
    [1 2 3 4 5]
    [3 4 5 6 7]
    ```

## Optimizing the Data Augmentation Performance

During image classification training, especially when the dataset is small, users can use data augmentation to preprocess images to enrich the dataset. MindSpore provides multiple data augmentation methods, including:
- Use the built-in C operator (`c_transforms` module) to perform data augmentation.
- Use the built-in Python operator (`py_transforms` module) to perform data augmentation.
- Users can define Python functions as needed to perform data augmentation.

For details, see [Data Augmentation](https://www.mindspore.cn/tutorial/en/master/use/data_preparation/data_processing_and_augmentation.html#id3). The performance varies according to the underlying implementation methods.

| Module | Underlying API | Description |
| :----: | :----: | :----: |
| c_transforms | C++ (based on OpenCV) | High performance |
| py_transforms | Python (based on PIL) | This module provides multiple image augmentation functions and the method for converting PIL images into NumPy arrays. |


### Performance Optimization Solution

![title](./images/data_enhancement_performance_scheme.png)


Suggestions on data augmentation performance optimization are as follows:
- The `c_transforms` module is preferentially used to perform data augmentation for its highest performance. If the performance cannot meet the requirements, refer to [Multi-thread Optimization Solution](#multi-thread-optimization-solution), [Compose Optimization Solution](#compose-optimization-solution), or [Operator Fusion Optimization Solution](#operator-fusion-optimization-solution).
- If the `py_transforms` module is used to perform data augmentation and the performance still cannot meet the requirements, refer to [Multi-thread Optimization Solution](#multi-thread-optimization-solution), [Multi-process Optimization Solution](#multi-process-optimization-solution), [Compose Optimization Solution](#compose-optimization-solution), or [Operator Fusion Optimization Solution](#operator-fusion-optimization-solution).
- The `c_transforms` module maintains buffer management in C++, and the `py_transforms` module maintains buffer management in Python. Because of the performance cost of switching between Python and C++, it is advised not to use different operator types together.
- If the user-defined Python functions are used to perform data augmentation and the performance still cannot meet the requirements, use the [Multi-thread Optimization Solution](#multi-thread-optimization-solution) or [Multi-process Optimization Solution](#multi-process-optimization-solution). If the performance still cannot be improved, in this case, optimize the user-defined Python code.

### Code Example

Based on the preceding suggestions of data augmentation performance optimization, the `c_transforms` module and user-defined Python function are used to perform data augmentation. The code is displayed as follows:

1. The `c_transforms` module is used to perform data augmentation. During data augmentation, the multi-thread optimization solution is used. Four threads are enabled to concurrently complete the task. The operator fusion optimization solution is used and the `RandomResizedCrop` fusion class is used to replace the `RandomResize` and `RandomCrop` classes.


    ```python
    import mindspore.dataset.transforms.c_transforms as c_transforms
    import mindspore.dataset.vision.c_transforms as C
    import matplotlib.pyplot as plt
    cifar10_path = "./dataset/Cifar10Data/cifar-10-batches-bin/"

    # create Cifar10Dataset for reading data
    cifar10_dataset = ds.Cifar10Dataset(cifar10_path, num_parallel_workers=4)
    transforms = C.RandomResizedCrop((800, 800))
    # apply the transform to the dataset through dataset.map()
    cifar10_dataset = cifar10_dataset.map(operations=transforms, input_columns="image", num_parallel_workers=4)

    data = next(cifar10_dataset.create_dict_iterator())
    plt.imshow(data["image"].asnumpy())
    plt.show()
    ```

    The output is as follows:

    ![png](./images/cifar10_c_transforms.png)


2. A user-defined Python function is used to perform data augmentation. During data augmentation, the multi-process optimization solution is used, and four processes are enabled to concurrently complete the task.


    ```python
    def generator_func():
        for i in range(5):
            yield (np.array([i, i+1, i+2, i+3, i+4]),)

    ds3 = ds.GeneratorDataset(source=generator_func, column_names=["data"])
    print("before map:")
    for data in ds3.create_dict_iterator():
        print(data["data"])

    func = lambda x:x**2
    ds4 = ds3.map(operations=func, input_columns="data", python_multiprocessing=True, num_parallel_workers=4)
    print("after map:")
    for data in ds4.create_dict_iterator():
        print(data["data"])
    ```

    The output is as follows:
    ```
    before map:
    [0 1 2 3 4]
    [1 2 3 4 5]
    [2 3 4 5 6]
    [3 4 5 6 7]
    [4 5 6 7 8]
    after map:
    [ 0  1  4  9 16]
    [ 1  4  9 16 25]
    [ 4  9 16 25 36]
    [ 9 16 25 36 49]
    [16 25 36 49 64]
    ```

## Performance Optimization Solution Summary

### Multi-thread Optimization Solution

During the data pipeline process, the number of threads for related operators can be set to improve the concurrency and performance. For example:
- During data loading, the `num_parallel_workers` parameter in the built-in data loading class is used to set the number of threads.
- During data augmentation, the `num_parallel_workers` parameter in the `map` function is used to set the number of threads.
- During batch processing, the `num_parallel_workers` parameter in the `batch` function is used to set the number of threads.

For details, see [Built-in Loading Operators](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.dataset.html).

### Multi-process Optimization Solution

During data processing, operators implemented by Python support the multi-process mode. For example:
- By default, the `GeneratorDataset` class is in multi-process mode. The `num_parallel_workers` parameter indicates the number of enabled processes. The default value is 1. For details, see [Generator Dataset](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.dataset.html#mindspore.dataset.GeneratorDataset)
- If the user-defined Python function or the `py_transforms` module is used to perform data augmentation and the `python_multiprocessing` parameter of the `map` function is set to True, the `num_parallel_workers` parameter indicates the number of processes and the default value of the `python_multiprocessing` parameter is False. In this case, the `num_parallel_workers` parameter indicates the number of threads. For details, see [Built-in Loading Operators](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.dataset.html).

### Compose Optimization Solution

Map operators can receive the Tensor operator list and apply all these operators based on a specific sequence. Compared with the Map operator used by each Tensor operator, such Fat Map operators can achieve better performance, as shown in the following figure:

![title](./images/compose.png)

### Operator Fusion Optimization Solution

Some fusion operators are provided to aggregate the functions of two or more operators into one operator. For details, see [Data Augmentation Operators](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.dataset.vision.html). Compared with the pipelines of their components, such fusion operators provide better performance. As shown in the figure:

![title](./images/operator_fusion.png)
