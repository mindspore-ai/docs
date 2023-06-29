# Optimizing the Data Processing

`Linux` `Ascend` `GPU` `CPU` `Data Preparation` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.2/tutorials/training/source_en/advanced_use/optimize_data_processing.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

Data is the most important factor of deep learning. Data quality determines the upper limit of deep learning result, whereas model quality enables the result to approach the upper limit. Therefore, high-quality data input is beneficial to the entire deep neural network. During the entire data processing and data augmentation process, data continuously flows through a pipeline to the training system.

![title](./images/pipeline.png)

MindSpore provides data processing and data augmentation functions for users. In the pipeline process, if each step can be properly used, the data performance will be greatly improved. This section describes how to optimize performance during data loading, data processing, and data augmentation based on the CIFAR-10 dataset [1].

In addition, the storage, architecture and computing resources of the operating system will influence the performance of data processing to a certain extent.

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

Run the following command to download the dataset:
Download the CIFAR-10 Binary format dataset, decompress them and store them in the `./datasets` path, use this dataset when loading data.

```bash
!wget -N https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz
!mkdir -p datasets
!tar -xzf cifar-10-binary.tar.gz -C datasets
!mkdir -p datasets/cifar-10-batches-bin/train datasets/cifar-10-batches-bin/test
!mv -f datasets/cifar-10-batches-bin/test_batch.bin datasets/cifar-10-batches-bin/test
!mv -f datasets/cifar-10-batches-bin/data_batch*.bin datasets/cifar-10-batches-bin/batches.meta.txt datasets/cifar-10-batches-bin/train
!tree ./datasets/cifar-10-batches-bin
```

```text
./datasets/cifar-10-batches-bin
├── readme.html
├── test
│   └── test_batch.bin
└── train
    ├── batches.meta.txt
    ├── data_batch_1.bin
    ├── data_batch_2.bin
    ├── data_batch_3.bin
    ├── data_batch_4.bin
    └── data_batch_5.bin

2 directories, 8 files
```

Download cifar-10 Python file format dataset, decompress them in the `./datasets/cifar-10-batches-py` path, use this dataset when converting data.

```bash
!wget -N https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-python.tar.gz
!mkdir -p datasets
!tar -xzf cifar-10-python.tar.gz -C datasets
!tree ./datasets/cifar-10-batches-py
```

```text
./datasets/cifar-10-batches-py
├── batches.meta
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
├── readme.html
└── test_batch

0 directories, 8 files
```

## Optimizing the Data Loading Performance

MindSpore provides multiple data loading methods, including common dataset loading, user-defined dataset loading, and the MindSpore data format loading. The dataset loading performance varies depending on the underlying implementation method.

|      | Common Dataset | User-defined Dataset | MindRecord Dataset |
| :----: | :----: | :----: | :----: |
| Underlying implementation | C++ | Python | C++ |
| Performance | High | Medium | High |

### Performance Optimization Solution

![title](./images/data_loading_performance_scheme.png)

Suggestions on data loading performance optimization are as follows:

- Built-in loading operators are preferred for supported dataset formats. For details, see [Built-in Loading Operators](https://www.mindspore.cn/doc/api_python/en/r1.2/mindspore/mindspore.dataset.html), if the performance cannot meet the requirements, use the multi-thread concurrency solution. For details, see [Multi-thread Optimization Solution](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/optimize_data_processing.html#multi-thread-optimization-solution).
- For a dataset format that is not supported, convert the format to the MindSpore data format and then use the `MindDataset` class to load the dataset (Please refer to the [API](https://www.mindspore.cn/doc/api_python/en/r1.2/mindspore/dataset/mindspore.dataset.MindDataset.html) for detailed use). Please refer to [Converting Dataset to MindRecord](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/convert_dataset.html), if the performance cannot meet the requirements, use the multi-thread concurrency solution, for details, see [Multi-thread Optimization Solution](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/optimize_data_processing.html#multi-thread-optimization-solution).
- For dataset formats that are not supported, the user-defined `GeneratorDataset` class is preferred for implementing fast algorithm verification (Please refer to the [API](https://www.mindspore.cn/doc/api_python/en/r1.2/mindspore/dataset/mindspore.dataset.GeneratorDataset.html) for detailed use), if the performance cannot meet the requirements, the multi-process concurrency solution can be used. For details, see [Multi-process Optimization Solution](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/optimize_data_processing.html#multi-process-optimization-solution).

### Code Example

Based on the preceding suggestions of data loading performance optimization, the `Cifar10Dataset` class of built-in loading operators (Please refer to the [API](https://www.mindspore.cn/doc/api_python/en/r1.2/mindspore/dataset/mindspore.dataset.Cifar10Dataset.html) for detailed use), the `MindDataset` class after data conversion, and the `GeneratorDataset` class are used to load data. The sample code is displayed as follows:

1. Use the `Cifar10Dataset` class of built-in operators to load the CIFAR-10 dataset in binary format. The multi-thread optimization solution is used for data loading. Four threads are enabled to concurrently complete the task. Finally, a dictionary iterator is created for the data and a data record is read through the iterator.

    ```python
    cifar10_path = "./datasets/cifar-10-batches-bin/train"

    # create Cifar10Dataset for reading data
    cifar10_dataset = ds.Cifar10Dataset(cifar10_path, num_parallel_workers=4)
    # create a dictionary iterator and read a data record through the iterator
    print(next(cifar10_dataset.create_dict_iterator()))
    ```

    The output is as follows:

    ```text
    {'image': Tensor(shape=[32, 32, 3], dtype=UInt8, value=
        [[[181, 185, 194],
        [184, 187, 196],
        [189, 192, 201],
        ...
        [178, 181, 191],
        [171, 174, 183],
        [166, 170, 179]],
        [[182, 185, 194],
        [184, 187, 196],
        [189, 192, 201],
        ...
        [180, 183, 192],
        [173, 176, 185],
        [167, 170, 179]],
        [[185, 188, 197],
        [187, 190, 199],
        [193, 196, 205],
        ...
        [182, 185, 194],
        [176, 179, 188],
        [170, 173, 182]],
        ...
        [[176, 174, 185],
        [172, 171, 181],
        [174, 172, 183],
        ...
        [168, 171, 180],
        [164, 167, 176],
        [160, 163, 172]],
        [[172, 170, 181],
        [171, 169, 180],
        [173, 171, 182],
        ...
        [164, 167, 176],
        [160, 163, 172],
        [156, 159, 168]],
        [[171, 169, 180],
        [173, 171, 182],
        [177, 175, 186],
        ...
        [162, 165, 174],
        [158, 161, 170],
        [152, 155, 164]]]), 'label': Tensor(shape=[], dtype=UInt32, value= 6)}
    ```

2. Use the `Cifar10ToMR` class to convert the CIFAR-10 dataset into the MindSpore data format. In this example, the CIFAR-10 dataset in Python file format is used. Then use the `MindDataset` class to load the dataset in the MindSpore data format. The multi-thread optimization solution is used for data loading. Four threads are enabled to concurrently complete the task. Finally, a dictionary iterator is created for data and a data record is read through the iterator.

    ```python
    import os
    from mindspore.mindrecord import Cifar10ToMR

    trans_path = "./transform/"

    if not os.path.exists(trans_path):
        os.mkdir(trans_path)

    os.system("rm -f {}cifar10*".format(trans_path))

    cifar10_path = './datasets/cifar-10-batches-py'
    cifar10_mindrecord_path = './transform/cifar10.record'

    cifar10_transformer = Cifar10ToMR(cifar10_path, cifar10_mindrecord_path)
    # execute transformation from CIFAR-10 to MindRecord
    cifar10_transformer.transform(['label'])

    # create MindDataset for reading data
    cifar10_mind_dataset = ds.MindDataset(dataset_file=cifar10_mindrecord_path, num_parallel_workers=4)
    # create a dictionary iterator and read a data record through the iterator
    print(next(cifar10_mind_dataset.create_dict_iterator()))
    ```

    The output is as follows:

    ```text
    {'data': Tensor(shape=[1289], dtype=UInt8, value= [255, 216, 255, 224,   0,  16,  74,  70,  73,  70,   0,   1,   1,   0,   0,   1,   0,   1,   0,   0, 255, 219,   0,  67,
    0,   2,   1,   1,   1,   1,   1,   2,   1,   1,   1,   2,   2,   2,   2,   2,   4,   3,   2,   2,   2,   2,   5,   4,
    4,   3,   4,   6,   5,   6,   6,   6,   5,   6,   6,   6,   7,   9,   8,   6,   7,   9,   7,   6,   6,   8,  11,   8,
    9,  10,  10,  10,  10,  10,   6,   8,  11,  12,  11,  10,  12,   9,  10,  10,  10, 255, 219,   0,  67,   1,   2,   2,
    ...
    ...
    ...
    39, 227, 206, 143, 241,  91, 196, 154, 230, 189, 125, 165, 105, 218,  94, 163, 124, 146,  11, 187,  29,  34, 217, 210,
    23, 186,  56,  14, 192,  19, 181,   1,  57,  36,  14,  51, 211, 173, 105,   9, 191, 100, 212, 174, 122,  25, 110,  39,
    11, 133, 193, 226, 169,  73,  36, 234,  69,  90, 222,  93,  31, 223, 115, 255, 217]), 'id': Tensor(shape=[], dtype=Int64, value= 46084), 'label': Tensor(shape=[], dtype=Int64, value= 5)}
    ```

3. The `GeneratorDataset` class is used to load the user-defined dataset, and the multi-process optimization solution is used. Four processes are enabled to concurrently complete the task. Finally, a dictionary iterator is created for the data, and a data record is read through the iterator.

    ```python
    def generator_func(num):
        for i in range(num):
            yield (np.array([i]),)

    # create a GeneratorDataset object for reading data
    dataset = ds.GeneratorDataset(source=generator_func(5), column_names=["data"], num_parallel_workers=4)
    # create a dictionary iterator and read a data record through the iterator
    print(next(dataset.create_dict_iterator()))
    ```

    The output is as follows:

    ```text
    {'data': Tensor(shape=[1], dtype=Int64, value= [0])}
    ```

## Optimizing the Shuffle Performance

The shuffle operation is used to shuffle ordered datasets or repeated datasets. MindSpore provides the `shuffle` function for users.  A larger value of `buffer_size` indicates a higher shuffling degree, consuming more time and computing resources. This API allows users to shuffle the data at any time during the entire pipeline process.Please refer to [shuffle](https://www.mindspore.cn/doc/programming_guide/en/r1.2/pipeline.html#shuffle). However, because the underlying implementation methods are different, the performance of this method is not as good as that of setting the `shuffle` parameter to directly shuffle data by referring to the [Built-in Loading Operators](https://www.mindspore.cn/doc/api_python/en/r1.2/mindspore/mindspore.dataset.html).

### Performance Optimization Solution

![title](./images/shuffle_performance_scheme.png)

Suggestions on shuffle performance optimization are as follows:

- Use the `shuffle` parameter of built-in loading operators to shuffle data.
- If the `shuffle` function is used and the performance still cannot meet the requirements, adjust the value of the `buffer_size` parameter to improve the performance.

### Code Example

Based on the preceding shuffle performance optimization suggestions, the `shuffle` parameter of the `Cifar10Dataset` class of built-in loading operators and the `Shuffle` function are used to shuffle data. The sample code is displayed as follows:

1. Use the `Cifar10Dataset` class of built-in operators to load the CIFAR-10 dataset. In this example, the CIFAR-10 dataset in binary format is used, and the `shuffle` parameter is set to True to perform data shuffle. Finally, a dictionary iterator is created for the data and a data record is read through the iterator.

    ```python
    cifar10_path = "./datasets/cifar-10-batches-bin/train"

    # create Cifar10Dataset for reading data
    cifar10_dataset = ds.Cifar10Dataset(cifar10_path, shuffle=True)
    # create a dictionary iterator and read a data record through the iterator
    print(next(cifar10_dataset.create_dict_iterator()))
    ```

    The output is as follows:

    ```text
    {'image': Tensor(shape=[32, 32, 3], dtype=UInt8, value=
    [[[213, 205, 194],
    [215, 207, 196],
    [219, 210, 200],
    ...
    [253, 254, 249],
    [253, 254, 249],
    [253, 254, 249]],
    [[218, 208, 198],
    [220, 210, 200],
    [222, 212, 202],
    ...
    [253, 254, 249],
    [253, 254, 249],
    [253, 254, 249]],
    [[219, 209, 198],
    [222, 211, 200],
    [224, 214, 202],
    ...
    [254, 253, 248],
    [254, 253, 248],
    [254, 253, 248]],
    ...
    [[135, 141, 139],
    [135, 141, 139],
    [146, 152, 150],
    ...
    [172, 174, 172],
    [181, 182, 182],
    [168, 168, 167]],
    [[113, 119, 117],
    [109, 115, 113],
    [117, 123, 121],
    ...
    [155, 159, 156],
    [150, 155, 155],
    [135, 140, 140]],
    [[121, 127, 125],
    [117, 123, 121],
    [121, 127, 125],
    ...
    [180, 184, 180],
    [141, 146, 144],
    [125, 130, 129]]]), 'label': Tensor(shape=[], dtype=UInt32, value= 8)}
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

    The output is as follows:

    ```text
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

Please refer to [Data Augmentation](https://www.mindspore.cn/doc/programming_guide/en/r1.2/augmentation.html). The performance varies according to the underlying implementation methods.

| Module | Underlying API | Description |
| :----: | :----: | :----: |
| c_transforms | C++ (based on OpenCV) | High performance |
| py_transforms | Python (based on PIL) | This module provides multiple image augmentation functions and the method for converting PIL images into NumPy arrays |

### Performance Optimization Solution

![title](./images/data_enhancement_performance_scheme.png)

Suggestions on data augmentation performance optimization are as follows:

- The `c_transforms` module is preferentially used to perform data augmentation for its highest performance. If the performance cannot meet the requirements, refer to [Multi-thread Optimization Solution](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/optimize_data_processing.html#multi-thread-optimization-solution), [Compose Optimization Solution](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/optimize_data_processing.html#compose-optimization-solution), or [Operator Fusion Optimization Solution](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/optimize_data_processing.html#operator-fusion-optimization-solution).
- If the `py_transforms` module is used to perform data augmentation and the performance still cannot meet the requirements, refer to [Multi-thread Optimization Solution](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/optimize_data_processing.html#multi-thread-optimization-solution), [Multi-process Optimization Solution](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/optimize_data_processing.html#multi-process-optimization-solution), [Compose Optimization Solution](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/optimize_data_processing.html#compose-optimization-solution), or [Operator Fusion Optimization Solution](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/optimize_data_processing.html#operator-fusion-optimization-solution).
- The `c_transforms` module maintains buffer management in C++, and the `py_transforms` module maintains buffer management in Python. Because of the performance cost of switching between Python and C++, it is advised not to use different operator types together.
- If the user-defined Python functions are used to perform data augmentation and the performance still cannot meet the requirements, use the [Multi-thread Optimization Solution](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/optimize_data_processing.html#multi-thread-optimization-solution) or [Multi-process Optimization Solution](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/optimize_data_processing.html#multi-process-optimization-solution). If the performance still cannot be improved, in this case, optimize the user-defined Python code.

### Code Example

Based on the preceding suggestions of data augmentation performance optimization, the `c_transforms` module and user-defined Python function are used to perform data augmentation. The code is displayed as follows:

1. The `c_transforms` module is used to perform data augmentation. During data augmentation, the multi-thread optimization solution is used. Four threads are enabled to concurrently complete the task. The operator fusion optimization solution is used and the `RandomResizedCrop` fusion class is used to replace the `RandomResize` and `RandomCrop` classes.

    ```python
    import mindspore.dataset.transforms.c_transforms as c_transforms
    import mindspore.dataset.vision.c_transforms as C
    import matplotlib.pyplot as plt

    cifar10_path = "./datasets/cifar-10-batches-bin/train"

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

    ```text
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

## Optimizing the Operating System Performance

Data processing is performed on the host. Therefore, configurations of the host or operating system may affect the performance of data processing.  Major factors include storage, NUMA architecture, and CPU (computing resources).

1. Storage

    Solid State Drive (SSD) is recommended for storing large datasets. SSD reduces the impact of I/O on data processing.

    > In most cases, after a dataset is loaded, it is stored in page cache of the operating system. To some extent, this reduces I/O overheads and accelerates reading subsequent epochs.

2. NUMA architecture

    NUMA (Non-uniform Memory Architecture) is developed to solve the scalability problem of traditional Symmetric Multi-processor systems. The NUMA system has multiple memory buses. Several processors are connected to one memory via memory bus to form a group. This way, the entire large system is divided into several groups, the concept of this group is called a node in the NUMA system. Memory belonging to this node is called local memory, memory belonging to other nodes (with respect to this node) is called foreign memory. Therefore, the latency for each node to access its local memory is different from accessing foreign memory. This needs to be avoided during data processing. Generally, the following command can be used to bind a process to a node:

    ```shell
    numactl --cpubind=0 --membind=0 python train.py
    ```

    The example above binds the `train.py` process to `numa node` 0.

3. CPU (computing resource)

    CPU affects data processing in two aspects: resource allocation and CPU frequency.

    - Resource allocation

        In distributed training, multiple training processes are run on one device. These training processes allocate and compete for computing resources based on the policy of the operating system. When there is a large number of processes, data processing performance may deteriorate due to resource contention. In some cases, users need to manually allocate resources to avoid resource contention.

        ```shell
        numactl --cpubind=0 python train.py
        ```

        or

        ```shell
        taskset -c 0-15 python train.py
        ```

        > The `numactl` method directly specifies `numa node id`. The `taskset` method allows for finer control by specifying `cpu core` within a `numa node`. The `core id` range from 0 to 15.

    - CPU frequency

        The setting of CPU frequency is critical to maximizing the computing power of the host CPU. Generally, the Linux kernel supports the tuning of the CPU frequency to reduce power consumption. Power consumption can be reduced to varying degrees by selecting power management policies for different system idle states. However, lower power consumption means slower CPU wake-up which in turn impacts performance. Therefore, if the CPU's power setting is in the conservative or powersave mode, `cpupower` command can be used to switch performance modes, resulting in significant data processing performance improvement.

        ```shell
        cpupower frequency-set -g performance
        ```

## Performance Optimization Solution Summary

### Multi-thread Optimization Solution

During the data pipeline process, the number of threads for related operators can be set to improve the concurrency and performance. For example:

- During data loading, the `num_parallel_workers` parameter in the built-in data loading class is used to set the number of threads.
- During data augmentation, the `num_parallel_workers` parameter in the `map` function is used to set the number of threads.
- During batch processing, the `num_parallel_workers` parameter in the `batch` function is used to set the number of threads.

For details, see [Built-in Loading Operators](https://www.mindspore.cn/doc/api_python/en/r1.2/mindspore/mindspore.dataset.html).

### Multi-process Optimization Solution

During data processing, operators implemented by Python support the multi-process mode. For example:

- By default, the `GeneratorDataset` class is in multi-process mode. The `num_parallel_workers` parameter indicates the number of enabled processes. The default value is 1. For details, see [GeneratorDataset](https://www.mindspore.cn/doc/api_python/en/r1.2/mindspore/dataset/mindspore.dataset.GeneratorDataset.html).
- If the user-defined Python function or the `py_transforms` module is used to perform data augmentation and the `python_multiprocessing` parameter of the `map` function is set to True, the `num_parallel_workers` parameter indicates the number of processes and the default value of the `python_multiprocessing` parameter is False. In this case, the `num_parallel_workers` parameter indicates the number of threads. For details, see [Built-in Loading Operators](https://www.mindspore.cn/doc/api_python/en/r1.2/mindspore/mindspore.dataset.html).

### Compose Optimization Solution

Map operators can receive the Tensor operator list and apply all these operators based on a specific sequence. Compared with the Map operator used by each Tensor operator, such Fat Map operators can achieve better performance, as shown in the following figure:

![title](./images/compose.png)

### Operator Fusion Optimization Solution

Some fusion operators are provided to aggregate the functions of two or more operators into one operator. For details, see [Augmentation Operators](https://www.mindspore.cn/doc/api_python/en/r1.2/mindspore/mindspore.dataset.vision.html). Compared with the pipelines of their components, such fusion operators provide better performance. As shown in the figure:

![title](./images/operator_fusion.png)

### Operating System Optimization Solution

- Use Solid State Drives to store the data.
- Bind the process to a NUMA node.
- Manually allocate more computing resources.
- Set a higher CPU frequency.

## References

[1] Alex Krizhevsky. [Learning Multiple Layers of Features from Tiny Images](http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf).
