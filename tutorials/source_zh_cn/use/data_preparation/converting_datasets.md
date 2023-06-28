# 将数据集转换为MindSpore数据格式

<a href="https://gitee.com/mindspore/docs/blob/r0.3/tutorials/source_zh_cn/use/data_preparation/converting_datasets.md" target="_blank"><img src="../../_static/logo_source.png"></a>

## 概述

用户可以将非标准的数据集和常见的数据集转换为MindSpore数据格式，从而方便地加载到MindSpore中进行训练。同时，MindSpore在部分场景做了性能优化，使用MindSpore数据格式可以获得更好的性能体验。
MindSpore数据格式具备的特征如下：
1. 实现多变的用户数据统一存储、访问，训练数据读取更简便。
2. 数据聚合存储，高效读取，且方便管理、移动。
3. 高效数据编解码操作，对用户透明、无感知。
4. 灵活控制分区大小，实现分布式训练。

## 将非标准数据集转换为MindSpore数据格式

MindSpore提供写操作工具，可将用户定义的原始数据写为MindSpore数据格式。

### 转换图片及标注数据

1. 导入文件写入工具类`FileWriter`。

    ```python
    from mindspore.mindrecord import FileWriter
    ```

2. 定义数据集Schema，Schema用于定义数据集包含哪些字段以及字段的类型。

    ```python
    cv_schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
    ```
    其中，Schema的相关规范如下：
    字段名：字母、数字、下划线。
    字段属性type：int32、int64、float32、float64、string、bytes。
    字段属性shape：[...], ...可以是一维数组，用[-1]表示; 可以是二维数组，用[m, n]表示；可以是三维数组，用[x, y, z]表示。

    > 1. 如果字段有属性Shape，暂时只支持type为int32、int64、float32、float64类型。
    > 2. 如果字段有属性Shape，则用户在准备数据并传入write_raw_data接口时必须是numpy.ndarray类型。

    举例：
    - 图片分类
        ```python
        cv_schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
        ```
    - NLP类
        ```python
        cv_schema_json = {"id": {"type": "int32"}, "masks": {"type": "int32", "shape": [-1]}, "inputs": {"type": "int64", "shape": [4, 32]}, "labels": {"type": "int64", "shape": [-1]}}
        ```

3. 准备需要写入的数据，按照用户定义的Schema形式，准备需要写入的样本列表。

    ```python
    data = [{"file_name": "1.jpg", "label": 0, "data": b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff\xd9"},
            {"file_name": "2.jpg", "label": 56, "data": b"\xe6\xda\xd1\xae\x07\xb8>\xd4\x00\xf8\x129\x15\xd9\xf2q\xc0\xa2\x91YFUO\x1dsE1\x1ep"},
            {"file_name": "3.jpg", "label": 99, "data": b"\xaf\xafU<\xb8|6\xbd}\xc1\x99[\xeaj+\x8f\x84\xd3\xcc\xa0,i\xbb\xb9-\xcdz\xecp{T\xb1\xdb\"}]
    ```

4. 准备索引字段，添加索引字段可以加速数据读取，该步骤非必选。

    ```python
    indexes = ["file_name", "label"]
    ```

5. 创建`FileWriter`对象，传入文件名，分片数量，然后添加Schema，添加索引，调用`write_raw_data`接口写入数据，最后调用`commit`接口生成本地数据文件。

    ```python
    writer = FileWriter(file_name="testWriter.mindrecord", shard_num=4)
    writer.add_schema(cv_schema_json, "test_schema")
    writer.add_index(indexes)
    writer.write_raw_data(data)
    writer.commit()
    ```
    其中，
    `write_raw_data`：会将数据写入到内存中。
    `commit`：最终将内存中的数据写入到磁盘。

6. 在现有数据格式文件中增加新数据，调用`open_for_append`接口打开已存在的数据文件，继续调用`write_raw_data`接口写入新数据，最后调用`commit`接口生成本地数据文件。
    ```python
    writer = FileWriter.open_for_append("testWriter.mindrecord0")
    writer.write_raw_data(data)
    writer.commit()
    ```

## 将常见数据集转换为MindSpore数据格式

MindSpore提供转换常见数据集的工具类，将常见数据集转换为MindSpore数据格式。对于常见的数据集以及调用的工具类列表如下：

| 数据集   | 调用的工具类 |
| -------- | ------------ |
| CIFAR-10 | Cifar10ToMR  |
| CIFAR-100| Cifar100ToMR |
| ImageNet | ImageNetToMR |
| MNIST    | MnistToMR    |


### 转换CIFAR-10数据集
用户可以通过`Cifar10ToMR`类，将CIFAR-10原始数据转换为MindSpore数据格式。

1. 准备好CIFAR-10数据集，这里使用的是用于python解析的数据集（CIFAR-10 python version），将文件解压至指定的目录（示例中将数据集保存到`cifar10`目录），如下所示：
    ```
    % ll cifar10/cifar-10-batches-py/
    batches.meta
    data_batch_1
    data_batch_2
    data_batch_3
    data_batch_4
    data_batch_5
    readme.html
    test_batch
    ```
    > CIFAR-10数据集下载地址：<https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>

2. 导入转换数据集的工具类`Cifar10ToMR`。

    ```python
    from mindspore.mindrecord import Cifar10ToMR
    ```
3. 实例化`Cifar10ToMR`对象，调用`transform`接口，将CIFAR-10数据集转换为MindSpore数据格式。

    ```python
    CIFAR10_DIR = "./cifar10/cifar-10-batches-py"
    MINDRECORD_FILE = "./cifar10.mindrecord"
    cifar10_transformer = Cifar10ToMR(CIFAR10_DIR, MINDRECORD_FILE)
    cifar10_transformer.transform(['label'])
    ```
    其中，
    `CIFAR10_DIR`：CIFAR-10数据集的文件夹路径。
    `MINDRECORD_FILE`：输出的MindSpore数据格式文件路径。

### 转换CIFAR-100数据集
用户可以通过`Cifar100ToMR`类，将CIFAR-100原始数据转换为MindSpore数据格式。

1. 准备好CIFAR-100数据集，将文件解压至指定的目录（示例中将数据集保存到`cifar100`目录），如下所示：
    ```
    % ll cifar100/cifar-100-python/
    meta
    test
    train
    ```
    > CIFAR-100数据集下载地址：<https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz>

2. 导入转换数据集的工具类`Cifar100ToMR`。

    ```python
    from mindspore.mindrecord import Cifar100ToMR
    ```
3. 实例化`Cifar100ToMR`对象，调用`transform`接口，将CIFAR-100数据集转换为MindSpore数据格式。

    ```python
    CIFAR100_DIR = "./cifar100/cifar-100-python"
    MINDRECORD_FILE = "./cifar100.mindrecord"
    cifar100_transformer = Cifar100ToMR(CIFAR100_DIR, MINDRECORD_FILE)
    cifar100_transformer.transform(['fine_label', 'coarse_label'])
    ```
    其中，
    `CIFAR100_DIR`：CIFAR-100数据集的文件夹路径。
    `MINDRECORD_FILE`：输出的MindSpore数据格式文件路径。

### 转换ImageNet数据集

用户可以通过`ImageNetToMR`类，将ImageNet原始数据（图片、标注）转换为MindSpore数据格式。

1. 下载并按照要求准备好ImageNet数据集。

    > ImageNet数据集下载地址：<http://image-net.org/download>

    对下载后的ImageNet数据集，整理数据集组织形式为一个包含所有图片的文件夹，以及一个记录图片对应标签的映射文件。

    标签映射文件包含3列，分别为各类别图片目录、标签ID、标签名，用空格隔开，映射文件示例如下：
    ```
    n02119789 1 pen
    n02100735 2 notbook
    n02110185 3 mouse
    n02096294 4 orange
    ```

2. 导入转换数据集的工具类`ImageNetToMR`。

    ```python
    from mindspore.mindrecord import ImageNetToMR
    ```

3. 实例化`ImageNetToMR`对象，调用`transform`接口，将数据集转换为MindSpore数据格式。
    ```python
    IMAGENET_MAP_FILE = "./testImageNetDataWhole/labels_map.txt"
    IMAGENET_IMAGE_DIR = "./testImageNetDataWhole/images"
    MINDRECORD_FILE = "./testImageNetDataWhole/imagenet.mindrecord"
    PARTITION_NUMBER = 4
    imagenet_transformer = ImageNetToMR(IMAGENET_MAP_FILE, IMAGENET_IMAGE_DIR, MINDRECORD_FILE, PARTITION_NUMBER)
    imagenet_transformer.transform()
    ```
    其中，
    `IMAGENET_MAP_FILE`：ImageNetToMR数据集的标签映射文件路径。
    `IMAGENET_IMAGE_DIR`：包含ImageNet所有图片的文件夹路径。
    `MINDRECORD_FILE`：输出的MindSpore数据格式文件路径。

### 转换MNIST数据集
用户可以通过`MnistToMR`类，将MNIST原始数据转换为MindSpore数据格式。

1. 准备MNIST数据集，将下载好的文件放至指定的目录，如下所示：
    ```
    % ll mnist_data/
    train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz
    t10k-images-idx3-ubyte.gz
    t10k-labels-idx1-ubyte.gz
    ```
    > MNIST数据集下载地址：<http://yann.lecun.com/exdb/mnist>

2. 导入转换数据集的工具类`MnistToMR`。

    ```python
    from mindspore.mindrecord import MnistToMR
    ```
3. 实例化`MnistToMR`对象，调用`transform`接口，将MNIST数据集转换为MindSpore数据格式。

    ```python
    MNIST_DIR = "./mnist_data"
    MINDRECORD_FILE = "./mnist.mindrecord"
    mnist_transformer = MnistToMR(MNIST_DIR, MINDRECORD_FILE)
    mnist_transformer.transform()
    ```
    其中，
    `MNIST_DIR`：MNIST数据集的文件夹路径。
    `MINDRECORD_FILE`：输出的MindSpore数据格式文件路径。
