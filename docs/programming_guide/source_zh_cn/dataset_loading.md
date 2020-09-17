# 数据集加载

<!-- TOC -->

- [数据集加载](#数据集加载)
    - [概述](#概述)
    - [经典数据集加载](#经典数据集加载)
        - [MNIST数据集](#mnist数据集)
        - [CIFAR10/100数据集](#cifar10100数据集)
        - [VOC数据集](#voc数据集)
        - [COCO数据集](#coco数据集)
    - [特定格式数据集加载](#特定格式数据集加载)
        - [MindRecord数据格式](#mindrecord数据格式)
        - [Manifest数据格式](#manifest数据格式)
        - [TFRecord数据格式](#tfrecord数据格式)
        - [Numpy数据格式](#numpy数据格式)
        - [text数据格式](#text数据格式)
        - [CSV数据格式](#csv数据格式)
    - [自定义数据集加载](#自定义数据集加载)
        - [构造数据集生成函数](#构造数据集生成函数)
        - [构造可迭代的数据集类](#构造可迭代的数据集类)
        - [构造可随机访问的数据集类](#构造可随机访问的数据集类)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/api/source_zh_cn/programming_guide/dataset_loading.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

MindSpore支持加载图像领域常用的经典数据集，用户可以直接使用`mindspore.dataset`中对应的类实现数据集的加载。目前支持的经典数据集及对应的数据集类如下表所示。

|  图像数据集    | 数据集类  | 数据集简介 |
|  ----                    | ----  | ----           |
| MNIST               | MnistDataset | MNIST是一个大型手写数字图像数据集，拥有60,000张训练图像和10,000张测试图像，常用于训练各种图像处理系统。 |
| CIFAR-10          | Cifar10Dataset | CIFAR-10是一个微小图像数据集，包含10种类别下的60,000张32x32大小彩色图像，平均每种类别6,000张，其中5,000张为训练集，1,000张为测试集。 |
| CIFAR-100       | Cifar100Dataset | CIFAR-100与CIFAR-10类似，但拥有100种类别，平均每种类别600张，其中500张为训练集，100张为测试集。 |
|CelebA               | CelebADataset | CelebA是一个大型人脸图像数据集，包含超过200,000张名人人脸图像，每张图像拥有40个特征标记。 |
| PASCAL-VOC  | VOCDataset | PASCAL-VOC是一个经典图像数据集，被广泛用于目标检测、图像分割等计算机视觉领域。 |
| COCO                | CocoDataset | COCO是一个大型目标检测、图像分割、姿态估计数据集。 |
| CLUE                | CLUEDataset | CLUE是一个大型中文语义理解数据集。 |

MindSpore还支持加载多种数据存储格式下的数据集，用户可以直接使用`mindspore.dataset`中对应的类加载磁盘中的数据文件。目前支持的数据格式及对应加载方式如下表所示。

|  数据格式    | 数据集类  | 数据格式简介 |
|  ----                    | ----  | ----           |
| MindRecord | MindDataset | MindRecord是MindSpore的自研数据格式，具有读写高效、易于分布式处理等优势。 |
| Manifest | ManifestDataset | Manifest是华为ModelArts支持的一种数据格式，描述了原始文件和标注信息，可用于标注、训练、推理场景。 |
| TFRecord | TFRecordDataset | TFRecord是Tensorflow定义的一种二进制数据文件格式。 |
| Numpy | NumpySlicesDataset | Numpy数据源指的是已经读入内存中的Numpy arrays格式数据集。 |
| Text File | TextFileDataset | Text File指的是常见的文本格式数据。 |
| CSV File | CSVDataset | CSV指逗号分隔值，其文件以纯文本形式存储表格数据。 |

MindSpore也同样支持使用GeneratorDataset自定义数据集的加载方式，用户可以根据需要实现自己的数据集类。

更多详细的数据集加载接口说明，参见[API文档](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.dataset.html)。

## 经典数据集加载

### MNIST数据集

```python
# 通过MNIST API读取、解析MNIST数据集，并构建数据管道

import mindspore.dataset as ds

# 下载MNIST数据集，将其解压到MnistData目录
DATA_DIR = "MnistData/"

# 使用MnistDataset读取数据集，指定num_samples以获取5个样本数据
# shuffle参数为True时，是随机获取5个样本，每次运行的label结果可能不一致
dataset = ds.MnistDataset(DATA_DIR, num_samples=5, shuffle=True)

# 启动数据管道，输出5个样本数据
for data in dataset.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

```
Image shape: (28, 28, 1) , Label: 4
Image shape: (28, 28, 1) , Label: 9
Image shape: (28, 28, 1) , Label: 4
Image shape: (28, 28, 1) , Label: 0
Image shape: (28, 28, 1) , Label: 9
```

### CIFAR10/100数据集

```python
# 通过Cifar API读取、解析CIFAR数据集，并构建数据管道（以CIFAR10数据集为例）

import mindspore.dataset as ds

# 下载CIFAR10数据集，将其解压到CIFAR10Data目录
DATA_DIR = "Cifar10Data/"

# 指定一个顺序采样器SequentialSampler，按照读取顺序获取5个样本数据
sampler = ds.SequentialSampler(num_samples=5)

# 使用CIFAR10Dataset读取数据集，指定sampler为上述采样器
dataset = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

# 启动数据管道，输出5个样本数据
for data in dataset.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

```
Image shape: (32, 32, 3) , Label: 0
Image shape: (32, 32, 3) , Label: 1
Image shape: (32, 32, 3) , Label: 2
Image shape: (32, 32, 3) , Label: 3
Image shape: (32, 32, 3) , Label: 4
```

### VOC数据集

```python
# 通过VOC API读取、解析VOC数据集，并构建数据管道

import mindspore.dataset as ds

# 下载VOC数据集，将其解压到VOC2012目录
DATA_DIR = "VOC2012/"

# 使用VOCDataset读取数据集，指定为Segmentation任务，同时指定num_samples以获取2个样本数据
# decode参数会将读取的图像解码
dataset = ds.VOCDataset(DATA_DIR, task="Segmentation", mode="train", num_samples=2, decode=True, shuffle=False)
print("[Segmentation]:")
for data in dataset.create_dict_iterator():
    # 原图像
    print("image shape:", data["image"].shape)
    # 分割后图像
    print("target shape:", data["target"].shape)

# 接下来是Detection任务
dataset = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", num_samples=1, decode=True, shuffle=False)
print("[Detection]:")
for data in dataset.create_dict_iterator():
    # 原图像
    print("image shape:", data["image"].shape)
    # 目标框
    print("bbox shape:", data["bbox"].shape)
```

```
[Segmentation]:
image shape: (281, 500, 3)
target shape: (281, 500, 3)
image shape: (375, 500, 3)
target shape: (375, 500, 3)
[Detection]:
image shape: (442, 500, 3)
bbox shape: (2, 4)
```

### COCO数据集

```python
# 通过Coco API读取、解析Coco数据集，并构建数据管道

import mindspore.dataset as ds

# 下载Coco数据集，将其解压到CocoData目录
DATA_DIR = "COCO/train/"
ANNOTATION_FILE = "COCO/annotations/train.json"
KEYPOINT_FILE = "COCO/annotations/key_point.json"
PANOPTIC_FILE = "COCO/annotations/panoptic.json"

# 使用CocoDataset读取数据集，指定为Detection任务，同时指定num_samples以获取1个样本数据
dataset = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Detection", num_samples=1)
for data in dataset.create_dict_iterator():
    print("Detection:", data.keys())

# 让我们来观察一下，在指定Coco不同任务时，我们获取到的不同数据
# Stuff 任务
dataset = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Stuff", num_samples=1)
for data in dataset.create_dict_iterator():
    print("Stuff:", data.keys())

# Keypoint 任务
dataset = ds.CocoDataset(DATA_DIR, annotation_file=KEYPOINT_FILE, task="Keypoint", num_samples=1)
for data in dataset.create_dict_iterator():
    print("Keypoint:", data.keys())

# Panoptic 任务
dataset = ds.CocoDataset(DATA_DIR, annotation_file=PANOPTIC_FILE, task="Panoptic", num_samples=1)
for data in dataset.create_dict_iterator():
    print("Panoptic:", data.keys())
```

```
Detection: dict_keys(['bbox', 'image', 'iscrowd', 'category_id'])
Stuff: dict_keys(['segmentation', 'iscrowd', 'image'])
Keypoint: dict_keys(['keypoints', 'num_keypoints', 'image'])
Panoptic: dict_keys(['bbox', 'image', 'area', 'category_id', 'iscrowd'])
```

> 更多经典数据集加载接口说明，参见对应[API文档](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.dataset.html)。

## 特定格式数据集加载

### MindRecord数据格式

MindRecord是MindSpore的自研数据格式，具有更好的性能和特性。

>阅读[数据格式转换](https://www.mindspore.cn/api/zh-CN/master/programming_guide/dataset_conversion.html)章节，了解如何将数据集转化为MindSpore数据格式。

```python
import mindspore.dataset as ds

# 指定MindRecord数据格式地址
DATA_DIR = "mindrecord_dataset_path"
mindrecord_dataset = ds.MindDataset(DATA_DIR)

# 启动数据管道读取
for data in mindrecord_dataset.create_dict_iterator(output_numpy=True):
    print(data["label"])
```

### Manifest数据格式

Manifest是华为ModelArts支持的数据格式文件，详细说明请参见相关[文档](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0009.html)。

```python
import mindspore.dataset as ds

# 指定Manifest数据集地址
DATA_DIR = "manifest_dataset_path"
manifest_dataset = ds.ManifestDataset(DATA_DIR)

# 启动数据管道读取
for data in manifest_dataset.create_dict_iterator():
    print(data["label"])
```

### TFRecord数据格式

TFRecord是Tensorflow定义的一种二进制数据文件格式。

1. 传入数据集路径或`.tfrecord`文件列表，创建TFRecordDataset对象。

    ```python
    import mindspore.dataset as ds

    DATA_DIR = "tfrecord_dataset_path"
    dataset = ds.TFRecordDataset(DATA_DIR)
    ```

2. 用户可以选择通过创建Schema文件或Schema类，设定数据集格式及特征。

    - 创建Schema文件

    Schema文件示例：

    ```
    {
     "datasetType": "TF",
     "numRows": 3,
     "columns": {
         "image": {
             "type": "uint8",
             "rank": 1
             },
         "label" : {
             "type": "int64",
             "rank": 1
             }
         }
     }
    ```

        - `datasetType`: 数据格式的类型，这里`TF`是指TFrecord数据格式。

        - `columns`：列信息字段，需要根据数据集的实际列名定义，上面Schema文件示例中，数据集列为`image`和`label`两列。

        - `numRows`：行数信息字段，控制加载数据的最大行数。如果定义的行数大于实际行数，加载时则以实际行数为准。

    在创建TFRecordDataset时将Schema文件路径传入。

    ```python
    DATA_DIR = "tfrecord_dataset_path"
    SCHEMA_DIR = "dataset_schema_path/schema.json"
    dataset = ds.TFRecordDataset(DATA_DIR, schema=SCHEMA_DIR)
    ```

    - 创建Schema类

    ```python
    import mindspore.common.dtype as mstype
    schema = ds.Schema()
    schema.add_column('image', de_type=mstype.uint8)
    schema.add_column('label', de_type=mstype.int32)
    dataset = ds.TFRecordDataset(DATA_DIR, schema=schema)
    ```

### Numpy数据格式

如果所有数据已经读入内存，可以直接使用NumpySlicesDataset类将其加载。

- 加载Numpy arrays数据

    ```python
    # 从Numpy arrays构建数据管道

    import numpy as np
    import mindspore.dataset as ds

    # 使用numpy构建一个数组
    features, labels = np.random.sample((5, 2)), np.random.sample((5, 1))
    # 从numpy中构建数据管道
    # 注意：传入参数需要是一个tuple，即是(features, labels)；column_names用于指定生成的数据集名称为col1, col2
    data = (features, labels)
    dataset = ds.NumpySlicesDataset(data, column_names=["col1", "col2"], shuffle=False)

    # 启动数据管道
    for data in dataset:
        print(data[0], " ", data[1])
    ```

    ```
    [0.49893939 0.36348882]   [0.15234002]
    [0.83845534 0.19721032]   [0.94602561]
    [0.2361873  0.79506755]   [0.88118559]
    [0.71856343 0.16168491]   [0.55517421]
    [0.21266013 0.33411312]   [0.74858382]
    ```

- 加载Python list数据

    ```python
    # 从Python list构建数据管道

    import mindspore.dataset as ds

    # 构建一个list
    data1 = [[1, 2], [3, 4]]

    # 从list中构建数据管道
    # column_names用于指定生成的数据集名称为col1
    dataset = ds.NumpySlicesDataset(data1, column_names=["col1"], shuffle=False)

    # 启动数据管道
    for data in dataset:
        print(data[0])
    ```

    ```
    [1 2]
    [3 4]
    ```

- 加载Python dict数据

    ```python
    # 从Python dict构建数据管道

    import mindspore.dataset as ds

    # 构建一个dict
    data1 = {"a": [1, 2], "b": [3, 4]}

    # 从dict中构建数据管道
    # column_names用于指定生成的数据集名称为col1, col2
    dataset = ds.NumpySlicesDataset(data1, column_names=["col1", "col2"], shuffle=False)

    # 启动数据管道
    for data in dataset.create_dict_iterator():
        print(data)
    ```

    ```
    {'col1': Tensor(shape=[], dtype=Int64, value= 1), 'col2': Tensor(shape=[], dtype=Int64, value= 3)}
    {'col1': Tensor(shape=[], dtype=Int64, value= 2), 'col2': Tensor(shape=[], dtype=Int64, value= 4)}
    ```

### text数据格式

```python
import mindspore.dataset as ds

# 指定text数据格式地址
DATA_DIR = "text_file_path"
text_dataset = ds.TextFileDataset(DATA_DIR)

# 启动数据管道读取
for data in text_dataset.create_dict_iterator(output_numpy=True):
    print(data["label"])
```

### CSV数据格式

```python
import mindspore.dataset as ds

# 指定CSV数据格式地址
DATA_DIR = "csv_file_path"
csv_dataset = ds.CSVDataset(DATA_DIR)

# 启动数据管道读取
for data in csv_dataset.create_dict_iterator(output_numpy=True):
    print(data["label"])
```

>更多数据格式文件加载说明，参见对应[API文档](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.dataset.html)。

## 自定义数据集加载

对于目前MindSpore不支持直接加载的数据集，可以通过构造GeneratorDataset对象实现自定义方式的加载，或者将其转换成MindRecord数据格式。目前自定义数据集加载有以下几种方式。

### 构造数据集生成函数

构造生成函数定义数据返回方式，再使用此函数构建自定义数据集对象。

```python
import numpy as np
import mindspore.dataset as ds

# 随机生成一个数据集
np.random.seed(58)
data = np.random.sample((5, 2))
label = np.random.sample((5, 1))

# 自定义数据返回方式
def GeneratorFunc():
    for i in range(5):
        yield (data[i], label[i])

# 构建自定义数据集对象
dataset = ds.GeneratorDataset(GeneratorFunc, ["data", "label"])

for data in dataset.create_dict_iterator():
    print(data["data"], data["label"])
```

```
[0.36510558 0.45120592] [0.78888122]
[0.49606035 0.07562207] [0.38068183]
[0.57176158 0.28963401] [0.16271622]
[0.30880446 0.37487617] [0.54738768]
[0.81585667 0.96883469] [0.77994068]
```

### 构造可迭代的数据集类

构造数据集类实现`__iter__`和`__next__`方法，再使用此类的对象构建自定义数据集对象。

```python
import numpy as np
import mindspore.dataset as ds

class IterDatasetGenerator:
    def __init__(self):
        np.random.seed(58)
        self.__index = 0
        self.__data = np.random.sample((5, 2))
        self.__label = np.random.sample((5, 1))

    def __next__(self):
        if self.__index >= len(self.__data):
            raise StopIteration
        else:
            item = (self.__data[self.__index], self.__label[self.__index])
            self.__index += 1
            return item

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.__data)

dataset_generator = IterDatasetGenerator()
dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)

for data in dataset.create_dict_iterator():
    print(data["data"], data["label"])
```

```
[0.36510558 0.45120592] [0.78888122]
[0.49606035 0.07562207] [0.38068183]
[0.57176158 0.28963401] [0.16271622]
[0.30880446 0.37487617] [0.54738768]
[0.81585667 0.96883469] [0.77994068]
```

### 构造可随机访问的数据集类

构造数据集类实现`__getitem__`方法，再使用此类的对象构建自定义数据集对象。

```python
import numpy as np
import mindspore.dataset as ds

class GetDatasetGenerator:
    def __init__(self):
        np.random.seed(58)
        self.__data = np.random.sample((5, 2))
        self.__label = np.random.sample((5, 1))

    def __getitem__(self, index):
        return (self.__data[index], self.__label[index])

    def __len__(self):
        return len(self.__data)

dataset_generator = GetDatasetGenerator()
dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)

for data in dataset.create_dict_iterator():
    print(data["data"], data["label"])
```

```
[0.36510558 0.45120592] [0.78888122]
[0.49606035 0.07562207] [0.38068183]
[0.57176158 0.28963401] [0.16271622]
[0.30880446 0.37487617] [0.54738768]
[0.81585667 0.96883469] [0.77994068]
```

如果用户希望使用分布式训练，则需要在此方式的基础上，改为在采样器类中实现`__iter__`方法，每次返回采样数据的索引。

```python
import math

class MySampler():
    def __init__(self, dataset, local_rank, world_size):
        self.__num_data = len(dataset)
        self.__local_rank = local_rank
        self.__world_size = world_size
        self.samples_per_rank = int(math.ceil(self.__num_data / float(self.__world_size)))
        self.total_num_samples = self.samples_per_rank * self.__world_size

    def __iter__(self):
        indices = list(range(self.__num_data))
        indices.extend(indices[:self.total_num_samples-len(indices)])
        indices = indices[self.__local_rank:self.total_num_samples:self.__world_size]
        return iter(indices)

    def __len__(self):
        return self.samples_per_rank

dataset_generator = GetDatasetGenerator()
sampler = MySampler(dataset_generator, local_rank=0, world_size=2)
dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False, sampler=sampler)

for data in dataset.create_dict_iterator():
    print(data["data"], data["label"])
```

```
[0.36510558 0.45120592] [0.78888122]
[0.57176158 0.28963401] [0.16271622]
[0.81585667 0.96883469] [0.77994068]
```
