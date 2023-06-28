# 数据集加载

<a href="https://gitee.com/mindspore/docs/blob/r1.0/docs/programming_guide/source_zh_cn/dataset_loading.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 概述

MindSpore支持加载图像领域常用的数据集，用户可以直接使用`mindspore.dataset`中对应的类实现数据集的加载。目前支持的常用数据集及对应的数据集类如下表所示。

|  图像数据集    | 数据集类  | 数据集简介 |
|  ----                    | ----  | ----           |
| MNIST               | MnistDataset | MNIST是一个大型手写数字图像数据集，拥有60,000张训练图像和10,000张测试图像，常用于训练各种图像处理系统。 |
| CIFAR-10          | Cifar10Dataset | CIFAR-10是一个微小图像数据集，包含10种类别下的60,000张32x32大小彩色图像，平均每种类别6,000张，其中5,000张为训练集，1,000张为测试集。 |
| CIFAR-100       | Cifar100Dataset | CIFAR-100与CIFAR-10类似，但拥有100种类别，平均每种类别600张，其中500张为训练集，100张为测试集。 |
| CelebA              | CelebADataset | CelebA是一个大型人脸图像数据集，包含超过200,000张名人人脸图像，每张图像拥有40个特征标记。 |
| PASCAL-VOC  | VOCDataset | PASCAL-VOC是一个常用图像数据集，被广泛用于目标检测、图像分割等计算机视觉领域。 |
| COCO                | CocoDataset | COCO是一个大型目标检测、图像分割、姿态估计数据集。 |
| CLUE                | CLUEDataset | CLUE是一个大型中文语义理解数据集。 |

MindSpore还支持加载多种数据存储格式下的数据集，用户可以直接使用`mindspore.dataset`中对应的类加载磁盘中的数据文件。目前支持的数据格式及对应加载方式如下表所示。

|  数据格式    | 数据集类  | 数据格式简介 |
|  ----                    | ----  | ----           |
| MindRecord | MindDataset | MindRecord是MindSpore的自研数据格式，具有读写高效、易于分布式处理等优势。 |
| Manifest | ManifestDataset | Manifest是华为ModelArts支持的一种数据格式，描述了原始文件和标注信息，可用于标注、训练、推理场景。 |
| TFRecord | TFRecordDataset | TFRecord是TensorFlow定义的一种二进制数据文件格式。 |
| NumPy | NumpySlicesDataset | NumPy数据源指的是已经读入内存中的NumPy arrays格式数据集。 |
| Text File | TextFileDataset | Text File指的是常见的文本格式数据。 |
| CSV File | CSVDataset | CSV指逗号分隔值，其文件以纯文本形式存储表格数据。 |

MindSpore也同样支持使用`GeneratorDataset`自定义数据集的加载方式，用户可以根据需要实现自己的数据集类。

> 更多详细的数据集加载接口说明，参见[API文档](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.dataset.html)。

## 常用数据集加载

下面将介绍几种常用数据集的加载方式。

### CIFAR-10/100数据集

下载[CIFAR-10数据集](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)并解压，目录结构如下。

```text
└─cifar-10-batches-bin
    ├── batches.meta.txt
    ├── data_batch_1.bin
    ├── data_batch_2.bin
    ├── data_batch_3.bin
    ├── data_batch_4.bin
    ├── data_batch_5.bin
    ├── readme.html
    └── test_batch.bin
```

下面的样例通过`Cifar10Dataset`接口加载CIFAR-10数据集，使用顺序采样器获取其中5个样本，然后展示了对应图片的形状和标签。

CIFAR-100数据集和MNIST数据集的加载方式也与之类似。

```python
import mindspore.dataset as ds

DATA_DIR = "cifar-10-batches-bin/"

sampler = ds.SequentialSampler(num_samples=5)
dataset = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)

for data in dataset.create_dict_iterator():
    print("Image shape:", data['image'].shape, ", Label:", data['label'])
```

输出结果如下：

```text
Image shape: (32, 32, 3) , Label: 6
Image shape: (32, 32, 3) , Label: 9
Image shape: (32, 32, 3) , Label: 9
Image shape: (32, 32, 3) , Label: 4
Image shape: (32, 32, 3) , Label: 1
```

### VOC数据集

VOC数据集有多个版本，此处以VOC2012为例。下载[VOC2012数据集](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)并解压，目录结构如下。

```text
└─ VOCtrainval_11-May-2012
    └── VOCdevkit
        └── VOC2012
            ├── Annotations
            ├── ImageSets
            ├── JPEGImages
            ├── SegmentationClass
            └── SegmentationObject
```

下面的样例通过`VOCDataset`接口加载VOC2012数据集，分别演示了将任务指定为分割（Segmentation）和检测（Detection）时的原始图像形状和目标形状。

```python
import mindspore.dataset as ds

DATA_DIR = "VOCtrainval_11-May-2012/VOCdevkit/VOC2012/"

dataset = ds.VOCDataset(DATA_DIR, task="Segmentation", usage="train", num_samples=2, decode=True, shuffle=False)

print("[Segmentation]:")
for data in dataset.create_dict_iterator():
    print("image shape:", data["image"].shape)
    print("target shape:", data["target"].shape)

dataset = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", num_samples=1, decode=True, shuffle=False)

print("[Detection]:")
for data in dataset.create_dict_iterator():
    print("image shape:", data["image"].shape)
    print("bbox shape:", data["bbox"].shape)
```

输出结果如下：

```text
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

COCO数据集有多个版本，此处以COCO2017的验证数据集为例。下载COCO2017的[验证集](http://images.cocodataset.org/zips/val2017.zip)、[检测任务标注](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)和[全景分割任务标注](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip)并解压，只取其中的验证集部分，按以下目录结构存放。

```text
└─ COCO
    ├── val2017
    └── annotations
        ├── instances_val2017.json
        ├── panoptic_val2017.json
        └── person_keypoints_val2017.json
```

下面的样例通过`CocoDataset`接口加载COCO2017数据集，分别演示了将任务指定为目标检测（Detection）、背景分割（Stuff）、关键点检测（Keypoint）和全景分割（Panoptic）时获取到的不同数据。

```python
import mindspore.dataset as ds

DATA_DIR = "COCO/val2017/"
ANNOTATION_FILE = "COCO/annotations/instances_val2017.json"
KEYPOINT_FILE = "COCO/annotations/person_keypoints_val2017.json"
PANOPTIC_FILE = "COCO/annotations/panoptic_val2017.json"

dataset = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Detection", num_samples=1)
for data in dataset.create_dict_iterator():
    print("Detection:", data.keys())

dataset = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Stuff", num_samples=1)
for data in dataset.create_dict_iterator():
    print("Stuff:", data.keys())

dataset = ds.CocoDataset(DATA_DIR, annotation_file=KEYPOINT_FILE, task="Keypoint", num_samples=1)
for data in dataset.create_dict_iterator():
    print("Keypoint:", data.keys())

dataset = ds.CocoDataset(DATA_DIR, annotation_file=PANOPTIC_FILE, task="Panoptic", num_samples=1)
for data in dataset.create_dict_iterator():
    print("Panoptic:", data.keys())
```

输出结果如下：

```text
Detection: dict_keys(['image', 'bbox', 'category_id', 'iscrowd'])
Stuff: dict_keys(['image', 'segmentation', 'iscrowd'])
Keypoint: dict_keys(['image', 'keypoints', 'num_keypoints'])
Panoptic: dict_keys(['image', 'bbox', 'category_id', 'iscrowd', 'area'])
```

## 特定格式数据集加载

下面将介绍几种特定格式数据集文件的加载方式。

### MindRecord数据格式

MindRecord是MindSpore定义的一种数据格式，使用MindRecord能够获得更好的性能提升。

> 阅读[数据格式转换](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.0/dataset_conversion.html)章节，了解如何将数据集转化为MindSpore数据格式。

下面的样例通过`MindDataset`接口加载MindRecord文件，并展示已加载数据的标签。

```python
import mindspore.dataset as ds

DATA_FILE = ["mindrecord_file_0", "mindrecord_file_1", "mindrecord_file_2"]
mindrecord_dataset = ds.MindDataset(DATA_FILE)

for data in mindrecord_dataset.create_dict_iterator(output_numpy=True):
    print(data["label"])
```

### Manifest数据格式

Manifest是华为ModelArts支持的数据格式文件，详细说明请参见[Manifest文档](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0009.html)。

下面的样例通过`ManifestDataset`接口加载Manifest文件，并展示已加载数据的标签。

```python
import mindspore.dataset as ds

DATA_FILE = "manifest_file"
manifest_dataset = ds.ManifestDataset(DATA_FILE)

for data in manifest_dataset.create_dict_iterator():
    print(data["label"])
```

### TFRecord数据格式

TFRecord是TensorFlow定义的一种二进制数据文件格式。

下面的样例通过`TFRecordDataset`接口加载TFRecord文件，并介绍了两种不同的数据集格式设定方案。

1. 传入数据集路径或TFRecord文件列表，创建`TFRecordDataset`对象。

    ```python
    import mindspore.dataset as ds

    DATA_FILE = ["tfrecord_file_0", "tfrecord_file_1", "tfrecord_file_2"]
    tfrecord_dataset = ds.TFRecordDataset(DATA_FILE)
    ```

2. 用户可以通过编写Schema文件或创建Schema对象，设定数据集格式及特征。

    - 编写Schema文件

        将数据集格式和特征按JSON格式写入Schema文件，示例如下：

        ```json
        {
         "columns": {
             "image": {
                 "type": "uint8",
                 "rank": 1
                 },
             "label" : {
                 "type": "string",
                 "rank": 1
                 }
             "id" : {
                 "type": "int64",
                 "rank": 0
                 }
             }
         }
        ```

        - `columns`：列信息字段，需要根据数据集的实际列名定义。上面的示例中，数据集列为`image`、`label`和`id`。

        然后在创建`TFRecordDataset`时将Schema文件路径传入。

        ```python
        SCHEMA_DIR = "dataset_schema_path/schema.json"
        tfrecord_dataset = ds.TFRecordDataset(DATA_FILE, schema=SCHEMA_DIR)
        ```

    - 创建Schema对象

        创建Schema对象，为其添加自定义字段，然后在创建数据集对象时传入。

        ```python
        import mindspore.common.dtype as mstype
        schema = ds.Schema()
        schema.add_column('image', de_type=mstype.uint8)
        schema.add_column('label', de_type=mstype.int32)
        tfrecord_dataset = ds.TFRecordDataset(DATA_FILE, schema=schema)
        ```

### NumPy数据格式

如果所有数据已经读入内存，可以直接使用`NumpySlicesDataset`类将其加载。

下面的样例分别介绍了通过`NumpySlicesDataset`加载arrays数据、 list数据和dict数据的方式。

- 加载NumPy arrays数据

    ```python
    import numpy as np
    import mindspore.dataset as ds

    np.random.seed(58)
    features, labels = np.random.sample((5, 2)), np.random.sample((5, 1))

    data = (features, labels)
    dataset = ds.NumpySlicesDataset(data, column_names=["col1", "col2"], shuffle=False)

    for data in dataset:
        print(data[0], data[1])
    ```

    输出结果如下：

    ```text
    [0.36510558 0.45120592] [0.78888122]
    [0.49606035 0.07562207] [0.38068183]
    [0.57176158 0.28963401] [0.16271622]
    [0.30880446 0.37487617] [0.54738768]
    [0.81585667 0.96883469] [0.77994068]
    ```

- 加载Python list数据

    ```python

    import mindspore.dataset as ds

    data1 = [[1, 2], [3, 4]]

    dataset = ds.NumpySlicesDataset(data1, column_names=["col1"], shuffle=False)

    for data in dataset:
        print(data[0])
    ```

    输出结果如下：

    ```text
    [1 2]
    [3 4]
    ```

- 加载Python dict数据

    ```python
    import mindspore.dataset as ds

    data1 = {"a": [1, 2], "b": [3, 4]}

    dataset = ds.NumpySlicesDataset(data1, column_names=["col1", "col2"], shuffle=False)

    for data in dataset.create_dict_iterator():
        print(data)
    ```

    输出结果如下：

    ```text
    {'col1': Tensor(shape=[], dtype=Int64, value= 1), 'col2': Tensor(shape=[], dtype=Int64, value= 3)}
    {'col1': Tensor(shape=[], dtype=Int64, value= 2), 'col2': Tensor(shape=[], dtype=Int64, value= 4)}
    ```

### CSV数据格式

下面的样例通过`CSVDataset`加载CSV格式数据集文件，并展示了已加载数据的标签。

Text格式数据集文件的加载方式与CSV文件类似。

```python
import mindspore.dataset as ds

DATA_FILE = ["csv_file_0", "csv_file_1", "csv_file_2"]
csv_dataset = ds.CSVDataset(DATA_FILE)

for data in csv_dataset.create_dict_iterator(output_numpy=True):
    print(data["1"])
```

## 自定义数据集加载

对于目前MindSpore不支持直接加载的数据集，可以通过构造`GeneratorDataset`对象实现自定义方式的加载，或者将其转换成MindRecord数据格式。目前自定义数据集加载有以下几种方式。

### 构造数据集生成函数

构造生成函数定义数据返回方式，再使用此函数构建自定义数据集对象。

```python
import numpy as np
import mindspore.dataset as ds

np.random.seed(58)
data = np.random.sample((5, 2))
label = np.random.sample((5, 1))

def GeneratorFunc():
    for i in range(5):
        yield (data[i], label[i])

dataset = ds.GeneratorDataset(GeneratorFunc, ["data", "label"])

for sample in dataset.create_dict_iterator():
    print(sample["data"], sample["label"])
```

输出结果如下：

```text
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

输出结果如下：

```text
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

输出结果如下：

```text
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

输出结果如下：

```text
[0.36510558 0.45120592] [0.78888122]
[0.57176158 0.28963401] [0.16271622]
[0.81585667 0.96883469] [0.77994068]
```
