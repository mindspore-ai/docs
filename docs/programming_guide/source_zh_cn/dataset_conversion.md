# MindSpore数据格式转换

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/programming_guide/source_zh_cn/dataset_conversion.md)

## 概述

用户可以将非标准的数据集和常用的数据集转换为MindSpore数据格式，即MindRecord，从而方便地加载到MindSpore中进行训练。同时，MindSpore在部分场景做了性能优化，使用MindRecord可以获得更好的性能。

## 非标准数据集转换MindRecord

下面主要介绍如何将CV类数据和NLP类数据转换为MindRecord，并通过`MindDataset`实现MindRecord文件的读取。

### 转换CV类数据集

本示例主要介绍用户如何将自己的CV类数据集转换成MindRecord，并使用`MindDataset`读取。

本示例首先创建一个包含100条记录的MindRecord文件，其样本包含`file_name`（字符串）、
`label`（整形）、 `data`（二进制）三个字段，然后使用`MindDataset`读取该MindRecord文件。

```python
from io import BytesIO
import os
import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter
import mindspore.dataset.vision.c_transforms as vision
from PIL import Image

mindrecord_filename = "test.mindrecord"

if os.path.exists(mindrecord_filename):
    os.remove(mindrecord_filename)
    os.remove(mindrecord_filename + ".db")

writer = FileWriter(file_name=mindrecord_filename, shard_num=1)

cv_schema = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
writer.add_schema(cv_schema, "it is a cv dataset")

writer.add_index(["file_name", "label"])

data = []
for i in range(100):
    i += 1

    sample = {}
    white_io = BytesIO()
    Image.new('RGB', (i*10, i*10), (255, 255, 255)).save(white_io, 'JPEG')
    image_bytes = white_io.getvalue()
    sample['file_name'] = str(i) + ".jpg"
    sample['label'] = i
    sample['data'] = white_io.getvalue()

    data.append(sample)
    if i % 10 == 0:
        writer.write_raw_data(data)
        data = []

if data:
    writer.write_raw_data(data)

writer.commit()

data_set = ds.MindDataset(dataset_file=mindrecord_filename)
decode_op = vision.Decode()
data_set = data_set.map(operations=decode_op, input_columns=["data"], num_parallel_workers=2)
count = 0
for item in data_set.create_dict_iterator(output_numpy=True):
    print("sample: {}".format(item))
    count += 1
print("Got {} samples".format(count))
```

### 转换NLP类数据集

本示例主要介绍用户如何将自己的NLP类数据集转换成MindRecord，并使用`MindDataset`读取。为了方便展示，此处略去了将文本转换成字典序的预处理过程。

本示例首先创建一个包含100条记录的MindRecord文件，其样本包含八个字段，均为整形数组，然后使用`MindDataset`读取该MindRecord文件。

```python
import os
import numpy as np
import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter

mindrecord_filename = "test.mindrecord"

if os.path.exists(mindrecord_filename):
    os.remove(mindrecord_filename)
    os.remove(mindrecord_filename + ".db")

writer = FileWriter(file_name=mindrecord_filename, shard_num=1)

nlp_schema = {"source_sos_ids": {"type": "int64", "shape": [-1]},
            "source_sos_mask": {"type": "int64", "shape": [-1]},
            "source_eos_ids": {"type": "int64", "shape": [-1]},
            "source_eos_mask": {"type": "int64", "shape": [-1]},
            "target_sos_ids": {"type": "int64", "shape": [-1]},
            "target_sos_mask": {"type": "int64", "shape": [-1]},
            "target_eos_ids": {"type": "int64", "shape": [-1]},
            "target_eos_mask": {"type": "int64", "shape": [-1]}}
writer.add_schema(nlp_schema, "it is a preprocessed nlp dataset")

data = []
for i in range(100):
    i += 1

    sample = {"source_sos_ids": np.array([i, i+1, i+2, i+3, i+4], dtype=np.int64),
            "source_sos_mask": np.array([i*1, i*2, i*3, i*4, i*5, i*6, i*7], dtype=np.int64),
            "source_eos_ids": np.array([i+5, i+6, i+7, i+8, i+9, i+10], dtype=np.int64),
            "source_eos_mask": np.array([19, 20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int64),
            "target_sos_ids": np.array([28, 29, 30, 31, 32], dtype=np.int64),
            "target_sos_mask": np.array([33, 34, 35, 36, 37, 38], dtype=np.int64),
            "target_eos_ids": np.array([39, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
            "target_eos_mask": np.array([48, 49, 50, 51], dtype=np.int64)}

    data.append(sample)
    if i % 10 == 0:
      writer.write_raw_data(data)
      data = []

if data:
    writer.write_raw_data(data)

writer.commit()

data_set = ds.MindDataset(dataset_file=mindrecord_filename)
count = 0
for item in data_set.create_dict_iterator():
    print("sample: {}".format(item))
    count += 1
print("Got {} samples".format(count))
```

## 常用数据集转换MindRecord

MindSpore提供转换常用数据集的工具类，能够将常用的数据集转换为MindRecord。常用数据集及其对应的工具类列表如下。

| 数据集 | 格式转换工具类 |
| -------- | ------------ |
| CIFAR-10 | Cifar10ToMR |
| CIFAR-100 | Cifar100ToMR |
| ImageNet | ImageNetToMR |
| MNIST | MnistToMR |
| TFRecord | TFRecordToMR |
| CSV File | CsvToMR |

更多数据集转换的详细说明可参见[API文档](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.mindrecord.html)。

### 转换CIFAR-10数据集

用户可以通过`Cifar10ToMR`类，将CIFAR-10原始数据转换为MindRecord，并使用`MindDataset`读取。

1. 下载[CIFAR-10数据集](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)并解压，其目录结构如下所示。

    ```text
    └─cifar-10-batches-py
        ├─batches.meta
        ├─data_batch_1
        ├─data_batch_2
        ├─data_batch_3
        ├─data_batch_4
        ├─data_batch_5
        ├─readme.html
        └─test_batch
    ```

2. 导入数据集转换工具类`Cifar10ToMR`。

    ```python
    from mindspore.mindrecord import Cifar10ToMR
    ```

3. 创建`Cifar10ToMR`对象，调用`transform`接口，将CIFAR-10数据集转换为MindRecord。

    ```python
    CIFAR10_DIR = "./cifar-10-batches-py"
    MINDRECORD_FILE = "./cifar10.mindrecord"
    cifar10_transformer = Cifar10ToMR(CIFAR10_DIR, MINDRECORD_FILE)
    cifar10_transformer.transform(['label'])
    ```

    **参数说明：**
    - `CIFAR10_DIR`：CIFAR-10数据集的文件夹路径。
    - `MINDRECORD_FILE`：输出的MindRecord文件路径。

4. 通过`MindDataset`读取MindRecord。

    ```python
    import mindspore.dataset as ds
    import mindspore.dataset.vision.c_transforms as vision

    data_set = ds.MindDataset(dataset_file=MINDRECORD_FILE)
    decode_op = vision.Decode()
    data_set = data_set.map(operations=decode_op, input_columns=["data"], num_parallel_workers=2)
    count = 0
    for item in data_set.create_dict_iterator(output_numpy=True):
        count += 1
    print("Got {} samples".format(count))
    ```

### 转换ImageNet数据集

用户可以通过`ImageNetToMR`类，将ImageNet原始数据（图片、标注）转换为MindRecord，并使用`MindDataset`读取。

1. 下载[ImageNet数据集](http://image-net.org/download)，将所有图片存放在`images/`文件夹，用一个映射文件`labels_map.txt`记录图片和标签的对应关系。映射文件包含2列，分别为各类别图片目录和标签ID，用空格隔开，映射文件示例如下：

    ```text
    n01440760 0
    n01443537 1
    n01484850 2
    n01491361 3
    n01494475 4
    n01496331 5
    ```

    文件目录结构如下所示：

    ```text
    ├─ labels_map.txt
    └─ images
        └─ ......
    ```

2. 导入数据集转换工具类`ImageNetToMR`。

    ```python
    from mindspore.mindrecord import ImageNetToMR
    ```

3. 创建`ImageNetToMR`对象，调用`transform`接口，将数据集转换为MindRecord。

    ```python
    IMAGENET_MAP_FILE = "./labels_map.txt"
    IMAGENET_IMAGE_DIR = "./images/"
    MINDRECORD_FILE = "./imagenet.mindrecord"
    PARTITION_NUMBER = 8
    imagenet_transformer = ImageNetToMR(IMAGENET_MAP_FILE, IMAGENET_IMAGE_DIR, MINDRECORD_FILE, PARTITION_NUMBER)
    imagenet_transformer.transform()
    ```

    **参数说明：**
    - `IMAGENET_MAP_FILE`：ImageNet数据集标签映射文件的路径。
    - `IMAGENET_IMAGE_DIR`：包含ImageNet所有图片的文件夹路径。
    - `MINDRECORD_FILE`：输出的MindRecord文件路径。

4. 通过`MindDataset`读取MindRecord。

    ```python
    import mindspore.dataset as ds
    import mindspore.dataset.vision.c_transforms as vision

    data_set = ds.MindDataset(dataset_file=MINDRECORD_FILE + "0")
    decode_op = vision.Decode()
    data_set = data_set.map(operations=decode_op, input_columns=["data"], num_parallel_workers=2)
    count = 0
    for item in data_set.create_dict_iterator(output_numpy=True):
        print("sample: {}".format(item))
        count += 1
    print("Got {} samples".format(count))
    ```

### 转换CSV数据集

本示例首先创建一个包含5条记录的CSV文件，然后通过`CsvToMR`工具类将CSV文件转换为MindRecord，并最终通过`MindDataset`将其读取出来。

```python
import csv
import os
import mindspore.dataset as ds
from mindspore.mindrecord import CsvToMR

CSV_FILE_NAME = "test.csv"
MINDRECORD_FILE_NAME = "test.mindrecord"
PARTITION_NUM = 1

def generate_csv():
    headers = ["id", "name", "math", "english"]
    rows = [(1, "Lily", 78.5, 90),
          (2, "Lucy", 99, 85.2),
          (3, "Mike", 65, 71),
          (4, "Tom", 95, 99),
          (5, "Jeff", 85, 78.5)]
    with open(CSV_FILE_NAME, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

generate_csv()

if os.path.exists(MINDRECORD_FILE_NAME):
    os.remove(MINDRECORD_FILE_NAME)
    os.remove(MINDRECORD_FILE_NAME + ".db")

csv_transformer = CsvToMR(CSV_FILE_NAME, MINDRECORD_FILE_NAME, partition_number=PARTITION_NUM)

csv_transformer.transform()

assert os.path.exists(MINDRECORD_FILE_NAME)
assert os.path.exists(MINDRECORD_FILE_NAME + ".db")

data_set = ds.MindDataset(dataset_file=MINDRECORD_FILE_NAME)
count = 0
for item in data_set.create_dict_iterator(output_numpy=True):
    print("sample: {}".format(item))
    count += 1
print("Got {} samples".format(count))
```

### 转换TFRecord数据集

> 目前只支持TensorFlow 2.1.0及以上版本。

本示例首先通过TensorFlow创建一个TFRecord文件，然后通过`TFRecordToMR`工具类将TFRecord文件转换为MindRecord，最后通过`MindDataset`将其读取出来，并使用`Decode`算子对`image_bytes`字段进行解码。

```python
import collections
from io import BytesIO
import os
import mindspore.dataset as ds
from mindspore.mindrecord import TFRecordToMR
import mindspore.dataset.vision.c_transforms as vision
from PIL import Image
import tensorflow as tf

TFRECORD_FILE_NAME = "test.tfrecord"
MINDRECORD_FILE_NAME = "test.mindrecord"
PARTITION_NUM = 1

def generate_tfrecord():
    def create_int_feature(values):
        if isinstance(values, list):
            feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        else:
            feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))
        return feature

    def create_float_feature(values):
        if isinstance(values, list):
            feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        else:
            feature = tf.train.Feature(float_list=tf.train.FloatList(value=[values]))
        return feature

    def create_bytes_feature(values):
        if isinstance(values, bytes):
            white_io = BytesIO()
            Image.new('RGB', (10, 10), (255, 255, 255)).save(white_io, 'JPEG')
            image_bytes = white_io.getvalue()
            feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
        else:
            feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(values, encoding='utf-8')]))
        return feature

    writer = tf.io.TFRecordWriter(TFRECORD_FILE_NAME)

    example_count = 0
    for i in range(10):
        file_name = "000" + str(i) + ".jpg"
        image_bytes = bytes(str("aaaabbbbcccc" + str(i)), encoding="utf-8")
        int64_scalar = i
        float_scalar = float(i)
        int64_list = [i, i+1, i+2, i+3, i+4, i+1234567890]
        float_list = [float(i), float(i+1), float(i+2.8), float(i+3.2),
                    float(i+4.4), float(i+123456.9), float(i+98765432.1)]

        features = collections.OrderedDict()
        features["file_name"] = create_bytes_feature(file_name)
        features["image_bytes"] = create_bytes_feature(image_bytes)
        features["int64_scalar"] = create_int_feature(int64_scalar)
        features["float_scalar"] = create_float_feature(float_scalar)
        features["int64_list"] = create_int_feature(int64_list)
        features["float_list"] = create_float_feature(float_list)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        example_count += 1
    writer.close()
    print("Write {} rows in tfrecord.".format(example_count))

generate_tfrecord()

feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
              "image_bytes": tf.io.FixedLenFeature([], tf.string),
              "int64_scalar": tf.io.FixedLenFeature([], tf.int64),
              "float_scalar": tf.io.FixedLenFeature([], tf.float32),
              "int64_list": tf.io.FixedLenFeature([6], tf.int64),
              "float_list": tf.io.FixedLenFeature([7], tf.float32),
              }

if os.path.exists(MINDRECORD_FILE_NAME):
    os.remove(MINDRECORD_FILE_NAME)
    os.remove(MINDRECORD_FILE_NAME + ".db")

tfrecord_transformer = TFRecordToMR(TFRECORD_FILE_NAME, MINDRECORD_FILE_NAME, feature_dict, ["image_bytes"])
tfrecord_transformer.transform()

assert os.path.exists(MINDRECORD_FILE_NAME)
assert os.path.exists(MINDRECORD_FILE_NAME + ".db")

data_set = ds.MindDataset(dataset_file=MINDRECORD_FILE_NAME)
decode_op = vision.Decode()
data_set = data_set.map(operations=decode_op, input_columns=["image_bytes"], num_parallel_workers=2)
count = 0
for item in data_set.create_dict_iterator(output_numpy=True):
    print("sample: {}".format(item))
    count += 1
print("Got {} samples".format(count))
```
