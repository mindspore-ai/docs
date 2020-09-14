# MindSpore数据格式转换

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [MindSpore数据格式转换](#mindspore数据格式转换)
    - [概述](#概述)
	- [非标准数据集转换MindRecord](#非标准数据集转换mindrecord)
		- [CV类数据集](#cv类数据集)
		- [NLP类数据集](#nlp类数据集)
	- [常用数据集转换MindRecord](#常用数据集转换mindrecord)
		- [转换CIFAR-10数据集](#转换cifar-10数据集)
		- [转换CIFAR-100数据集](#转换cifar-100数据集)
		- [转换ImageNet数据集](#转换imagenet数据集)
        - [转换MNIST数据集](#转换mnist数据集)
		- [转换CSV数据集](#转换csv数据集)
		- [转换TFRecord数据集](#转换tfrecord数据集)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/api/source_zh_cn/programming_guide/dataset_conversion.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

用户可以将非标准的数据集和常见的经典数据集转换为MindSpore数据格式即MindRecord，从而方便地加载到MindSpore中进行训练。同时，MindSpore在部分场景做了性能优化，使用MindSpore数据格式可以获得更好的性能体验。

## 非标准数据集转换MindRecord

主要介绍如何将CV类数据和NLP类数据转换为MindRecord格式，并通过MindDataset实现MindRecoed格式文件的读取。

### CV类数据集

  ```python
  """
  示例说明：本示例主要介绍用户如何将自己的CV类数据集转换成MindRecoed格式，并使用MindDataset读取。
  详细步骤：
  1. 创建一个包含100条记录的MindRecord文件，其样本包含file_name（字符串）, label（整形）, data（二进制）三个字段；
  2. 使用MindDataset读取MindRecord文件。
  """

  from io import BytesIO
  import os
  import mindspore.dataset as ds
  from mindspore.mindrecord import FileWriter
  import mindspore.dataset.transforms.vision.c_transforms as vision
  from PIL import Image

  ################################ 生成MindRecord文件 ################################

  mindrecord_filename = "test.mindrecord"

  # 如果存在MindRecord文件，则需要先删除
  if os.path.exists(mindrecord_filename):
      os.remove(mindrecord_filename)
      os.remove(mindrecord_filename + ".db")

  # 创建写对象，将会生成 mindrecord_filename 和 mindrecord_filename.db 两个文件
  writer = FileWriter(file_name=mindrecord_filename, shard_num=1)

  # 定义数据集Schema
  cv_schema = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
  writer.add_schema(cv_schema, "it is a cv dataset")

  # [可选]定义索引字段，只能是标量字段
  writer.add_index(["file_name", "label"])

  # 按Schema方式组织训练数据，并将其写入MindRecord文件
  # 此处使用Image.new(...)模拟图片数据，真实场景可以使用io接口读取磁盘上的图像数据
  data = []
  for i in range(100):    # 模拟数据集有100个样本
      i += 1

      sample = {}
      white_io = BytesIO()
      Image.new('RGB', (i*10, i*10), (255, 255, 255)).save(white_io, 'JPEG')  # 图片大小可以不同
      image_bytes = white_io.getvalue()
      sample['file_name'] = str(i) + ".jpg"     # 对应file_name字段
      sample['label'] = i                       # 对应label字段
      sample['data'] = white_io.getvalue()     # 对应data字段

      data.append(sample)
      if i % 10 == 0:     # 每10条样本做一次写操作
          writer.write_raw_data(data)
          data = []

  if data:                # 写入可能剩余的数据
      writer.write_raw_data(data)

  writer.commit()         # 关闭写入操作

  ################################ 读取MindRecord文件 ################################

  data_set = ds.MindDataset(dataset_file=mindrecord_filename)  # 创建读取对象，默认开启shuffle
  decode_op = vision.Decode()
  data_set = data_set.map(input_columns=["data"], operations=decode_op, num_parallel_workers=2)  # 解码data字段
  count = 0
  for item in data_set.create_dict_iterator():                 # 循环读取MindRecord中所有数据
      print("sample: {}".format(item))
      count += 1
  print("Got {} samples".format(count))
  ```

### NLP类数据集

> 因为NLP类数据一般会经过预处理转换为字典序，此预处理过程不在本示例范围，该示例只演示转换后的字典序数据如何写入MindRecord。

  ```python
  """
  示例说明：本示例主要介绍用户如何将自己的NLP类数据集转换成MindRecoed格式，并使用MindDataset读取。
  详细步骤：
  1. 创建一个包含100条记录的MindRecord文件，其样本包含八个字段，均为整形数组；
  2. 使用MindDataset读取MindRecord文件。
  """

  import os
  import numpy as np
  import mindspore.dataset as ds
  from mindspore.mindrecord import FileWriter

  ################################ 生成MindRecord文件 ################################

  mindrecord_filename = "test.mindrecord"

  # 如果存在MindRecord文件，则需要先删除
  if os.path.exists(mindrecord_filename):
      os.remove(mindrecord_filename)
      os.remove(mindrecord_filename + ".db")

  # 创建写对象，将会生成 mindrecord_filename 和 mindrecord_filename.db 两个文件
  writer = FileWriter(file_name=mindrecord_filename, shard_num=1)

  # 定义数据集Schema，此处认为文本已经转为字典序
  nlp_schema = {"source_sos_ids": {"type": "int64", "shape": [-1]},
                "source_sos_mask": {"type": "int64", "shape": [-1]},
                "source_eos_ids": {"type": "int64", "shape": [-1]},
                "source_eos_mask": {"type": "int64", "shape": [-1]},
                "target_sos_ids": {"type": "int64", "shape": [-1]},
                "target_sos_mask": {"type": "int64", "shape": [-1]},
                "target_eos_ids": {"type": "int64", "shape": [-1]},
                "target_eos_mask": {"type": "int64", "shape": [-1]}}
  writer.add_schema(nlp_schema, "it is a preprocessed nlp dataset")

  # 按Schema方式组织训练数据，并将其写入MindRecord文件
  data = []
  for i in range(100):    # 模拟数据集有100个样本
      i += 1

      # 组织训练数据
      sample = {"source_sos_ids": np.array([i, i+1, i+2, i+3, i+4], dtype=np.int64),
                "source_sos_mask": np.array([i*1, i*2, i*3, i*4, i*5, i*6, i*7], dtype=np.int64),
                "source_eos_ids": np.array([i+5, i+6, i+7, i+8, i+9, i+10], dtype=np.int64),
                "source_eos_mask": np.array([19, 20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int64),
                "target_sos_ids": np.array([28, 29, 30, 31, 32], dtype=np.int64),
                "target_sos_mask": np.array([33, 34, 35, 36, 37, 38], dtype=np.int64),
                "target_eos_ids": np.array([39, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
                "target_eos_mask": np.array([48, 49, 50, 51], dtype=np.int64)}

      data.append(sample)
      if i % 10 == 0:     # 每10条样本做一次写操作
          writer.write_raw_data(data)
          data = []

  if data:                # 写入可能剩余的数据
      writer.write_raw_data(data)

  writer.commit()         # 关闭写入操作

  ################################ 读取MindRecord文件 ################################

  data_set = ds.MindDataset(dataset_file=mindrecord_filename)  # 创建读取对象，默认开启shuffle
  count = 0
  for item in data_set.create_dict_iterator():                 # 循环读取MindRecord中所有数据
      print("sample: {}".format(item))
      count += 1
  print("Got {} samples".format(count))
  ```

## 常用数据集转换MindRecord

MindSpore提供转换常见数据集的工具类，能够将常见的经典数据集转换为MindRecord格式。常见数据集及其对应的工具类列表如下。

| 数据集 | 格式转换工具类 |
| -------- | ------------ |
| CIFAR-10 | Cifar10ToMR |
| CIFAR-100 | Cifar100ToMR |
| ImageNet | ImageNetToMR |
| MNIST | MnistToMR |
| TFRecord | TFRecordToMR |
| CSV File | CsvToMR |

更多数据集转换的详细说明可参见[API文档](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.mindrecord.html)。

### 转换CIFAR-10数据集

用户可以通过`Cifar10ToMR`类，将CIFAR-10原始数据转换为MindRecord格式。

1. 下载[CIFAR-10数据集](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)并解压，目录结构如下所示。

    ```
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

3. 创建`Cifar10ToMR`对象，调用`transform`接口，将CIFAR-10数据集转换为MindRecord格式。

    ```python
    CIFAR10_DIR = "./cifar10/cifar-10-batches-py"
    MINDRECORD_FILE = "./cifar10.mindrecord"
    cifar10_transformer = Cifar10ToMR(CIFAR10_DIR, MINDRECORD_FILE)
    cifar10_transformer.transform(['label'])
    ```

    **参数说明：**
    - `CIFAR10_DIR`：CIFAR-10数据集的文件夹路径。  
    - `MINDRECORD_FILE`：输出的MindSpore数据格式文件路径。

### 转换CIFAR-100数据集

用户可以通过`Cifar100ToMR`类，将CIFAR-100原始数据转换为MindRecord格式。

1. 准备好CIFAR-100数据集，将文件解压至指定的目录（示例中将数据集保存到`cifar100`目录），如下所示。

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

    **参数说明：**  
    - `CIFAR100_DIR`：CIFAR-100数据集的文件夹路径。  
    - `MINDRECORD_FILE`：输出的MindSpore数据格式文件路径。

### 转换ImageNet数据集

用户可以通过`ImageNetToMR`类，将ImageNet原始数据（图片、标注）转换为MindSpore数据格式。

1. 下载并按照要求准备好ImageNet数据集。

    > ImageNet数据集下载地址：<http://image-net.org/download>

    对下载后的ImageNet数据集，整理数据集组织形式为一个包含所有图片的文件夹，以及一个记录图片对应标签的映射文件。

    标签映射文件包含2列，分别为各类别图片目录、标签ID，用空格隔开，映射文件示例如下：
    ```
    n01440760 0
    n01443537 1
    n01484850 2
    n01491361 3
    n01494475 4
    n01496331 5
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

    ***参数说明：***  
    - `MNIST_DIR`：MNIST数据集的文件夹路径。  
    - `MINDRECORD_FILE`：输出的MindSpore数据格式文件路径。


### 转换CSV数据集

  ```python
  """
  示例说明：本示例首先创建一个CSV文件，然后通过MindSpore中CsvToMR工具，
            将Csv文件转换为MindRecord文件，并最终通过MindDataset将其读取出来。
  详细步骤：
  1. 创建一个包含5条记录的CSV文件；
  2. 使用CsvToMR工具将CSV转换为MindRecord；
  3. 使用MindDataset读取MindRecord文件。
  """

  import csv
  import os
  import mindspore.dataset as ds
  from mindspore.mindrecord import CsvToMR

  CSV_FILE_NAME = "test.csv"                       # 创建的CSV文件
  MINDRECORD_FILE_NAME = "test.mindrecord"         # 转换后的MindRecord文件
  PARTITION_NUM = 1

  ################################ 创建CSV文件 ################################

  # 生成CSV文件
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

  ################################ CSV 转 MindRecord文件 ################################

  # 调用CsvToMR工具，初始化
  csv_transformer = CsvToMR(CSV_FILE_NAME, MINDRECORD_FILE_NAME, partition_number=PARTITION_NUM)
  # 执行转换操作
  csv_transformer.transform()

  assert os.path.exists(MINDRECORD_FILE_NAME)
  assert os.path.exists(MINDRECORD_FILE_NAME + ".db")

  ############################### 读取MindRecord文件 ################################

  data_set = ds.MindDataset(dataset_file=MINDRECORD_FILE_NAME)  # 创建读取对象，默认开启shuffle
  count = 0
  for item in data_set.create_dict_iterator():                  # 循环读取MindRecord中所有数据
      print("sample: {}".format(item))
      count += 1
  print("Got {} samples".format(count))
  ```

### 转换TFRecord数据集

  ```python
  """
  示例说明：本示例通过TF创建一个TFRecord文件，然后通过MindSpore中TFRecordToMR工具，
            将TFRecord文件转换为MindRecord文件，并最终通过MindDataset将其读取出来。
  详细步骤：
  1. 创建一个包含10条记录，且样本格式为：
     feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                     "image_bytes": tf.io.FixedLenFeature([], tf.string),
                     "int64_scalar": tf.io.FixedLenFeature([], tf.int64),
                     "float_scalar": tf.io.FixedLenFeature([], tf.float32),
                     "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                     "float_list": tf.io.FixedLenFeature([7], tf.float32)}
     的TFRecord文件；
  2. 使用TFRecordToMR工具将TFRecord转换为MindRecord；
  3. 使用MindDataset读取MindRecord文件，并通过Decode算子对其image_bytes字段进行解码。
  """

  import collections
  from io import BytesIO
  import os
  import mindspore.dataset as ds
  from mindspore.mindrecord import TFRecordToMR
  import mindspore.dataset.transforms.vision.c_transforms as vision
  from PIL import Image
  import tensorflow as tf    # 需要tensorflow >= 2.1.0

  TFRECORD_FILE_NAME = "test.tfrecord"             # 创建的TFRecord文件
  MINDRECORD_FILE_NAME = "test.mindrecord"         # 转换后的MindRecord文件
  PARTITION_NUM = 1

  ################################ 创建TFRecord文件 ################################

  # 生成TFRecord文件
  def generate_tfrecord():
      def create_int_feature(values):
          if isinstance(values, list):
              feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))  # values: [int, int, int]
          else:
              feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))      # values: int
          return feature

      def create_float_feature(values):
          if isinstance(values, list):
              feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))  # values: [float, float]
          else:
              feature = tf.train.Feature(float_list=tf.train.FloatList(value=[values]))      # values: float
          return feature

      def create_bytes_feature(values):
          if isinstance(values, bytes):
              white_io = BytesIO()
              Image.new('RGB', (10, 10), (255, 255, 255)).save(white_io, 'JPEG')                  # 图片大小可以不同
              image_bytes = white_io.getvalue()
              feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))      # values: bytes
          else:
              # values: string
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

  ################################ TFRecord 转 MindRecord文件 ################################

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

  # 调用TFRecordToMR工具，初始化
  tfrecord_transformer = TFRecordToMR(TFRECORD_FILE_NAME, MINDRECORD_FILE_NAME, feature_dict, ["image_bytes"])
  # 执行转换操作
  tfrecord_transformer.transform()

  assert os.path.exists(MINDRECORD_FILE_NAME)
  assert os.path.exists(MINDRECORD_FILE_NAME + ".db")

  ############################### 读取MindRecord文件 ################################

  data_set = ds.MindDataset(dataset_file=MINDRECORD_FILE_NAME)  # 创建读取对象，默认开启shuffle
  decode_op = vision.Decode()
  data_set = data_set.map(input_columns=["image_bytes"], operations=decode_op, num_parallel_workers=2)  # 解码图像字段
  count = 0
  for item in data_set.create_dict_iterator():                 # 循环读取MindRecord中所有数据
      print("sample: {}".format(item))
      count += 1
  print("Got {} samples".format(count))
  ```
