# 加载数据集

`Ascend` `GPU` `CPU` `数据准备` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r0.7/tutorials/source_zh_cn/use/data_preparation/loading_the_datasets.md" target="_blank"><img src="../../_static/logo_source.png"></a>&nbsp;&nbsp;
<a href="https://gitee.com/mindspore/docs/blob/r0.7/tutorials/notebook/loading_dataset.ipynb" target="_blank"><img src="../../_static/logo_notebook.png"></a>

## 概述

MindSpore可以帮助你加载常见的数据集、特定数据格式的数据集或自定义的数据集。加载数据集时，需先导入所需要依赖的库`mindspore.dataset`。
```python
import mindspore.dataset as ds
```

## 加载常见的数据集
MindSpore可以加载常见的标准数据集。支持的数据集如下表：

| 数据集    | 简要说明                                                                                                                   |
| --------- | -------------------------------------------------------------------------------------------------------------------------- |
| ImageNet  | ImageNet是根据WordNet层次结构组织的图像数据库，其中层次结构的每个节点都由成百上千个图像表示。                              |
| MNIST     | 是一个手写数字图像的大型数据库，通常用于训练各种图像处理系统。                                                             |
| CIFAR-10  | 常用于机器学习和计算机视觉领域训练的图像集合。CIFAR-10数据集包含10种不同类别的60,000张32x32彩色图像。                      |
| CIFAR-100 | 该数据集类似于CIFAR-10，不同之处在于它有100个类别，每个类别包含600张图像：500张训练图像和100张测试图像。         |
| PASCAL-VOC | 数据内容多样，可用于训练计算机视觉模型（分类、定位、检测、分割、动作识别等）。     |
| CelebA    | CelebA人脸数据集包含上万个名人身份的人脸图片，每张图片有40个特征标记，常用于人脸相关的训练任务。    |

加载常见数据集的详细步骤如下，以创建`CIFAR-10`对象为例，用于加载支持的数据集。

1. 下载[CIFAR-10数据集](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)，并解压。这里使用的是二进制格式的数据集（CIFAR-10 binary version）。
2. 配置数据集目录，定义需要加载的数据集实例。
    ```python
    DATA_DIR = "cifar10_dataset_dir/"

    cifar10_dataset = ds.Cifar10Dataset(DATA_DIR)
    ```
3. 创建迭代器，通过迭代器读取数据。
    ```python
    for data in cifar10_dataset.create_dict_iterator():
    # In CIFAR-10 dataset, each dictionary of data has keys "image" and "label".
        print(data["image"])
        print(data["label"]) 
    ```

## 加载特定数据格式的数据集

### MindSpore数据格式
MindSpore天然支持读取MindSpore数据格式——`MindRecord`存储的数据集，在性能和特性上有更好的支持。  
> 阅读[将数据集转换为MindSpore数据格式](converting_datasets.md)章节，了解如何将数据集转化为MindSpore数据格式。

可以通过`MindDataset`对象对数据集进行读取。详细方法如下所示：

1. 创建`MindDataset`，用于读取数据。
    ```python
    import os
    CV_FILE_NAME = os.path.join(MODULE_PATH, "./imagenet.mindrecord")
    data_set = ds.MindDataset(dataset_file=CV_FILE_NAME)
    ```
    其中，  
    `dataset_file`：指定MindRecord的文件或文件列表。

2. 创建字典迭代器，通过迭代器读取数据记录。
    ```python
    num_iter = 0
    for data in data_set.create_dict_iterator():
        print(data["label"])
        num_iter += 1
    ```

### `Manifest`数据格式
`Manifest`是华为ModelArts支持的数据格式文件，详细说明请参见：<https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0009.html>。

Mindspore对Manifest格式的数据集提供了对应的数据集类。如下所示，配置数据集目录，定义需要加载的数据集实例。
```python
DATA_DIR = "manifest_dataset_path"

manifest_dataset = ds.ManifestDataset(DATA_DIR)
```
目前ManifestDataset仅支持加载图片、标签类型的数据集，默认列名为"image"和"label"。

### `TFRecord`数据格式
MindSpore也支持读取`TFRecord`数据格式的数据集，可以通过`TFRecordDataset`对象进行数据集读取。

1. 只需传入数据集路径或.tfrecord文件列表，即可创建`TFRecordDataset`。
    ```python
    DATA_DIR = ["tfrecord_dataset_path/train-0000-of-0001.tfrecord"]

    dataset = ds.TFRecordDataset(DATA_DIR)
    ```
    
2. 用户可以通过创建Schema文件或Schema类，设定数据集格式及特征。

    Schema文件示例如下所示：

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
    其中，  
    `datasetType`: 数据格式的类型，这里“TF”是指TFrecord数据格式。  
    `columns`：列信息字段，需要根据数据集的实际列名定义，上面Schema文件示例中，数据集列为image和label两列。   
    `numRows`：行数信息字段，控制加载数据的最大行数。如果定义的行数大于实际行数，加载时则以实际行数为准。
    
    在创建TFRecordDataset时将Schema文件路径传入，使用样例如下：
    ```python
    DATA_DIR = ["tfrecord_dataset_path/train-0000-of-0001.tfrecord"]
    SCHEMA_DIR = "dataset_schema_path/schema.json"

    dataset = ds.TFRecordDataset(DATA_DIR, schema=SCHEMA_DIR)
    ```
    
    创建Schema类使用样例如下：
    ```python
    import mindspore.common.dtype as mstype
    schema = ds.Schema()
    schema.add_column('image', de_type=mstype.uint8)  # Binary data usually use uint8 here.
    schema.add_column('label', de_type=mstype.int32)

    dataset = ds.TFRecordDataset(DATA_DIR, schema=schema)
    ```

3. 创建字典迭代器，通过迭代器读取数据。
    ```python
    for data in dataset.create_dict_iterator():
    # The dictionary of data has keys "image" and "label" which are consistent with columns names in its schema.
        print(data["image"])
        print(data["label"]) 
    ```

## 加载自定义数据集
现实场景中，数据集的种类多种多样，对于自定义数据集或者目前不支持直接加载的数据集，有两种方法可以处理。
一种方法是将数据集转成MindRecord格式（请参考[将数据集转换为MindSpore数据格式](https://www.mindspore.cn/tutorial/zh-CN/r0.7/use/data_preparation/converting_datasets.html)章节），另一种方法是通过`GeneratorDataset`对象加载，以下将展示如何使用`GeneratorDataset`。

1. 定义一个可迭代的对象，用于生成数据集。以下展示了两种示例，一种是含有`yield`返回值的自定义函数，另一种是含有`__getitem__`的自定义类。两种示例都将产生一个含有从0到9数字的数据集。
   > 自定义的可迭代对象，每次返回`numpy array`的元组，作为一行数据。 

   自定义函数示例如下：
   ```python
   import numpy as np  # Import numpy lib.
   def generator_func(num):
       for i in range(num):
           yield (np.array([i]),)  # Notice, tuple of only one element needs following a comma at the end.
   ```
   
   自定义类示例如下：
   ```python
   import numpy as np  # Import numpy lib.
   class Generator():
      
       def __init__(self, num):
           self.num = num
      
       def __getitem__(self, item):
           return (np.array([item]),)  # Notice, tuple of only one element needs following a comma at the end.
      
       def __len__(self):
           return self.num
   ```
2. 使用`GeneratorDataset`创建数据集。将`generator_func`传入`GeneratorDataset`创建数据集`dataset1`，并设定`column`名为“data” 。
   将定义的`Generator`对象传入`GeneratorDataset`创建数据集`dataset2`，并设定`column`名为“data” 。
   ```python
   dataset1 = ds.GeneratorDataset(source=generator_func(10), column_names=["data"], shuffle=False)
   dataset2 = ds.GeneratorDataset(source=Generator(10), column_names=["data"], shuffle=False)
   ```

3. 在创建数据集后，可以通过给数据创建迭代器的方式，获取相应的数据。有两种创建迭代器的方法。
   - 创建返回值为序列类型的迭代器。以下分别对`dataset1`和`dataset2`创建迭代器，并打印输出数据观察结果。
      ```python
     print("dataset1:") 
     for data in dataset1.create_tuple_iterator():  # each data is a sequence
          print(data)
     
     print("dataset2:")
     for data in dataset2.create_tuple_iterator():  # each data is a sequence
         print(data)
      ```
     输出如下所示：
     ```
     dataset1:
     [array([0], dtype=int64)]
     [array([1], dtype=int64)]
     [array([2], dtype=int64)]
     [array([3], dtype=int64)]
     [array([4], dtype=int64)]
     [array([5], dtype=int64)]
     [array([6], dtype=int64)]
     [array([7], dtype=int64)]
     [array([8], dtype=int64)]
     [array([9], dtype=int64)]
     dataset2:
     [array([0], dtype=int64)]
     [array([1], dtype=int64)]
     [array([2], dtype=int64)]
     [array([3], dtype=int64)]
     [array([4], dtype=int64)]
     [array([5], dtype=int64)]
     [array([6], dtype=int64)]
     [array([7], dtype=int64)]
     [array([8], dtype=int64)]
     [array([9], dtype=int64)]
     ```

   - 创建返回值为字典类型的迭代器。以下分别对`dataset1`和`dataset2`创建迭代器，并打印输出数据观察结果。
      ```python
      print("dataset1:")
      for data in dataset1.create_dict_iterator():  # each data is a dictionary
          print(data["data"])
     
      print("dataset2:")
      for data in dataset2.create_dict_iterator():  # each data is a dictionary
          print(data["data"])
      ```
     输出如下所示：
     ```
     dataset1:
     {'data': array([0], dtype=int64)}
     {'data': array([1], dtype=int64)}
     {'data': array([2], dtype=int64)}
     {'data': array([3], dtype=int64)}
     {'data': array([4], dtype=int64)}
     {'data': array([5], dtype=int64)}
     {'data': array([6], dtype=int64)}
     {'data': array([7], dtype=int64)}
     {'data': array([8], dtype=int64)}
     {'data': array([9], dtype=int64)}
     dataset2:
     {'data': array([0], dtype=int64)}
     {'data': array([1], dtype=int64)}
     {'data': array([2], dtype=int64)}
     {'data': array([3], dtype=int64)}
     {'data': array([4], dtype=int64)}
     {'data': array([5], dtype=int64)}
     {'data': array([6], dtype=int64)}
     {'data': array([7], dtype=int64)}
     {'data': array([8], dtype=int64)}
     {'data': array([9], dtype=int64)}
     ```
