# Loading the Dataset

`Linux` `Ascend` `GPU` `CPU` `Data Preparation` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Loading the Dataset](#loading-the-dataset)
    - [Overview](#overview)
    - [Loading Common Datasets](#loading-common-datasets)
    - [Loading Datasets of a Specific Data Format](#loading-datasets-of-a-specific-data-format)
        - [MindSpore Data Format](#mindspore-data-format)
        - [`Manifest` Data Format](#manifest-data-format)
        - [`TFRecord` Data Format](#tfrecord-data-format)
    - [Loading a Custom Dataset](#loading-a-custom-dataset)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/use/data_preparation/loading_the_datasets.md" target="_blank"><img src="../../_static/logo_source.png"></a>

## Overview

MindSpore helps you load common datasets, datasets of specific data formats, or custom datasets. Before loading a dataset, you need to import the required library `mindspore.dataset`.
```python
import mindspore.dataset as ds
```

## Loading Common Datasets
MindSpore can load common standard datasets. The following table lists the supported datasets:

| Dataset    | Description                                                                                                                   |
| --------- | -------------------------------------------------------------------------------------------------------------------------- |
| ImageNet  | An image database organized based on the WordNet hierarchical structure. Each node in the hierarchical structure is represented by hundreds of images.                              |
| MNIST     | A large database of handwritten digit images, which is usually used to train various image processing systems.                                                             |
| CIFAR-10  | A collection of images that are commonly used to train machine learning and computer vision algorithms. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes.                      |
| CIFAR-100 | The dataset is similar to CIFAR-10. The difference is that this dataset has 100 classes, and each class contains 600 images, including 500 training images and 100 test images.         |
| PASCAL-VOC | The data content is diversified and can be used to train computer vision models (such as classification, positioning, detection, segmentation, and action recognition).     |
| CelebA    | CelebA face dataset contains tens of thousands of face images of celebrities with 40 attribute annotations, which are usually used for face-related training tasks.    |

The procedure for loading common datasets is as follows. The following describes how to create the `CIFAR-10` object to load supported datasets.

1. Download and decompress the [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz). The dataset in binary format (CIFAR-10 binary version) is used.
2. Configure the dataset directory and define the dataset instance to be loaded.
    ```python
    DATA_DIR = "cifar10_dataset_dir/"

    cifar10_dataset = ds.Cifar10Dataset(DATA_DIR)
    ```
3. Create an iterator and read data through the iterator.
    ```python
    for data in cifar10_dataset.create_dict_iterator():
    # In CIFAR-10 dataset, each dictionary of data has keys "image" and "label".
        print(data["image"])
        print(data["label"]) 
    ```

## Loading Datasets of a Specific Data Format

### MindSpore Data Format 
MindSpore supports reading datasets stored in MindSpore data format, such as reading datasets stored in `MindRecord`. By this nature, MindSpore may have better performance and characteristics. 
> For details about how to convert datasets to the MindSpore data format, see [Converting the Dataset to MindSpore Data Format](converting_datasets.md).

To read a dataset using the `MindDataset` object, perform the following steps:

1. Create `MindDataset` for reading data.
    ```python
    import os
    CV_FILE_NAME = os.path.join(MODULE_PATH, "./imagenet.mindrecord")
    data_set = ds.MindDataset(dataset_file=CV_FILE_NAME)
    ```
    In the preceding information:  
    `dataset_file`: specifies the MindRecord file or the list of MindRecord files.

2. Create a dictionary iterator and read data records through the iterator.
    ```python
    num_iter = 0
    for data in data_set.create_dict_iterator():
        print(data["label"])
        num_iter += 1
    ```

### `Manifest` Data Format
`Manifest` is a data format file supported by Huawei ModelArts. For details, see <https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0009.html>.

MindSpore provides dataset classes for datasets in `Manifest` format. Run the following commands to configure the dataset directory and define the dataset instance to be loaded:
```python
DATA_DIR = "manifest_dataset_path"

manifest_dataset = ds.ManifestDataset(DATA_DIR)
```
Currently, ManifestDataset supports only datasets of images and labels. The default column names are 'image' and 'label'.

### `TFRecord` Data Format
MindSpore can also read datasets in the `TFRecord` data format through the `TFRecordDataset` object.

1. Input the dataset path or the .tfrecord file list to create the `TFRecordDataset`.
    ```python
    DATA_DIR = ["tfrecord_dataset_path/train-0000-of-0001.tfrecord"]

    dataset = ds.TFRecordDataset(DATA_DIR)
    ```
    
2. Create schema files or schema classes to set the dataset format and features.

    The following is an example of the schema file:

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
    In the preceding information:  
    `datasetType`: data format. TF indicates the TFRecord data format.  
    `columns`: column information field, which is defined based on the actual column names of the dataset. In the preceding schema file example, the dataset columns are 'image' and 'label'.   
    `numRows`: row information field, which controls the maximum number of rows for loading data. If the number of defined rows is greater than the actual number of rows, the actual number of rows prevails during loading.
    
    When creating the TFRecordDataset, input the schema file path. An example is as follows:
    ```python
    DATA_DIR = ["tfrecord_dataset_path/train-0000-of-0001.tfrecord"]
    SCHEMA_DIR = "dataset_schema_path/schema.json"

    dataset = ds.TFRecordDataset(DATA_DIR, schema=SCHEMA_DIR)
    ```
    
    An example of creating a schema class is as follows:
    ```python
    import mindspore.common.dtype as mstype
    schema = ds.Schema()
    schema.add_column('image', de_type=mstype.uint8)  # Binary data usually use uint8 here.
    schema.add_column('label', de_type=mstype.int32)

    dataset = ds.TFRecordDataset(DATA_DIR, schema=schema)
    ```

3. Create a dictionary iterator and read data through the iterator.
    ```python
    for data in dataset.create_dict_iterator():
    # The dictionary of data has keys "image" and "label" which are consistent with columns names in its schema.
        print(data["image"])
        print(data["label"]) 
    ```

## Loading a Custom Dataset
In real scenarios, there are various datasets. For a custom dataset or a dataset that cannot be loaded by APIs directly, there are two ways.
One is to convert the dataset to MindSpore data format (for details, see [Converting Datasets to the Mindspore Data Format](https://www.mindspore.cn/tutorial/en/master/use/data_preparation/converting_datasets.html)). The other one is to use the `GeneratorDataset` object.
The following shows how to use `GeneratorDataset`.

1. Define an iterable object to generate a dataset. There are two examples following. One is a customized function which contains `yield`. The other one is a customized class which contains `__getitem__`.
   Both of them will generate a dataset with numbers from 0 to 9.
   > The custom iterable object returns a tuple of `numpy arrays` as a row of data each time. 

   An example of a custom function is as follows:
   ```python
   import numpy as np  # Import numpy lib.
   def generator_func(num):
       for i in range(num):
           yield (np.array([i]),)  # Notice, tuple of only one element needs following a comma at the end.
   ```
   An example of a custom class is as follows:   
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
   
2. Create a dataset with `GeneratorDataset`. Transfer `generator_func` to `GeneratorDataset` to create a dataset and set `column` to `data`. 
Define a `Generator` and transfer it to `GeneratorDataset` to create a dataset and set `column` to `data`.  
   ```python
   dataset1 = ds.GeneratorDataset(source=generator_func(10), column_names=["data"], shuffle=False)
   dataset2 = ds.GeneratorDataset(source=Generator(10), column_names=["data"], shuffle=False)
   ```

3. After creating a dataset, create an iterator for the dataset to obtain the corresponding data. Iterator creation methods are as follows:
   - Create an iterator whose return value is a sequence type. As shown in the following, create the iterators for `dataset1` and `dataset2`, and print the output.
      ```python
      print("dataset1:") 
      for data in dataset1.create_tuple_iterator():  # each data is a sequence
          print(data[0])
     
      print("dataset2:")
      for data in dataset2.create_tuple_iterator():  # each data is a sequence
          print(data[0])
      ```
     The output is as follows:
      ```
      dataset1:
      [Tensor(shape=[1], dtype=Int64, value= [0])]
      [Tensor(shape=[1], dtype=Int64, value= [1])]
      [Tensor(shape=[1], dtype=Int64, value= [2])]
      [Tensor(shape=[1], dtype=Int64, value= [3])]
      [Tensor(shape=[1], dtype=Int64, value= [4])]
      [Tensor(shape=[1], dtype=Int64, value= [5])]
      [Tensor(shape=[1], dtype=Int64, value= [6])]
      [Tensor(shape=[1], dtype=Int64, value= [7])]
      [Tensor(shape=[1], dtype=Int64, value= [8])]
      [Tensor(shape=[1], dtype=Int64, value= [9])]
      dataset2:
      [Tensor(shape=[1], dtype=Int64, value= [0])]
      [Tensor(shape=[1], dtype=Int64, value= [1])]
      [Tensor(shape=[1], dtype=Int64, value= [2])]
      [Tensor(shape=[1], dtype=Int64, value= [3])]
      [Tensor(shape=[1], dtype=Int64, value= [4])]
      [Tensor(shape=[1], dtype=Int64, value= [5])]
      [Tensor(shape=[1], dtype=Int64, value= [6])]
      [Tensor(shape=[1], dtype=Int64, value= [7])]
      [Tensor(shape=[1], dtype=Int64, value= [8])]
      [Tensor(shape=[1], dtype=Int64, value= [9])]
      ```

   - Create an iterator whose return value is a dictionary type. As shown in the following, create the iterators for `dataset1` and `dataset2`, and print the output.
      ```python
      print("dataset1:") 
      for data in dataset1.create_dict_iterator():  # each data is a dictionary
          print(data["data"])
     
      print("dataset2:")
      for data in dataset2.create_dict_iterator():  # each data is a dictionary
          print(data["data"])
      ```
     The output is as follows:
     ```
     dataset1:
     {'data': Tensor(shape=[], dtype=Int64, value= [0])}
     {'data': Tensor(shape=[], dtype=Int64, value= [1])}
     {'data': Tensor(shape=[], dtype=Int64, value= [2])}
     {'data': Tensor(shape=[], dtype=Int64, value= [3])}
     {'data': Tensor(shape=[], dtype=Int64, value= [4])}
     {'data': Tensor(shape=[], dtype=Int64, value= [5])}
     {'data': Tensor(shape=[], dtype=Int64, value= [6])}
     {'data': Tensor(shape=[], dtype=Int64, value= [7])}
     {'data': Tensor(shape=[], dtype=Int64, value= [8])}
     {'data': Tensor(shape=[], dtype=Int64, value= [9])}
     dataset2:
     {'data': Tensor(shape=[], dtype=Int64, value= [0])}
     {'data': Tensor(shape=[], dtype=Int64, value= [1])}
     {'data': Tensor(shape=[], dtype=Int64, value= [2])}
     {'data': Tensor(shape=[], dtype=Int64, value= [3])}
     {'data': Tensor(shape=[], dtype=Int64, value= [4])}
     {'data': Tensor(shape=[], dtype=Int64, value= [5])}
     {'data': Tensor(shape=[], dtype=Int64, value= [6])}
     {'data': Tensor(shape=[], dtype=Int64, value= [7])}
     {'data': Tensor(shape=[], dtype=Int64, value= [8])}
     {'data': Tensor(shape=[], dtype=Int64, value= [9])}
     ```