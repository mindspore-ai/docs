# Converting Datasets to the Mindspore Data Format

<a href="https://gitee.com/mindspore/docs/blob/r0.5/tutorials/source_en/use/data_preparation/converting_datasets.md" target="_blank"><img src="../../_static/logo_source.png"></a>

## Overview

You can convert non-standard datasets and common datasets into the MindSpore data format so that they can be easily loaded to MindSpore for training. In addition, the performance of MindSpore in some scenarios is optimized, which delivers better user experience when you use datasets in the MindSpore data format.   
The MindSpore data format has the following features:  
1. Unified storage and access of user data are implemented, simplifying training data reading.
2. Data is aggregated for storage, efficient reading, and easy management and transfer.
3. Data encoding and decoding are efficient and transparent to users.
4. The partition size is flexibly controlled to implement distributed training.

## Converting Non-Standard Datasets to the Mindspore Data Format

MindSpore provides write operation tools to write user-defined raw data in MindSpore format.

### Converting Images and Labels

1. Import the `FileWriter` class for file writing.

    ```python
    from mindspore.mindrecord import FileWriter
    ```

2. Define a dataset schema which defines dataset fields and field types.

    ```python
    cv_schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
    ```
    Schema specifications are as follows:  
    A field name can contain only letters, digits, and underscores (_). 
    The field type can be int32, int64, float32, float64, string, or bytes.  
    The field shape can be a one-dimensional array represented by [-1], a two-dimensional array represented by [m, n], or a three-dimensional array represented by [x, y, z].  
    > 1. The type of a field with the shape attribute can only be int32, int64, float32, or float64.  
    > 2. If the field has the shape attribute, prepare the data of `numpy.ndarray` type and transfer the data to the `write_raw_data` API.  
    
    Examples:  
    - Image classification
        ```python
        cv_schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
        ```
    - Natural Language Processing (NLP)
        ```python
        cv_schema_json = {"id": {"type": "int32"}, "masks": {"type": "int32", "shape": [-1]}, "inputs": {"type": "int64", "shape": [4, 32]}, "labels": {"type": "int64", "shape": [-1]}}
        ```

3. Prepare the data sample list to be written based on the user-defined schema format.

    ```python
    data = [{"file_name": "1.jpg", "label": 0, "data": b"\x10c\xb3w\xa8\xee$o&<q\x8c\x8e(\xa2\x90\x90\x96\xbc\xb1\x1e\xd4QER\x13?\xff\xd9"},
            {"file_name": "2.jpg", "label": 56, "data": b"\xe6\xda\xd1\xae\x07\xb8>\xd4\x00\xf8\x129\x15\xd9\xf2q\xc0\xa2\x91YFUO\x1dsE1\x1ep"},
            {"file_name": "3.jpg", "label": 99, "data": b"\xaf\xafU<\xb8|6\xbd}\xc1\x99[\xeaj+\x8f\x84\xd3\xcc\xa0,i\xbb\xb9-\xcdz\xecp{T\xb1\xdb\"}]
    ```
    
4. Prepare index fields. Adding index fields can accelerate data reading. This step is optional.

    ```python
    indexes = ["file_name", "label"]
    ```

5. Create a `FileWriter` object, transfer the file name and number of slices, add the schema and index, call the `write_raw_data` API to write data, and call the `commit` API to generate a local data file.

    ```python    
    writer = FileWriter(file_name="testWriter.mindrecord", shard_num=4)
    writer.add_schema(cv_schema_json, "test_schema")
    writer.add_index(indexes)
    writer.write_raw_data(data)
    writer.commit()
    ```
    In the preceding information:  
    `write_raw_data`: writes data to the memory.  
    `commit`: writes the data in the memory to the disk.

6. Add data to the existing data format file, call the `open_for_append` API to open the existing data file, call the `write_raw_data` API to write new data, and then call the `commit` API to generate a local data file.
    ```python
    writer = FileWriter.open_for_append("testWriter.mindrecord0")
    writer.write_raw_data(data)
    writer.commit()
    ```

## Converting Common Datasets to the MindSpore Data Format

MindSpore provides utility classes to convert common datasets to the MindSpore data format. The following table lists common datasets and called utility classes:  

| Dataset  | Called Utility Class |
| -------- | ------------ |
| CIFAR-10 | Cifar10ToMR  |
| CIFAR-100| Cifar100ToMR |
| ImageNet | ImageNetToMR |
| MNIST    | MnistToMR    |


### Converting the CIFAR-10 Dataset
You can use the `Cifar10ToMR` class to convert the raw CIFAR-10 data into the MindSpore data format.

1. Prepare the CIFAR-10 python version dataset and decompress the file to a specified directory (the `cifar10` directory in the example), as the following shows:  
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
    > CIFAR-10 dataset download address: <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>    

2. Import the `Cifar10ToMR` class for dataset converting.

    ```python
    from mindspore.mindrecord import Cifar10ToMR
    ```
3. Instantiate the `Cifar10ToMR` object and call the `transform` API to convert the CIFAR-10 dataset to the MindSpore data format.

    ```python
    CIFAR10_DIR = "./cifar10/cifar-10-batches-py"
    MINDRECORD_FILE = "./cifar10.mindrecord"
    cifar10_transformer = Cifar10ToMR(CIFAR10_DIR, MINDRECORD_FILE)
    cifar10_transformer.transform(['label'])
    ```
    In the preceding information:  
    `CIFAR10_DIR`: path where the CIFAR-10 dataset folder is stored.  
    `MINDRECORD_FILE`: path where the output file in the MindSpore data format is stored.

### Converting the CIFAR-100 Dataset
You can use the `Cifar100ToMR` class to convert the raw CIFAR-100 data to the MindSpore data format.

1. Prepare the CIFAR-100 dataset and decompress the file to a specified directory (the `cifar100` directory in the example).
    ```
    % ll cifar100/cifar-100-python/
    meta
    test
    train
    ```
    > CIFAR-100 dataset download address: <https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz>

2. Import the `Cifar100ToMR` class for dataset converting.

    ```python
    from mindspore.mindrecord import Cifar100ToMR
    ```
3. Instantiate the `Cifar100ToMR` object and call the `transform` API to convert the CIFAR-100 dataset to the MindSpore data format.

    ```python
    CIFAR100_DIR = "./cifar100/cifar-100-python"
    MINDRECORD_FILE = "./cifar100.mindrecord"
    cifar100_transformer = Cifar100ToMR(CIFAR100_DIR, MINDRECORD_FILE)
    cifar100_transformer.transform(['fine_label', 'coarse_label'])
    ```
    In the preceding information:  
    `CIFAR100_DIR`: path where the CIFAR-100 dataset folder is stored.  
    `MINDRECORD_FILE`: path where the output file in the MindSpore data format is stored.

### Converting the ImageNet Dataset

You can use the `ImageNetToMR` class to convert the raw ImageNet data (images and labels) to the MindSpore data format.

1. Download and prepare the ImageNet dataset as required.

    > ImageNet dataset download address: <http://image-net.org/download>

    Store the downloaded ImageNet dataset in a folder. The folder contains all images and a mapping file that records labels of the images.

    In the mapping file, there are two columns, which are separated by spaces. They indicate image classes and label IDs. The following is an example of the mapping file:
    ```
    n01440760 0
    n01443537 1
    n01484850 2
    n01491361 3
    n01494475 4
    n01496331 5
    ```

2. Import the `ImageNetToMR` class for dataset converting.

    ```python
    from mindspore.mindrecord import ImageNetToMR
    ```
    
3. Instantiate the `ImageNetToMR` object and call the `transform` API to convert the dataset to the MindSpore data format.
    ```python
    IMAGENET_MAP_FILE = "./testImageNetDataWhole/labels_map.txt"
    IMAGENET_IMAGE_DIR = "./testImageNetDataWhole/images"
    MINDRECORD_FILE = "./testImageNetDataWhole/imagenet.mindrecord"
    PARTITION_NUMBER = 4
    imagenet_transformer = ImageNetToMR(IMAGENET_MAP_FILE, IMAGENET_IMAGE_DIR, MINDRECORD_FILE, PARTITION_NUMBER)
    imagenet_transformer.transform()
    ```
    In the preceding information:  
    `IMAGENET_MAP_FILE`: path where the label mapping file of the ImageNetToMR dataset is stored.  
    `IMAGENET_IMAGE_DIR`: path where all ImageNet images are stored.  
    `MINDRECORD_FILE`: path where the output file in the MindSpore data format is stored.

### Converting the MNIST Dataset
You can use the `MnistToMR` class to convert the raw MNIST data to the MindSpore data format.

1. Prepare the MNIST dataset and save the downloaded file to a specified directory, as the following shows:
    ```
    % ll mnist_data/
    train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz
    t10k-images-idx3-ubyte.gz
    t10k-labels-idx1-ubyte.gz
    ```
    > MNIST dataset download address: <http://yann.lecun.com/exdb/mnist>

2. Import the `MnistToMR` class for dataset converting.

    ```python
    from mindspore.mindrecord import MnistToMR
    ```
3. Instantiate the `MnistToMR` object and call the `transform` API to convert the MNIST dataset to the MindSpore data format.

    ```python
    MNIST_DIR = "./mnist_data"
    MINDRECORD_FILE = "./mnist.mindrecord"
    mnist_transformer = MnistToMR(MNIST_DIR, MINDRECORD_FILE)
    mnist_transformer.transform()
    ```
    In the preceding information:  
    `MNIST_DIR`: path where the MNIST dataset folder is stored.  
    `MINDRECORD_FILE`: path where the output file in the MindSpore data format is stored.
