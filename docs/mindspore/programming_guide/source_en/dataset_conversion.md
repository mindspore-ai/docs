# MindSpore Data Format Conversion

`Ascend` `GPU` `CPU` `Data Preparation`

<!-- TOC -->

- [MindSpore Data Format Conversion](#mindspore-data-format-conversion)
    - [Overview](#overview)
    - [Converting Non-Standard Dataset to MindRecord](#converting-non-standard-dataset-to-mindrecord)
        - [Converting CV Dataset](#converting-cv-dataset)
        - [Converting NLP Dataset](#converting-nlp-dataset)
    - [Converting Common Dataset to MindRecord](#converting-common-dataset-to-mindrecord)
        - [Converting the CIFAR-10 Dataset](#converting-the-cifar-10-dataset)
        - [Converting the ImageNet Dataset](#converting-the-imagenet-dataset)
        - [Converting CSV Dataset](#converting-csv-dataset)
        - [Converting TFRecord Dataset](#converting-tfrecord-dataset)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_en/dataset_conversion.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## Overview

You can convert non-standard datasets and common datasets into the MindSpore data format (that is, MindRecord) to easily load the datasets to MindSpore for training. In addition, the performance of MindSpore is optimized in some scenarios. You can use MindRecord to achieve better performance.

## Converting Non-Standard Dataset to MindRecord

The following describes how to convert CV and NLP data into MindRecord and read the MindRecord file through `MindDataset`.

### Converting CV Dataset

This example describes how to convert a CV dataset into MindRecord and use `MindDataset` to load the dataset.

Create a MindRecord file containing 100 records, whose samples include the `file_name` (string), `label` (integer), and `data` (binary) fields. Use `MindDataset` to read the MindRecord file.

1. Import related modules.

    ```python
    from io import BytesIO
    import os
    import mindspore.dataset as ds
    from mindspore.mindrecord import FileWriter
    import mindspore.dataset.vision.c_transforms as vision
    from PIL import Image
    ```

2. Generate 100 images and convert them to MindRecord.

    ```python
    MINDRECORD_FILE = "test.mindrecord"

    if os.path.exists(MINDRECORD_FILE):
        os.remove(MINDRECORD_FILE)
        os.remove(MINDRECORD_FILE + ".db")

    writer = FileWriter(file_name=MINDRECORD_FILE, shard_num=1)

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
    ```

    ```text
    MSRStatus.SUCCESS
    ```

    **Parameter description:**
    - `MINDRECORD_FILE`: path of the output MindRecord file.

3. Read MindRecord using `MindDataset`.

    ```python
    data_set = ds.MindDataset(dataset_file=MINDRECORD_FILE)
    decode_op = vision.Decode()
    data_set = data_set.map(operations=decode_op, input_columns=["data"], num_parallel_workers=2)
    count = 0
    for item in data_set.create_dict_iterator(output_numpy=True):
        count += 1
    print("Got {} samples".format(count))
    ```

    ```text
    Got 100 samples
    ```

### Converting NLP Dataset

This example describes how to convert an NLP dataset into MindRecord and use `MindDataset` to load the dataset. The process of converting the text into the lexicographic order is omitted in this example.

Create a MindRecord file containing 100 records, whose samples include eight fields of the integer type. Use `MindDataset` to read the MindRecord file.

1. Import related modules.

    ```python
    import os
    import numpy as np
    import mindspore.dataset as ds
    from mindspore.mindrecord import FileWriter
    ```

2. Generate 100 text samples and convert them to MindRecord.

    ```python
    MINDRECORD_FILE = "test.mindrecord"

    if os.path.exists(MINDRECORD_FILE):
        os.remove(MINDRECORD_FILE)
        os.remove(MINDRECORD_FILE + ".db")

    writer = FileWriter(file_name=MINDRECORD_FILE, shard_num=1)

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

        sample = {"source_sos_ids": np.array([i, i + 1, i + 2, i + 3, i + 4], dtype=np.int64),
                "source_sos_mask": np.array([i * 1, i * 2, i * 3, i * 4, i * 5, i * 6, i * 7], dtype=np.int64),
                "source_eos_ids": np.array([i + 5, i + 6, i + 7, i + 8, i + 9, i + 10], dtype=np.int64),
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
    ```

    ```text
    MSRStatus.SUCCESS
    ```

    **Parameter description:**
    - `MINDRECORD_FILE`: path of the output MindRecord file.

3. Read MindRecord using `MindDataset`.

    ```python
    data_set = ds.MindDataset(dataset_file=MINDRECORD_FILE)
    count = 0
    for item in data_set.create_dict_iterator():
        count += 1
    print("Got {} samples".format(count))
    ```

    ```text
    Got 100 samples
    ```

## Converting Common Dataset to MindRecord

MindSpore provides tool classes for converting common datasets to MindRecord. The following table lists part of common datasets and their corresponding tool classes.

| Dataset | Tool Class |
| -------- | ------------ |
| CIFAR-10 | Cifar10ToMR |
| ImageNet | ImageNetToMR |
| TFRecord | TFRecordToMR |
| CSV File | CsvToMR |

For details about dataset conversion, see [MindSpore API](https://www.mindspore.cn/docs/api/en/r1.5/api_python/mindspore.mindrecord.html).

### Converting the CIFAR-10 Dataset

You can use the `Cifar10ToMR` class to convert the original CIFAR-10 data to MindRecord and use `MindDataset` to load the data.

1. Download and decompress the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz). Execute the following command in jupyter notebook:

    ```bash
    wget -N https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-python.tar.gz --no-check-certificate
    mkdir -p datasets
    tar -xzf cifar-10-python.tar.gz -C datasets
    tree ./datasets/cifar-10-batches-py
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

2. Import related modules.

    ```python
    import os
    import mindspore.dataset as ds
    import mindspore.dataset.vision.c_transforms as vision
    from mindspore.mindrecord import Cifar10ToMR
    ```

3. Create the `Cifar10ToMR` object and call the `transform` API to convert the CIFAR-10 dataset to MindRecord.

    ```python
    ds_target_path = "./datasets/mindspore_dataset_conversion/"
    # clean old run files
    os.system("rm -f {}*".format(ds_target_path))
    os.system("mkdir -p {}".format(ds_target_path))

    CIFAR10_DIR = "./datasets/cifar-10-batches-py"
    MINDRECORD_FILE = "./datasets/mindspore_dataset_conversion/cifar10.mindrecord"
    cifar10_transformer = Cifar10ToMR(CIFAR10_DIR, MINDRECORD_FILE)
    cifar10_transformer.transform(['label'])
    ```

    ```text
    MSRStatus.SUCCESS
    ```

     **Parameter description:**
    - `CIFAR10_DIR`: path of the CIFAR-10 dataset folder.
    - `MINDRECORD_FILE`: path of the output MindRecord file.

4. Read MindRecord using `MindDataset`.

    ```python
    data_set = ds.MindDataset(dataset_file=MINDRECORD_FILE)
    decode_op = vision.Decode()
    data_set = data_set.map(operations=decode_op, input_columns=["data"], num_parallel_workers=2)
    count = 0
    for item in data_set.create_dict_iterator(output_numpy=True):
        count += 1
    print("Got {} samples".format(count))
    ```

    ```text
    Got 50000 samples
    ```

### Converting the ImageNet Dataset

You can use the `ImageNetToMR` class to convert the original ImageNet data (images and annotations) to MindRecord and use `MindDataset` to load the data.

1. Download the [ImageNet dataset](http://image-net.org/download), save all images in the `images/` folder, and use a mapping file `labels_map.txt` to record the mapping between images and labels. The mapping file contains two columns, which are the directory and label ID of each type of images. The two columns are separated by spaces. The following is an example of the mapping file:

    ```text
    n01440760 0
    n01443537 1
    n01484850 2
    n01491361 3
    n01494475 4
    n01496331 5
    ```

    The file directory structure is as follows:

    ```text
    ├─ labels_map.txt
    └─ images
        └─ ......
    ```

2. Import the dataset conversion tool class `ImageNetToMR`.

    ```python
    import mindspore.dataset as ds
    import mindspore.dataset.vision.c_transforms as vision
    from mindspore.mindrecord import ImageNetToMR
    ```

3. Create the `ImageNetToMR` object and call the `transform` API to convert the dataset to MindRecord.

    ```python
    IMAGENET_MAP_FILE = "./labels_map.txt"
    IMAGENET_IMAGE_DIR = "./images"
    MINDRECORD_FILE = "./imagenet.mindrecord"
    imagenet_transformer = ImageNetToMR(IMAGENET_MAP_FILE, IMAGENET_IMAGE_DIR, MINDRECORD_FILE, partition_number=1)
    imagenet_transformer.transform()
    ```

    **Parameter description:**
    - `IMAGENET_MAP_FILE`: path of the label mapping file of the ImageNet dataset.
    - `IMAGENET_IMAGE_DIR`: path of the folder where all ImageNet images are stored.
    - `MINDRECORD_FILE`: path of the output MindRecord file.

4. Read MindRecord using `MindDataset`.

    ```python
    data_set = ds.MindDataset(dataset_file=MINDRECORD_FILE)
    decode_op = vision.Decode()
    data_set = data_set.map(operations=decode_op, input_columns=["image"], num_parallel_workers=2)
    count = 0
    for item in data_set.create_dict_iterator(output_numpy=True):
        count += 1
    print("Got {} samples".format(count))
    ```

### Converting CSV Dataset

Create a CSV file containing 5 records, convert the CSV file to MindRecord using the `CsvToMR` tool class, and then read the MindRecord file using `MindDataset`.

1. Import related modules.

    ```python
    import csv
    import os
    import mindspore.dataset as ds
    from mindspore.mindrecord import CsvToMR
    ```

2. Generate a CSV file and convert it to MindRecord.

    ```python
    CSV_FILE = "test.csv"
    MINDRECORD_FILE = "test.mindrecord"

    def generate_csv():
        headers = ["id", "name", "math", "english"]
        rows = [(1, "Lily", 78.5, 90),
                (2, "Lucy", 99, 85.2),
                (3, "Mike", 65, 71),
                (4, "Tom", 95, 99),
                (5, "Jeff", 85, 78.5)]
        with open(CSV_FILE, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    generate_csv()

    if os.path.exists(MINDRECORD_FILE):
        os.remove(MINDRECORD_FILE)
        os.remove(MINDRECORD_FILE + ".db")

    csv_transformer = CsvToMR(CSV_FILE, MINDRECORD_FILE, partition_number=1)

    csv_transformer.transform()

    assert os.path.exists(MINDRECORD_FILE)
    assert os.path.exists(MINDRECORD_FILE + ".db")
    ```

    **Parameter description:**
   - `CSV_FILE`: path of the CSV file.
   - `MINDRECORD_FILE`: path of the output MindRecord file.

3. Read MindRecord using `MindDataset`.

    ```python
    data_set = ds.MindDataset(dataset_file=MINDRECORD_FILE)
    count = 0
    for item in data_set.create_dict_iterator(output_numpy=True):
        count += 1
    print("Got {} samples".format(count))
    ```

    ```text
    Got 5 samples
    ```

### Converting TFRecord Dataset

> Currently, only TensorFlow 1.13.0-rc1 and later versions are supported.

In this part of the example, TensorFlow needs to be installed in advance. If it is not installed, execute the following command to install it. For example, when this document is running as a Notebook, after the installation is complete, you need to restart the kernel to execute the subsequent code.

```python
os.system('pip install tensorflow') if os.system('python -c "import tensorflow"') else print("TensorFlow installed")
```

```text
0
```

Use TensorFlow to create a TFRecord file and convert the file to MindRecord using the `TFRecordToMR` tool class. Read the file using `MindDataset` and decode the `image_bytes` field using the `Decode` operator.

1. Import related modules.

    ```python
    import collections
    from io import BytesIO
    import os
    import mindspore.dataset as ds
    from mindspore.mindrecord import TFRecordToMR
    import mindspore.dataset.vision.c_transforms as vision
    from PIL import Image
    import tensorflow as tf
    ```

2. Generate a TFRecord file.

    ```python
    TFRECORD_FILE = "test.tfrecord"
    MINDRECORD_FILE = "test.mindrecord"

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

        writer = tf.io.TFRecordWriter(TFRECORD_FILE)

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
    ```

    ```text
    Write 10 rows in tfrecord.
    ```

    **Parameter description:**
   - `TFRECORD_FILE`: path of the TFRecord file.
   - `MINDRECORD_FILE`: path of the output MindRecord file.

3. Convert the TFRecord to MindRecord.

    ```python
    feature_dict = {"file_name": tf.io.FixedLenFeature([], tf.string),
                    "image_bytes": tf.io.FixedLenFeature([], tf.string),
                    "int64_scalar": tf.io.FixedLenFeature([], tf.int64),
                    "float_scalar": tf.io.FixedLenFeature([], tf.float32),
                    "int64_list": tf.io.FixedLenFeature([6], tf.int64),
                    "float_list": tf.io.FixedLenFeature([7], tf.float32),
                    }

    if os.path.exists(MINDRECORD_FILE):
        os.remove(MINDRECORD_FILE)
        os.remove(MINDRECORD_FILE + ".db")

    tfrecord_transformer = TFRecordToMR(TFRECORD_FILE, MINDRECORD_FILE, feature_dict, ["image_bytes"])
    tfrecord_transformer.transform()

    assert os.path.exists(MINDRECORD_FILE)
    assert os.path.exists(MINDRECORD_FILE + ".db")
    ```

4. Read MindRecord using `MindDataset`.

    ```python
    data_set = ds.MindDataset(dataset_file=MINDRECORD_FILE)
    decode_op = vision.Decode()
    data_set = data_set.map(operations=decode_op, input_columns=["image_bytes"], num_parallel_workers=2)
    count = 0
    for item in data_set.create_dict_iterator(output_numpy=True):
        count += 1
    print("Got {} samples".format(count))
    ```

    ```text
    Got 10 samples
    ```
