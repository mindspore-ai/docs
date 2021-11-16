# Data Processing

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/faq/source_en/data_processing.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

<font size=3>**Q: How do I offload data if I do not use high-level APIs?**</font>

A: You can refer to the [test_tdt_data_transfer.py](https://gitee.com/mindspore/mindspore/blob/master/tests/st/data_transfer/test_tdt_data_transfer.py) example of the manual offloading mode without using the `model.train` API. Currently, the GPU-based and Ascend-based hardware is supported.

<br/>

<font size=3>**Q: Why is there no difference between `shuffle=True` and `shuffle=False` in `GeneratorDataset`?**</font>

A: If `shuffle` is enabled, the input `Dataset` must support random access (for example, the user-defined `Dataset` has the `getitem` method). If data is returned in `yeild` mode in the user-defined `Dataset`, random access is not supported. For details, see section [Loading Dataset Overview](https://www.mindspore.cn/docs/programming_guide/en/master/dataset_loading.html#id5).

<br/>

<font size=3>**Q: How does `Dataset` combine two `columns` into one `column`?**</font>

A: You can perform the following operations to combine the two columns into one:

```python
def combine(x, y):
    x = x.flatten()
    y = y.flatten()
    return np.append(x, y)

dataset = dataset.map(operations=combine, input_columns=["data", "data2"], output_columns=["data"])
```

Note: The `shapes`of the two `columns` are different. Therefore, you need to `flatten` them before combining.

<br/>

<font size=3>**Q: Does `GeneratorDataset` support `ds.PKSampler` sampling?**</font>

A: `GeneratorDataset` does not support `PKSampler` sampling logic. The main reason is that the custom data operation is too flexible. The built-in `PKSampler` cannot be universal. Therefore, a message is displayed at the API layer, indicating that the operation is not supported. However, for `GeneratorDataset`, you can easily define the required `Sampler` logic. That is, you can define specific `sampler` rules in the `__getitem__` function of the `ImageDataset` class and return the required data.

<br/>

<font size=3>**Q: How does MindSpore load the existing pre-trained word vector?**</font>

A: When defining EmbedingLookup or Embedding, you only need to transfer the pre-trained word vector and encapsulate the pre-trained word vector into a tensor as the initial value of EmbeddingLookup.

<br/>

<font size=3>**Q: What is the difference between `c_transforms` and `py_transforms`? Which one is recommended?**</font>

A: `c_transforms` is recommended. Its performance is better because it is executed only at the C layer.

Principle: The underlying layer of `c_transform` uses `opencv/jpeg-turbo` of the C version for data processing, and `py_transform` uses `Pillow` of the Python version for data processing.

<br/>

<font size=3>**Q: A piece of data contains multiple images which have different widths and heights. I need to perform the `map` operation on the data in mindrecord format for data processing. However, the data I read from `record` is in `np.ndarray` format. My `operations` are for the image format. How can I preprocess the generated data in mindrecord format?**</font>

A: You are advised to perform the following operations:

```python
#1 The defined schema is as follows: Among them, data1, data2, data3, ... These fields store your image, and only the binary of the image is stored here.

cv_schema_json = {"label": {"type": "int32"}, "data1": {"type": "bytes"}, "data2": {"type": "bytes"}, "data3": {"type": "bytes"}}

#2 The organized data can be as follows, and then this data_list can be written by FileWriter.write_raw_data(...).

data_list = []
data = {}
data['label'] = 1

f = open("1.jpg", "rb")
image_bytes = f.read()
f.close

data['data1'] = image_bytes

f2 = open("2.jpg", "rb")
image_bytes2 = f2.read()
f2.close

data['data2'] = image_bytes2

f3 = open("3.jpg", "rb")
image_bytes3 = f3.read()
f3.close

data['data3'] = image_bytes3

data_list.append(data)

#3 Use MindDataset to load, then use the decode operator we provide to decode, and then perform subsequent processing.

data_set = ds.MindDataset("mindrecord_file_name")
data_set = data_set.map(input_columns=["data1"], operations=vision.Decode(), num_parallel_workers=2)
data_set = data_set.map(input_columns=["data2"], operations=vision.Decode(), num_parallel_workers=2)
data_set = data_set.map(input_columns=["data3"], operations=vision.Decode(), num_parallel_workers=2)
resize_op = vision.Resize((32, 32), interpolation=Inter.LINEAR)
data_set = data_set.map(operations=resize_op, input_columns=["data1"], num_parallel_workers=2)
for item in data_set.create_dict_iterator(output_numpy=True):
    print(item)
```

<br/>

<font size=3>**Q: When a custom image dataset is converted to the mindrecord format, the data is in the `numpy.ndarray` format and `shape` is [4,100,132,3], indicating four three-channel frames, and each value ranges from 0 to 255. However, when I view the data that is converted into the mindrecord format, I find that the `shape` is `[19800]` but that of the original data is `[158400]`. Why?**</font>

A: The value of `dtype` in `ndarray` might be set to `int8`. The difference between `[158400]` and `[19800]` is eight times. You are advised to set `dtype` of `ndarray` to `float64`.

<br/>

<font size=3>**Q: I want to save the generated image, but the image cannot be found in the corresponding directory after the code is executed. Similarly, a dataset is generated in JupyterLab for training. During training, data can be read in the corresponding path, but the image or dataset cannot be found in the path. Why?**</font>

A: The images or datasets generated by JumperLab are stored in Docker. The data downloaded by `moxing` can be viewed only in Docker during the training process. After the training is complete, the data is released with Docker. You can try to transfer the data that needs to be downloaded to `obs` through `moxing` in the training task, and then download the data to the local host through `obs`.

<br/>

<font size=3>**Q: How do I understand the `dataset_sink_mode` parameter in `model.train` of MindSpore?**</font>

A: When `dataset_sink_mode` is set to `True`, data processing and network computing are performed in pipeline mode. That is, when data processing is performed step by step, after a `batch` of data is processed, the data is placed in a queue which is used to cache the processed data. Then, network computing obtains data from the queue for training. In this case, data processing and network computing are performed in pipeline mode. The entire training duration is the longest data processing/network computing duration.

When `dataset_sink_mode` is set to `False`, data processing and network computing are performed in serial mode. That is, after a `batch` of data is processed, it is transferred to the network for computation. After the computation is complete, the next `batch` of data is processed and transferred to the network for computation. This process repeats until the training is complete. The total time consumed is the time consumed for data processing plus the time consumed for network computing.

<br/>

<font size=3>**Q: Can MindSpore train image data of different sizes by batch?**</font>

A: You can refer to the usage of YOLOv3 which contains the resizing of different images. For details about the script, see [yolo_dataset](https://gitee.com/mindspore/models/blob/master/official/cv/yolov3_darknet53/src/yolo_dataset.py).

<br/>

<font size=3>**Q: Must data be converted into MindRecords when MindSpore is used for segmentation training?**</font>

A: [build_seg_data.py](https://gitee.com/mindspore/models/blob/master/official/cv/deeplabv3/src/data/build_seg_data.py) is used to generate MindRecords based on a dataset. You can directly use or adapt it to your dataset. Alternatively, you can use `GeneratorDataset` if you want to read the dataset by yourself.

[GenratorDataset example](https://www.mindspore.cn/docs/programming_guide/en/master/dataset_loading.html#loading-user-defined-dataset)

[GeneratorDataset API description](https://www.mindspore.cn/docs/api/en/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset)

<br/>

<font size=3>**Q: When MindSpore performs multi-device training on the Ascend hardware platform, how does the user-defined dataset transfer data to different chip?**</font>

A: When `GeneratorDataset` is used, the `num_shards=num_shards` and `shard_id=device_id` parameters can be used to control which shard of data is read by different devices. `__getitem__` and `__len__` are processed as full datasets.

An example is as follows:

```python
# Device 0:
ds.GeneratorDataset(..., num_shards=8, shard_id=0, ...)
# Device 1:
ds.GeneratorDataset(..., num_shards=8, shard_id=1, ...)
# Device 2:
ds.GeneratorDataset(..., num_shards=8, shard_id=2, ...)
...
# Device 7:
ds.GeneratorDataset(..., num_shards=8, shard_id=7, ...)
```

<br/>

<font size=3>**Q: How do I build a multi-label MindRecord dataset for images?**</font>

A: The data schema can be defined as follows:`cv_schema_json = {"label": {"type": "int32", "shape": [-1]}, "data": {"type": "bytes"}}`

Note: A label is an array of the numpy type, where label values 1, 1, 0, 1, 0, 1 are stored. These label values correspond to the same data, that is, the binary value of the same image.
For details, see [Converting Dataset to MindRecord](https://www.mindspore.cn/docs/programming_guide/en/master/convert_dataset.html#id3).

<br/>

<font size=3>**Q: What can I do if an error message `wrong shape of image` is displayed when I use a model trained by MindSpore to perform prediction on a `28 x 28` digital image with white text on a black background?**</font>

A: The MNIST gray scale image dataset is used for MindSpore training. Therefore, when the model is used, the data must be set to a `28 x 28` gray scale image, that is, a single channel.

<br/>

<font size=3>**Q: Can you introduce the dedicated data processing framework?**</font>

A: MindData provides the heterogeneous hardware acceleration function for data processing. The high-concurrency data processing `pipeline` supports `Ascend`, `GPU` and `CPU`. The `CPU` usage is reduced by 30%. For details, see [Optimizing Data Processing](https://www.mindspore.cn/docs/programming_guide/en/master/optimize_data_processing.html).

<br/>

<font size=3>**Q: When error raised during network training, indicating that sending data failed like "TDT Push data into device Failed", how to locate the problem?**</font>

A: Firstly, above error refers failed sending data to the device through the training data transfer channel (TDT). Here are several possible reasons for this error. Therefore, the corresponding checking suggestions are given in the log. In detail:

  1. Commonly, we will find the first error (the first ERROR level error) or error TraceBack thrown in the log, and try to find information that helps locate the cause of the error.

  2. **When error raised in the graph compiling stage, as training has not started** (for example, the loss has not been printed in the log), please check the error log if there are errors reported by the network related operators or the environment configuration resulted Errors (such as hccl.json is incorrect, resulted abnormal initialization of multi-card communication)

  3. **When error raised during the training process**, usually this is caused by the mismatch between the amount of data (batch number) has been sent and the amount of data (step number) required for network training. You can print and check the number of batches of an epoch with `get_dataset_size` interface，several possible reason are as follows:

      - With checking the print times of loss to figure out the trained steps when error raised, when data amount(trained steps) is just an integer multiple of the batches number in an epoch, there may be a problem in the data processing part involving Epoch processing, such as the following case:

          ```python
          ...
          dataset = dataset.create_tuple_iteator(num_epochs=-1) # Here, if you want to return an iterator, num_epochs should be 1, but it is recommended to return dataset directly
          return dataset
          ```

      - The data processing performance is slow, and cannot keep up with the speed of network training. For this case, you can use the profiler tool and MindInsight to see if there is an obvious iteration gap, or manually iterating the dataset, and print the average single batch time , if longer than the combined forward and backward time of the network, there is a high probability that the performance of the data processing part needs to be optimized.

      - During the training process, the occurrence of abnormal data may resulted in exception, causing sending data failed. In this case, there will be other `ERROR` logs that shows which part of the data processing process is abnormal and checking advice. If it is not obvious, you can also try to find the abnormal data by iterating each data batch in the dataset (such as turning off shuffle, and using dichotomy).

  4. **when error raised after training**（this is probably caused by forced release of resources), this error can be ignored.

  5. If the specific cause cannot be located, please create issue or raise question in huawei clound forum for help.

<br/>

<font size=3>**Q: Can the py_transforms and c_transforms operators be used together? If yes, how should I use them?**</font>

A: To ensure high performance, you are not advised to use the py_transforms and c_transforms operators together. For details, see [Image Data Processing and Enhancement](https://www.mindspore.cn/docs/programming_guide/en/master/augmentation.html#usage-instructions). However, if the main consideration is to streamline the process, the performance can be compromised more or less. If you cannot use all the c_transforms operators, that is, certain c_transforms operators are not available, the py_transforms operators can be used instead. In this case, the two operators are used together.
Note that the c_transforms operator usually outputs numpy array, and the py_transforms operator outputs PIL Image. For details, check the operator description. The common method to use them together is as follows:

- c_transforms operator + ToPIL operator + py_transforms operator + ToTensor operator
- py_transforms operator + ToTensor operator + c_transforms operator

```python
# example that using c_transforms and py_transforms operators together
# in following case: c_vision refers to c_transforms, py_vision refer to py_transforms

decode_op = c_vision.Decode()

# If input type is not PIL, then add ToPIL operator.
transforms = [
    py_vision.ToPIL(),
    py_vision.CenterCrop(375),
    py_vision.ToTensor()
]
transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
data1 = data1.map(operations=decode_op, input_columns=["image"])
data1 = data1.map(operations=transform, input_columns=["image"])
```

<br/>

<font size=3>**Q: Why is the error message "The data pipeline is not a tree (i.e., one node has 2 consumers)" displayed?**</font>

A: The preceding error is usually caused by incorrect script writing. In normal cases, operations in the data processing pipeline are connected in sequence. In the following exception scenario, `dataset1` has two consumption nodes `dataset2` and `dataset3`. As a result, the preceding error occurs.

```python
 dataset2 = dataset1.map(***)
 dataset3 = dataset1.map(***)
```

`dataset3` is obtained by performing data enhancement on `dataset2` rather than `dataset1`. The correct format is as follows:

```python
 dataset2 = dataset1.map(***)
 dataset3 = dataset2.map(***)
```

<br/>

<font size=3>**Q: What is the operator corresponding to dataloader in MindSpore?**</font>

A: If the dataloader is considered as an API for receiving user-defined datasets, the GeneratorDataset in the MindSpore data processing API is similar to that in the dataloader and can receive user-defined datasets. For details about how to use the GeneratorDataset, see the [Loading Dataset Overview](https://www.mindspore.cn/docs/programming_guide/en/master/dataset_loading.html#loading-user-defined-dataset), and for details about the differences, see the [API Mapping](https://www.mindspore.cn/docs/note/en/master/index.html#operator_api).

<br/>

<font size=3>**Q: How do I debug a user-defined dataset when an error occurs?**</font>

A: Generally, a user-defined dataset is imported to GeneratorDataset. If the user-defined dataset is incorrectly pointed to, you can use some methods for debugging (for example, adding printing information and printing the shape and dtype of the return value). The intermediate processing result of a user-defined dataset is numpy array. You are not advised to use this operator together with the MindSpore network computing operator. In addition, you can directly traverse the user-defined dataset, such as MyDataset shown below, after initialization (to simplify debugging and analyze problems in the original dataset, you do not need to import GeneratorDataset). The debugging complies with common Python syntax rules.

```python
Dataset = MyDataset()
for item in Dataset:
   print("item:", item)
```

<br/>

<font size=3>**Q: Can the data processing operator and network computing operator be used together?**</font>

A: Generally, if the data processing operator and network computing operator are used together, the performance deteriorates. If the corresponding data processing operator is unavailable and the user-defined py_transforms operator is inappropriate, you can try to use the data processing operator and network computing operator together. Note that the input of the data processing operator is Numpy array or PIL Image, but the input of the network computing operator must be MindSpore.Tensor.
To use the two operators together, ensure that the output format of the previous operator is the same as the input format required by the next operator. Data processing operators refer to operators starting with mindspore.dataset in the API document on the official website, for example, mindspore.dataset.vision.c_transforms.CenterCrop. Network computing operators include operators in the mindspore.nn and mindspore.ops directories.

<br/>

<font size=3>**Q: Why is a .db file generated in MindRecord? What is the error reported when I load a dataset without a .db file?**</font>

A: The .db file is the index file corresponding to the MindRecord file. If the .db file is missing, an error is reported when the total data volume of the dataset is obtained. The error message `MindRecordOp Count total rows failed` is displayed.

<br/>

<font size=3>**Q: How to read image and perform Decode operation in user defined Dataset?**</font>

A: The user defined Dataset that passed into GeneratorDataset, after reading the image inside the function (such as `__getitem__` function), it can directly return bytes type data, numpy array type array or numpy array that has been decoded, as shown below:

- Return bytes of data directly after reading the image

    ```python
    class ImageDataset:
        def __init__(self, data_path):
            self.data = data_path

        def __getitem__(self, index):
            # use file open and read method
            f = open(self.data[index], 'rb')
            img_bytes = f.read()
            f.close()

            # return bytes directly
            return (img_bytes, )

        def __len__(self):
            return len(self.data)

    # data_path is a list of image file name
    dataset1 = ds.GeneratorDataset(ImageDataset(data_path), ["data"])
    decode_op = py_vision.Decode()
    to_tensor = py_vision.ToTensor(output_type=np.int32)
    dataset1 = dataset1.map(operations=[decode_op, to_tensor], input_columns=["data"])
    ```

- Return numpy array after reading the image

    ```python
    # In the above use case, the __getitem__ function can be modified as follows, and the Decode operation is the same as the above use case
    def __getitem__(self, index):
        # use np.fromfile to read image
        img_np = np.fromfile(self.data[index])

        # return Numpy array directly
        return (img_np, )
    ```

- Perform decode operation directly after reading the image

    ```python
    # According to the above use case, the __getitem__ function can be modified as follows to directly return the data after Decode. After that, there is no need to add Decode operation through the map operator.
    def __getitem__(self, index):
        # use Image.Open to open file, and convert to RGC
        img_rgb = Image.Open(self.data[index]).convert("RGB")
        return (img_rgb, )
    ```
