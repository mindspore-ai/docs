# Data Processing

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/faq/data_processing.md)

## Q: How do I offload data if I do not use high-level APIs?

A: You can implement by referring to the [test_tdt_data_transfer.py](https://gitee.com/mindspore/mindspore/blob/master/tests/st/data_transfer/test_tdt_data_transfer.py) example of the manual offloading mode without using the `model.train` API. Currently, the GPU-based and Ascend-based hardware is supported.

<br/>

## Q: In the process of using `Dataset` to process data, the memory consumption is high. How to optimize it?

A: You can refer to the following steps to reduce the memory occupation, which may also reduce the efficiency of data processing.

1. Before defining the dataset `**Dataset` object, set the prefetch size of `Dataset` data processing, `ds.config.set_prefetch_size(2)`.

2. When defining the `**Dataset` object, set its parameter `num_parallel_workers` as 1.

3. If you further use `.map(...)` operation on `**Dataset` object, you can set `.map(...)` operation's parameter `num_parallel_workers` as 1.

4. If you further use `.batch(...)` operation on `**Dataset` object, you can set `.batch(...)` operation's parameter `num_parallel_workers` as 1.

5. If you further use `.shuffle(...)` operation on `**Dataset` object, you can reduce the parameter `buffer_size`.

<br/>

## Q: In the process of using `Dataset` to process data, the CPU occupation is high which shows that sy occupation is high and us occupation is low. How to optimize it?

A: You can refer to the following steps to reduce CPU consumption (mainly due to resource competition between third-party library multithreading and data processing multithreading) and further improve performance.

1. If there is a `cv2` operation of opencv in the data processing, use `cv2.setNumThreads(2)` to set the number of `cv2` global threads.

2. If there is a `numpy` operation in the data processing, use `export OPENBLAS_NUM_THREADS=1` to set the number of `OPENBLAS` threads.

3. If there is a `numba` operation in the data processing, use `numba.set_num_threads(1)` to set the number of threads for `numba`.

<br/>

## Q:  Why there is no difference between the parameter `shuffle` in `GeneratorDataset`, and `shuffle=True` and `shuffle=False` when the task is run?

A: If `shuffle` is enabled, the input `Dataset` must support random access (for example, the user-defined `Dataset` has the `getitem` method). If data is returned in `yeild` mode in the user-defined `Dataset`, random access is not supported. For details, see section [GeneratorDataset example](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html).

<br/>

## Q: How does `Dataset` combine two `columns` into one `column`?

A: You can perform the following operations to combine the two columns into one:

```python
def combine(x, y):
    x = x.flatten()
    y = y.flatten()
    return np.append(x, y)

dataset = dataset.map(operations=combine, input_columns=["data", "data2"], output_columns=["data"])
```

Note: The `shapes` of the two `columns` are different. Therefore, you need to `flatten` them before combining.

<br/>

## Q: Does `GeneratorDataset` support `ds.PKSampler` sampling?

A: The user-defined dataset `GeneratorDataset` does not support `PKSampler` sampling logic. The main reason is that the customizing data operation is too flexible. The built-in `PKSampler` cannot be universal. Therefore, a message is displayed at the API layer, indicating that the operation is not supported. However, for `GeneratorDataset`, you can easily define the required `Sampler` logic. That is, you can define specific `sampler` rules in the `__getitem__` function of the `ImageDataset` class and return the required data.

<br/>

## Q: How does MindSpore load the existing pre-trained word vector?

A: When defining EmbedingLookup or Embedding, you only need to transfer the pre-trained word vector and encapsulate the pre-trained word vector into a tensor as the initial value of EmbeddingLookup.

<br/>

## Q: What is the difference between `c_transforms` and `py_transforms`? Which one is recommended?

A: `c_transforms` is recommended. Its performance is better because it is executed only at the C layer.

Principle: The underlying layer of `c_transform` uses `opencv/jpeg-turbo` of the C version for data processing, and `py_transform` uses `Pillow` of the Python version for data processing.

Data augmentation APIs are unified in MindSpore 1.8. Transformations of `c_transforms` and `py_transforms` will be selected automatically due to input tensor type instead of importing them manually. `c_transforms` is set to default option since its performance is better. More details please refer to [Latest API doc and import note](https://mindspore.cn/docs/en/master/api_python/mindspore.dataset.transforms.html#module-mindspore.dataset.vision).

<br/>

## Q: A piece of data contains multiple images which have different widths and heights. I need to perform the `map` operation on the data in mindrecord format. However, the data I read from `record` is in `np.ndarray` format. My `operations` of data processing are for the image format. How can I preprocess the generated data in mindrecord format?

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

#3 Use MindDataset to load, then use the decode operation we provide to decode, and then perform subsequent processing.

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

## Q: When a customizing image dataset is converted to the mindrecord format, the data is in the `numpy.ndarray` format and `shape` is [4,100,132,3], indicating four three-channel frames, and each value ranges from 0 to 255. However, when I view the data that is converted into the mindrecord format, I find that the `shape` is `[19800]` and the dimensions of the original data are all expanded as`[158400]`. Why?

A: The value of `dtype` in `ndarray` might be set to `int8`. The difference between `[158400]` and `[19800]` is eight times. You are advised to set `dtype` of `ndarray` to `float64`.

<br/>

## Q: I want to save the generated image, but the image cannot be found in the corresponding directory after the code is executed. Similarly, a dataset is generated in JupyterLab for training. During training, data can be read in the corresponding path, but the image or dataset cannot be found in the path. Why?

A: The images or datasets generated by JumperLab are stored in Docker. The data downloaded by `moxing` can be viewed only in Docker during the training process. After the training is complete, the data is released with Docker. You can try to transfer the data that needs to be `download` to `obs` through `moxing` in the training task, and then download the data to the local host through `obs`.

<br/>

## Q: How do I understand the `dataset_sink_mode` parameter in `model.train` of MindSpore?

A: When `dataset_sink_mode` is set to `True`, data processing and network computing are performed in pipeline mode. That is, when data processing is performed step by step, after a `batch` of data is processed, the data is placed in a queue which is used to cache the processed data. Then, network computing obtains data from the queue for training. In this case, data processing and network computing are performed in `pipeline` mode. The entire training duration is the longest data processing/network computing duration.

When `dataset_sink_mode` is set to `False`, data processing and network computing are performed in serial mode. That is, after a `batch` of data is processed, it is transferred to the network for computation. After the computation is complete, the next `batch` of data is processed and transferred to the network for computation. This process repeats until the training is complete. The total time consumed for the training is the time consumed for data processing plus the time consumed for network computing.

<br/>

## Q: Can MindSpore train image data of different sizes by batch?

A: You can refer to the usage of YOLOv3 which contains the resizing of different images. For details about the script, see [yolo_dataset](https://gitee.com/mindspore/models/blob/master/official/cv/YOLOv3/src/yolo_dataset.py).

<br/>

## Q: Must data be converted into MindRecords when MindSpore is used for segmentation training?

A: [build_seg_data.py](https://gitee.com/mindspore/models/blob/master/research/cv/FCN8s/src/data/build_seg_data.py) is the script of MindRecords generated by the dataset. You can directly use or adapt it to your dataset. Alternatively, you can use `GeneratorDataset` to customize the dataset loading if you want to implement the dataset reading by yourself.

[GeneratorDataset example](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html)

<br/>

## Q: When MindSpore performs multi-device training on the Ascend hardware platform, how does the user-defined dataset transfer data to different chip?

A: When `GeneratorDataset` is used, the `num_shards=num_shards` can be used. `shard_id=device_id` parameters can be used to control which shard of data is read by different devices. `__getitem__` and `__len__` are processed as full datasets.

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

## Q: How do I build a multi-label MindRecord dataset for images?

A: The data schema can be defined as follows:`cv_schema_json = {"label": {"type": "int32", "shape": [-1]}, "data": {"type": "bytes"}}`

Note: A label is an array of the numpy type, where label values 1, 1, 0, 1, 0, 1 are stored. These label values correspond to the same data, that is, the binary value of the same image.
For details, see [Converting Dataset to MindRecord](https://www.mindspore.cn/tutorials/en/master/dataset/record.html#converting-dataset-to-record-format).

<br/>

## Q: What can I do if an error message `wrong shape of image` is displayed when I use a model trained by MindSpore to perform prediction on a `28 x 28` digital image made by myself with white text on a black background?

A: The MNIST gray scale image dataset is used for MindSpore training. Therefore, when the model is used, the data must be set to a `28 x 28` gray scale image, that is, a single channel.

<br/>

## Q: Can you introduce the data processing framework in MindSpore?

A: MindSpore Dataset module makes it easy for users to define data preprocessing pipelines and transform samples efficiently with multiprocessing or multithreading. MindSpore Dataset also provides variable APIs for users to load and process datasets, more introduction please refer to [MindSpore Dataset](https://mindspore.cn/docs/en/master/api_python/mindspore.dataset.html). If you want to further study the performance optimization of dataset pipeline, please read [Optimizing Data Processing](https://www.mindspore.cn/tutorials/en/master/dataset/optimize.html).

<br/>

## Q: When an error message  that "TDT Push data into device Failed" is displayed during network training, how to locate the problem?

A: Firstly, above error refers to failed sending data to the device through the training data transfer channel (TDT). Here are several possible reasons for this error. Therefore, the corresponding checking suggestions are given in the log. In detail:

1. Commonly, we will find the first error (the first ERROR level error) or error TraceBack thrown in the log, and try to find information that helps locate the cause of the error.

2. **When error raised in the graph compiling stage, as training has not started** (for example, the loss has not been printed in the log), please check the error log if there are errors reported by the network related operators or the environment configuration resulted Errors (such as hccl.json is incorrect, resulted abnormal initialization of multi-card communication)

3. **When error raised during the training process**, usually this is caused by the mismatch between the amount of data (batch number) has been sent and the amount of data (step number) required for network training. You can print and check the number of batches of an epoch with [get_dataset_size](https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/attribute/mindspore.dataset.Dataset.get_dataset_size.html) interface, several possible reason are as follows:

    - With checking the print times of loss to figure out that when data amount(trained steps) is just an integer multiple of the batches number in an epoch, there may be a processing existence problem in the data processing part involving Epoch processing, such as the following case:

        ```python
        ...
        dataset = dataset.create_tuple_iteator(num_epochs=-1) # Here, if you want to return an iterator, num_epochs should be 1, but it is recommended to return dataset directly
        return dataset
        ```

    - The data processing performance is slow, and cannot keep up with the speed of network training. For this case, you can use the profiler tool to see if there is an obvious iteration gap, or manually iterating the dataset, and print the average single batch time if it is longer than the combined forward and backward time of the network. There is a high probability that the performance of the data processing part needs to be optimized if yes.

    - During the training process, the occurrence of abnormal data may resulted in exception, causing sending data failed. In this case, there will be other `ERROR` logs that shows which part of the data processing process is abnormal and checking advice. If it is not obvious, you can also try to find the abnormal data by iterating each data batch in the dataset (such as turning off shuffle, and using dichotomy).

4. **When after training** the log is printed (this is probably caused by forced release of resources), this error can be ignored.

5. If the specific cause cannot be located, please create issue or raise question to ask the module developers for help.

<br/>

## Q: Can the py_transforms and c_transforms operations be used together? If yes, how should I use them?

A: To ensure high performance, you are not advised to use the py_transforms and c_transforms operations together. However, if the main consideration is to streamline the process, the performance can be compromised more or less. If you cannot use all the c_transforms operations, that is, corresponding certain c_transforms operations are not available, the py_transforms operations can be used instead. In this case, the two operations are used together.
Note that the c_transforms operation usually outputs numpy array, and the py_transforms operation outputs PIL Image. For details, check the operation description. The common method to use them together is as follows:

- c_transforms operation + ToPIL operation + py_transforms operation + ToNumpy operation
- py_transforms operation + ToNumpy operation + c_transforms operation

```python
# example that using c_transforms and py_transforms operations together
# in following case: c_vision refers to c_transforms, py_vision refer to py_transforms
import mindspore.vision.c_transforms as c_vision
import mindspore.vision.py_transforms as py_vision

decode_op = c_vision.Decode()

# If input type is not PIL, then add ToPIL operation.
transforms = [
    py_vision.ToPIL(),
    py_vision.CenterCrop(375),
    py_vision.ToTensor()
]
transform = mindspore.dataset.transforms.Compose(transforms)
data1 = data1.map(operations=decode_op, input_columns=["image"])
data1 = data1.map(operations=transform, input_columns=["image"])
```

From MindSpore 1.8, the code above can be simpler since we unify the APIs of data augmentation.

```python
import mindspore.vision as vision

transforms = [
    vision.Decode(),         # default to use c_transforms
    vision.ToPIL(),          # switch to PIL backend
    vision.CenterCrop(375),  # use py_transforms
]

data1 = data1.map(operations=transforms, input_columns=["image"])
```

<br/>

## Q: Why is the error message "The data pipeline is not a tree (i.e., one node has 2 consumers)" displayed?

A: The preceding error is usually caused by incorrect script writing. In normal cases, operations in the data processing pipeline are connected in sequence, for example

```python
# pipeline definition
# dataset1 -> map -> shuffle -> batch
dataset1 = XXDataset()
dataset1 = dataset1.map(...)
dataset1 = dataset1.shuffle(...)
dataset1 = dataset1.batch(...)
```

However, in the following exception scenario, dataset1 has two consumption nodes dataset2 and dataset3. As a result, the direction of data flow from dataset1 is undefined, thus the pipeline definition is invalid.

```python
# pipeline definition:
# dataset1 -> dataset2 -> map
#          |
#          --> dataset3 -> map
dataset1 = XXDataset()
dataset2 = dataset1.map(***)
dataset3 = dataset1.map(***)
```

The correct format is as follows. dataset3 is obtained by performing data enhancement on dataset2 rather than dataset1.

```python
dataset2 = dataset1.map(***)
dataset3 = dataset2.map(***)
```

<br/>

## Q: What is the API corresponding to DataLoader in MindSpore?

A: If the DataLoader is considered as an API for receiving user-defined datasets, the GeneratorDataset in the MindSpore data processing API is similar to that in the DataLoader and can receive user-defined datasets. For details about how to use the GeneratorDataset, see the [GeneratorDataset example](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html), and for details about the differences, see the [API Mapping](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html).

<br/>

## Q: How do I debug a user-defined dataset when an error occurs?

A: Generally, a user-defined dataset is imported to GeneratorDataset. If the user-defined dataset is incorrectly pointed to, you can use some methods for debugging (for example, adding printing information and printing the shape and dtype of the return value). The intermediate processing result of a user-defined dataset is numpy array. You are not advised to use it together with the MindSpore network computing operator. In addition, for the user-defined dataset, such as MyDataset shown below, after initialization, you can directly perform the following inritations (to simplify debugging and analyze problems in the original dataset, you do not need to import GeneratorDataset). The debugging complies with common Python syntax rules.

```python
Dataset = MyDataset()
for item in Dataset:
   print("item:", item)
```

<br/>

## Q: Can the data processing operation and network computing operator be used together?

A: Generally, if the data processing operation and network computing operator are used together, the performance deteriorates. If the corresponding data processing operation is unavailable and the user-defined py_transforms operation is inappropriate, you can try to use the data processing operation and network computing operator together. Note that because the inputs required are different, the input of the data processing operation is Numpy array or PIL Image, but the input of the network computing operator must be MindSpore.Tensor.
To use these two together, ensure that the output format of the previous one is the same as the input format of the next. Data processing operations refer to APIs in [mindspore.dataset](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.html) module on the official website, for example, [mindspore.dataset.vision.CenterCrop](https://www.mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.CenterCrop.html). Network computing operators include operators in the mindspore.nn and mindspore.ops modules.

<br/>

## Q: Why is a .db file generated in MindRecord? What is the error reported when I load a dataset without a .db file?

A: The .db file is the index file corresponding to the MindRecord file. If the .db file is missing, an error is reported when the total data volume of the dataset is obtained. The error message `MindRecordOp Count total rows failed` is displayed.

<br/>

## Q: How to read image and perform Decode operation in user-defined Dataset?

A: The user-defined Dataset is passed into GeneratorDataset, and after reading the image inside the interface (such as `__getitem__` function), it can directly return bytes type data, numpy array type array or numpy array that has been decoded, as shown below:

- Return bytes type data directly after reading the image

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
    # In the above case, the __getitem__ function can be modified as follows, and the Decode operation is the same as the above use case
    def __getitem__(self, index):
        # use np.fromfile to read image
        img_np = np.fromfile(self.data[index])

        # return Numpy array directly
        return (img_np, )
    ```

- Perform Decode operation directly after reading the image

    ```python
    # According to the above case, the __getitem__ function can be modified as follows to directly return the data after Decode. After that, there is no need to add Decode operation through the map operation.
    def __getitem__(self, index):
        # use Image.Open to open file, and convert to RGC
        img_rgb = Image.Open(self.data[index]).convert("RGB")
        return (img_rgb, )
    ```

<br/>

## Q: In the process of using `Dataset` to process data, an error `RuntimeError: can't start new thread` is reported. How to solve it?

A: The main reason is that the parameter `num_parallel_workers` is configured too large while using `**Dataset`, `.map(...)` and `.batch(...)` and the number of user processes reaches the maximum. You can increase the range of the maximum number of user processes through `ulimit -u MAX_PROCESSES`, or reduce `num_parallel_workers`.

<br/>

## Q: In the process of using `GeneratorDataset` to load data, an error `RuntimeError: Failed to copy data into tensor.` is reported. How to solve it?

A: When the `GeneratorDataset` is used to load Numpy array returned by Pyfunc, MindSpore performs conversion from the Numpy array to the MindSpore Tensor. If the memory pointed to by the Numpy array has been freed, a memory copy error may occur. An example is as shown below:

- Perform an in place conversion among Numpy array, MindSpore Tensor and Numpy array in `__getitem__` function. Tensor `tensor` and Numpy array `ndarray_1` share the same memory and Tensor `tensor` will go out of scope when the function exits, and the memory which is pointed to by Numpy array will be freed.

    ```python

    class RandomAccessDataset:
        def __init__(self):
            pass

        def __getitem__(self, item):
            ndarray = np.zeros((544, 1056, 3))
            tensor = Tensor.from_numpy(ndarray)
            ndarray_1 = tensor.asnumpy()
            return ndarray_1

        def __len__(self):
            return 8

    data1 = ds.GeneratorDataset(RandomAccessDataset(), ["data"])

    ```

- Ignore the cyclic conversion in the example above. When `__getitem__` function exits, Tensor `tensor` is freed, and the behavior of Numpy array `ndarray_1` that shares the same memory with `tensor` will become unpredictable. To avoid the issue, we can use the `deepcopy` function to apply for independent memory for the returned Numpy array `ndarray_2`.

    ```python

    class RandomAccessDataset:
        def __init__(self):
            pass

        def __getitem__(self, item):
            ndarray = np.zeros((544, 1056, 3))
            tensor = Tensor.from_numpy(ndarray)
            ndarray_1 = tensor.asnumpy()
            ndarray_2 = copy.deepcopy(ndarray_1)
            return ndarray_2

        def __len__(self):
            return 8

    data1 = ds.GeneratorDataset(RandomAccessDataset(), ["data"])

    ```

<br/>

## Q: How to determine the cause of GetNext timeout based on the exit status of data preprocessing?

A: When using the data sinking mode (where `data preprocessing` -> `sending queue` -> `network computing` form the pipeline mode) for training and there is a GetNext timeout error, the data preprocessing module will output status information to help users analyze the cause of the error. Users can enable log output through the environment variable `export MS_SUBMODULE_LOG_v={MD:1}`. `channel_name` represents the name of the data channel sent by the host to the device side, `have_sent` represents the total number of data sent to the device, `host_queue` represents the size of the host side queue for the last 10 times, `device_queue` represents the size of the device side queue for the last 10 times, `push_first_start_time` and `push_first_end_time` represent the starting time and the ending time of the first data sent by the host to the device side, and `push_start_time` and `push_end_time` represent the starting time and the ending time of the last 10 data sent by the host to the device side. Users can see the following situations in the log, and for the specific reasons and improvement methods, refer to:

1. When the log output is similar to the following, it indicates that the data preprocessing has not generated any data that can be used for training.

    ```
    channel_name: 29475464-f51b-11ee-b72b-8feb6783b0c3
    have_sent: 0;
    host_queue: ;
    device_queue: ;
          push_first_start_time -> push_first_end_time
                             -1 -> -1
                push_start_time -> push_end_time
    ```

    Improvement method: You can loop through the dataset to confirm if the dataset preprocessing is normal.

2. When the log output is similar to the following, it indicates that data preprocessing has generated a batch of data, but it has not been sent to the device side yet.

    ```
    channel_name: 29475464-f51b-11ee-b72b-8feb6783b0c3
    have_sent: 0;
    host_queue: 1;
    device_queue: ;
          push_first_start_time -> push_first_end_time
    2022-05-09-11:36:00.521.386 -> -1
                push_start_time -> push_end_time
    2022-05-09-11:36:00.521.386 ->
    ```

    Improvement method: You can check if the device plog has an error message.

3. When the log output is similar to the following, it indicates that data preprocessing has generated three batches of data, all of which have been sent to the device side, and the fourth batch of data is being preprocessed.

    ```
    channel_name: 29475464-f51b-11ee-b72b-8feb6783b0c3
    have_sent: 3;
    host_queue: 1, 0, 1;
    device_queue: 1, 2, 3;
          push_first_start_time -> push_first_end_time
    2022-05-09-11:36:00.521.386 -> 2022-05-09-11:36:00.782.215
                push_start_time -> push_end_time
    2022-05-09-11:36:00.521.386 -> 2022-05-09-11:36:00.782.215
    2022-05-09-11:36:01.212.621 -> 2022-05-09-11:36:01.490.139
    2022-05-09-11:36:01.893.412 -> 2022-05-09-11:36:02.006.771
    ```

    Improvement method: View the time difference between the last item of `push_end_time` and GetNext error reporting time. If the default GetNext timeout is exceeded (default: 1900s, and can be modified through  `mindspore.device_context.ascend.op_debug.execute_timeout(xx)`), it indicates poor data preprocessing performance. Please refer to [Optimizing the Data Processing](https://www.mindspore.cn/tutorials/en/master/dataset/optimize.html) to improve data preprocessing performance.

4. When the log output is similar to the following, it indicates that data preprocessing has generated 182 batches of data and the 183st batch of data is being sent to the device. And the `device_queue` shows that there is sufficient data cache on the device side.

    ```
    channel_name: 29475464-f51b-11ee-b72b-8feb6783b0c3
    have_sent: 182;
    host_queue: 1, 0, 1, 1, 2, 1, 0, 1, 1, 0;
    device_queue: 100, 100, 99, 100, 100, 100, 100, 100, 99, 100;
          push_first_start_time -> push_first_end_time
    2022-05-09-13:23:00.179.611 -> 2022-05-09-13:23:00.181.784
                push_start_time -> push_end_time
                                -> 2022-05-09-14:31:00.603.866
    2022-05-09-14:31:00.621.146 -> 2022-05-09-14:31:01.018.964
    2022-05-09-14:31:01.043.705 -> 2022-05-09-14:31:01.396.650
    2022-05-09-14:31:01.421.501 -> 2022-05-09-14:31:01.807.671
    2022-05-09-14:31:01.828.931 -> 2022-05-09-14:31:02.179.945
    2022-05-09-14:31:02.201.960 -> 2022-05-09-14:31:02.555.941
    2022-05-09-14:31:02.584.413 -> 2022-05-09-14:31:02.943.839
    2022-05-09-14:31:02.969.583 -> 2022-05-09-14:31:03.309.299
    2022-05-09-14:31:03.337.607 -> 2022-05-09-14:31:03.684.034
    2022-05-09-14:31:03.717.230 -> 2022-05-09-14:31:04.038.521
    2022-05-09-14:31:04.064.571 ->
    ```

    Improvement method: You can check if the device plog has an error message.

5. When the log output is similar to the following, many zeros appear in `device_queue` , indicating that data preprocessing is too slow, which can lead to slower network training.

    ```
    channel_name: 29475464-f51b-11ee-b72b-8feb6783b0c3
    have_sent: 390;
    host_queue: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    device_queue: 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
          push_first_start_time -> push_first_end_time
    2022-05-09-13:23:00.179.611 -> 2022-05-09-13:23:00.181.784
                push_start_time -> push_end_time
                                -> 2022-05-09-14:31:00.603.866
    2022-05-09-14:31:00.621.146 -> 2022-05-09-14:31:01.018.964
    2022-05-09-14:31:01.043.705 -> 2022-05-09-14:31:01.396.650
    2022-05-09-14:31:01.421.501 -> 2022-05-09-14:31:01.807.671
    2022-05-09-14:31:01.828.931 -> 2022-05-09-14:31:02.179.945
    2022-05-09-14:31:02.201.960 -> 2022-05-09-14:31:02.555.941
    2022-05-09-14:31:02.584.413 -> 2022-05-09-14:31:02.943.839
    2022-05-09-14:31:02.969.583 -> 2022-05-09-14:31:03.309.299
    2022-05-09-14:31:03.337.607 -> 2022-05-09-14:31:03.684.034
    2022-05-09-14:31:03.717.230 -> 2022-05-09-14:31:04.038.521
    2022-05-09-14:31:04.064.571 ->
    ```

    Improvement method: Please refer to [Optimizing the Data Processing](https://www.mindspore.cn/tutorials/en/master/dataset/optimize.html) to improve data preprocessing performance.

<br/>

## Q: How to handle data processing error `Malloc device memory failed, free memory size is less than half of total memory size.Device 0 Device MOC total size:65464696832 Device MOC free size:3596279808 may be other processes occupying this card, ...` ?

A: This is usually caused by the use of customized data enhancement operations (which include Ascend-based data enhancement operations) and enabling multiprocessing mode, resulting in multiple processes using the same card resources to run out of device memory.

The error message is as follows:

```text
E    ------------------------------------------------------------------
E    - Python Call Stack:
E    ------------------------------------------------------------------
E    map operation: [PyFunc] failed. The corresponding data file is: ../ut/data/dataset/testImageNetData2/train/class1/1_1.jpg. Error description:
E    RuntimeError: Traceback (most recent call last):
E      File "/opt/buildtools/python-3.9.11/lib/python3.9/site-packages/mindspore/dataset/transforms/py_transforms_util.py", line 199, in __call__
E        result = self.transform(*args)
E      File "/data/test/mindspore/tests/st/dataset/test_map_dvpp.py", line 63, in pyfunc2
E        img_decode = vision.Decode().device("Ascend")(img_bytes)
E      File "/opt/buildtools/python-3.9.11/lib/python3.9/site-packages/mindspore/dataset/vision/transforms.py", line 1564, in __call__
E        return super().__call__(img)
E      File "/opt/buildtools/python-3.9.11/lib/python3.9/site-packages/mindspore/dataset/vision/transforms.py", line 97, in __call__
E        return super().__call__(*input_tensor_list)
E      File "/opt/buildtools/python-3.9.11/lib/python3.9/site-packages/mindspore/dataset/transforms/transforms.py", line 105, in __call__
E        executor = cde.Execute(self.parse())
E    RuntimeError: Ascend kernel runtime initialization failed. The details refer to 'Ascend Error Message'.
E
E    ----------------------------------------------------
E    - Ascend Error Message:
E    ----------------------------------------------------
E    EE1001: The argument is invalid.Reason: rtGetDevMsg execute failed, reason=[context pointer null]
E            Solution: 1.Check the input parameter range of the function. 2.Check the function invocation relationship.
E            TraceBack (most recent call last):
E            ctx is NULL![FUNC:GetDevErrMsg][FILE:api_impl.cc][LINE:4692]
E            The argument is invalid.Reason: rtGetDevMsg execute failed, reason=[context pointer null]
E
E    (Please search "CANN Common Error Analysis" at https://www.mindspore.cn for error code description)
E
E    ----------------------------------------------------
E    - Framework Error Message:
E    ----------------------------------------------------
E    Malloc device memory failed, free memory size is less than half of total memory size.Device 0 Device MOC total size:65464696832 Device MOC free size:3596279808 may be other processes occupying this card, check as: ps -ef|grep python
E
E    ----------------------------------------------------
E    - C++ Call Stack: (For framework developers)
E    ----------------------------------------------------
E    mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_kernel_runtime.cc:354 Init
E    mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_memory_adapter.cc:65 Initialize
E
E    ------------------------------------------------------------------
E    - Dataset Pipeline Error Message:
E    ------------------------------------------------------------------
E    [ERROR] Execute user Python code failed, check 'Python Call Stack' above.
E
E    ------------------------------------------------------------------
E    - C++ Call Stack: (For framework developers)
E    ------------------------------------------------------------------
E    mindspore/ccsrc/minddata/dataset/engine/datasetops/map_op/map_job.h(57).
```

It can be resolved through the following methods, set `ms.runtime.set_memory(max_size="2GB")` in the custom function to reduce device memory usage for multiple processes.

The error script is as follows:

```python
def pyfunc(img_bytes):
    img_decode = vision.Decode().device("Ascend")(img_bytes)

    # resize(cpu)
    img_resize = vision.Resize(size=(64, 32))(img_decode)

    # normalize(dvpp)
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_resize)
    return img_normalize

# multi process mode
data2 = data2.map(pyfunc, input_columns="image", python_multiprocessing=True)
```

The repaired script is as follows:

```python
def pyfunc(img_bytes):
    ms.runtime.set_memory(max_size="2GB")

    img_decode = vision.Decode().device("Ascend")(img_bytes)

    # resize(cpu)
    img_resize = vision.Resize(size=(64, 32))(img_decode)

    # normalize(dvpp)
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_resize)
    return img_normalize

# multi process mode
data2 = data2.map(pyfunc, input_columns="image", python_multiprocessing=True)
```

<br/>

## Q: In which scenarios do GeneratorDataset and map support calling dvpp operator?

A: For GeneratorDataset and map:

<table>
    <tr>
        <td rowspan="2"></td>
        <td rowspan="2" style="text-align: center">Multithreading</td>
        <td colspan="2" style="text-align: center">Multiprocessing</td>
    </tr>
    <tr>
        <td style="text-align: center">spawn</td>
        <td style="text-align: center">fork</td>
    </tr>
    <tr>
        <td>Independent process mode</td>
        <td>Data Processing: support<br>Data Processing + Network training: not support</td>
        <td>Data Processing: support<br>Data Processing + Network training: support</td>
        <td>Data Processing: support<br>Data Processing + Network training: not support</td>
    </tr>
    <tr>
        <td>Non-independent process mode</td>
        <td>Data Processing: support<br>Data Processing + Network training: support</td>
        <td>Data Processing: support<br>Data Processing + Network training: support</td>
        <td>Data Processing: support<br>Data Processing + Network training: not support</td>
    </tr>
</table>

Unsupported scenario description: error reporting behavior such as scoped acquire::dec_ref(): internal error, nullptr, coredump, out of memory, stuck, etc. may occur.

1. Do not support data processing + network training under independent process (where multi-threaded data processing is used): because the independent process is created by fork, with the network at the same time running, the device will be set in the main process first, the device cannot be reset in the dataset independent process generated by fork. As a result, the flow fails to be created.

    Some of the error messages are as follows:

    ```text
    terminate called after throwing an instance of 'std::runtime_error'
      what():  scoped acquire::dec_ref(): internal error:
    Fatal Python error: Aborted

    Current thread 0x0000fffd90b18120 (most recent call first):
    <no Python frame>
    ```

2. Do not support fork mode to start multiple processes to execute data processing + network training: When the data processing subprocess created by fork invokes the dvpp operation, gil preemption may occur during network running.

    Some of the error messages are as follows:

    ```text
    Fatal Python error: Segmentation fault

    Thread 0x0000fffef36cd120 (most recent call first):
    File "/opt/buildtools/python-3.9.11/lib/python3.9/threading.py", line 312 in wait
    File "/opt/buildtools/python-3.9.11/lib/python3.9/multiprocessing/queue.py", line 233 in _feed
    File "/opt/buildtools/python-3.9.11/lib/python3.9/threading.py", line 910 in run
    File "/opt/buildtools/python-3.9.11/lib/python3.9/threading.py", line 973 in _bootstrap_inner
    File "/opt/buildtools/python-3.9.11/lib/python3.9/threading.py", line 930 in _bootstrap
    ```

Suggestion: Replace with the above supported scenarios.

<br/>
