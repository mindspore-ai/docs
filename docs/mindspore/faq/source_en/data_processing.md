# Data Processing

`Linux` `Windows` `Ascend` `GPU` `CPU` `Environment Preparation` `Basic` `Intermediate`

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/faq/source_en/data_processing.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

<font size=3>**Q: How do I understand the `dataset_sink_mode` parameter in `model.train` of MindSpore?**</font>

A: When `dataset_sink_mode` is set to `True`, data processing and network computing are performed in pipeline mode. That is, when data processing is performed step by step, after a `batch` of data is processed, the data is placed in a queue which is used to cache the processed data. Then, network computing obtains data from the queue for training. In this case, data processing and network computing are performed in pipeline mode. The entire training duration is the longest data processing/network computing duration.

When `dataset_sink_mode` is set to `False`, data processing and network computing are performed in serial mode. That is, after a `batch` of data is processed, it is transferred to the network for computation. After the computation is complete, the next `batch` of data is processed and transferred to the network for computation. This process repeats until the training is complete. The total time consumed is the time consumed for data processing plus the time consumed for network computing.

<br/>

<font size=3>**Q: Can MindSpore train image data of different sizes by batch?**</font>

A: You can refer to the usage of YOLOv3 which contains the resizing of different images. For details about the script, see [yolo_dataset](https://gitee.com/mindspore/mindspore/blob/r1.3/model_zoo/official/cv/yolov3_darknet53/src/yolo_dataset.py).

<br/>

<font size=3>**Q: Must data be converted into MindRecords when MindSpore is used for segmentation training?**</font>

A: [build_seg_data.py](https://github.com/mindspore-ai/mindspore/blob/r1.3/model_zoo/official/cv/deeplabv3/src/data/build_seg_data.py) is used to generate MindRecords based on a dataset. You can directly use or adapt it to your dataset. Alternatively, you can use `GeneratorDataset` if you want to read the dataset by yourself.

[GenratorDataset example](https://www.mindspore.cn/docs/programming_guide/en/r1.3/dataset_loading.html#loading-user-defined-dataset)

[GeneratorDataset API description](https://www.mindspore.cn/doc/api_python/en/r1.3/mindspore/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset)

<br/>

<font size=3>**Q: How do I perform training without processing data in MindRecord format?**</font>

A: You can use the customized data loading method `GeneratorDataset`. For details, click [here](https://www.mindspore.cn/tutorial/en/r0.7/use/data_preparation/loading_the_datasets.html#id5).

<br/>

<font size=3>**Q: When MindSpore performs multi-device training on the NPU hardware platform, how does the user-defined dataset transfer data to different NPUs?**</font>

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
For details, see [Converting Dataset to MindRecord](https://www.mindspore.cn/docs/programming_guide/en/r1.3/convert_dataset.html#id3).

<br/>

<font size=3>**Q: What can I do if an error message `wrong shape of image` is displayed when I use a model trained by MindSpore to perform prediction on a `28 x 28` digital image with white text on a black background?**</font>

A: The MNIST gray scale image dataset is used for MindSpore training. Therefore, when the model is used, the data must be set to a `28 x 28` gray scale image, that is, a single channel.

<br/>

<font size=3>**Q: Can you introduce the dedicated data processing framework?**</font>

A: MindData provides the heterogeneous hardware acceleration function for data processing. The high-concurrency data processing `pipeline` supports `NPU`, `GPU` and `CPU`. The `CPU` usage is reduced by 30%. For details, see [Optimizing Data Processing](https://www.mindspore.cn/docs/programming_guide/en/r1.3/optimize_data_processing.html).

<br/>

<font size=3>**Q：When error raised during network training, indicating that sending data failed like "TDT Push data into device Failed", how to locate the problem?**</font>

A：Firstly, above error refers failed sending data to the device through the training data transfer channel (TDT). Here are several possible reasons for this error. Therefore, the corresponding checking suggestions are given in the log. In detail:

  1. Commonly, we will find the first error (the first ERROR level error) or error traceBack thrown in the log, and try to find information that helps locate the cause of the error.

  2. **When error raised in the graph compiling stage, as training has not started** (for example, the loss has not been printed in the log), please check the error log if there are errors reported by the network related operators or the environment configuration resulted Errors (such as hccl.json is incorrect, resulted abnormal initialization of multi-card communication)

  3. **When error raised during training process**, usually this is caused by the mismatch between the amount of data (batch number) sent by the host and the amount of data (step number) required for network training. You can print and check the number of batches of an epoch with `get_dataset_size` interface. And check the amount of data sent by the host and the amount of data received on the device (checking method is as follows):

      ```bash
      # Note: The amount of data refers to the number of columns. For example, a batch contains two columns (like image and label), then 5 batches contain 10 columns of data.
      # Note: If the environment variable "export ASCEND_SLOG_PRINT_TO_STDOUT=1" is enabled, the log in the following plog file will be printed directly on the screen or in the redirected log file.

      # Obtain data amount of host data process queue sending with tdt, pid is the training process id
      # Log file name like: plog-64944-20210531165504682.log, we can find data amount with `|wc -l` to calculate the num of `has got` log.

      grep -rn "has got" ~/ascend/log/plog/plog-pid_timestamp0.log

      # Calculate the data amount that host tdt sending to device tdt, find it with following command, value after the key words like"index is" refer to the data amount.
      grep -rn "has sent" ~/ascend/log/plog/plog-pid_timestamp0.log

      # Calculate the data amount received in device, pid refers to training process id, which is same with above host process id.
      # Find with following command, the value after key words "index=" refers to the data amount received in device.
      grep -rn "enqueue data" ~/ascend/log/device-id/device-pid_timestamp1.log
      ```

      - If the amount of data sent by the host side is equal to the amount of data received by the device side, and that value is less than the amount of data that the network training needed in normal case, here sending data failed mainly due to abnormal data processing on the host side, resulting failure to feed network training. There are three possible analysing ideas:
          - when data amount is just an integer multiple of the batches number in an epoch, there may be a problem in the data processing part involving Epoch processing, such as the following case:

          ```python
          ...
          dataset = dataset.create_tuple_iteator(num_epochs=-1) # Here, if you want to return an iterator, num_epochs should be 1, but it is recommended to return dataset directly
          return dataset
          ```

          - The data processing performance is slow, and cannot keep up with the speed of network training. For this case, you can use the profiler tool and MindInsight to see if there is an obvious iteration gap, or manually iterating the dataset, and print the average single batch time , if longer than the combined forward and backward time of the network, there is a high probability that the performance of the data processing part needs to be optimized.
          - Abnormal data occurred resulted exception raised during the training process, causing sending data failed. In this case, there will be other `ERROR` logs that shows which part of the data processing is abnormal and checking advice. If it is not obvious, you can also try to find the abnormal data by iterating each data batch in the dataset (such as turning off shuffle, and using dichotomy).
      - if the data amount mismatch in host and device(commonly, host send much more data), in this case, there might be some problem in tdt module(like back pressure), here might need module developer helps to analyse the problem.

  4. **when error raised after training**（this is probably caused by forced release of resources), this error can be ignored.

  5. If still cannot locate the specific cause, please set log level into info level level of mindspore and CANN, and check the log to see the context near error location for helpful information. The CANN host log file path is: ~/ascend/log/plog/plog-pid-timestamp.log.

      ```bash
      export GLOG_v=1                  # set mindspore log level into info level
      export GLOBAL_ASCEND_LOG_LEVEL=1 # set CANN log level into info level
      ```
