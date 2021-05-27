# Data Processing

`Linux` `Windows` `Ascend` `GPU` `CPU` `Environment Preparation` `Basic` `Intermediate`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_cn/data_processing.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

<font size=3>**Q: How do I understand the `dataset_sink_mode` parameter in `model.train` of MindSpore?**</font>

A: When `dataset_sink_mode` is set to `True`, data processing and network computing are performed in pipeline mode. That is, when data processing is performed step by step, after a `batch` of data is processed, the data is placed in a queue which is used to cache the processed data. Then, network computing obtains data from the queue for training. In this case, data processing and network computing are performed in pipeline mode. The entire training duration is the longest data processing/network computing duration.

When `dataset_sink_mode` is set to `False`, data processing and network computing are performed in serial mode. That is, after a `batch` of data is processed, it is transferred to the network for computation. After the computation is complete, the next `batch` of data is processed and transferred to the network for computation. This process repeats until the training is complete. The total time consumed is the time consumed for data processing plus the time consumed for network computing.

<br/>

<font size=3>**Q: Can MindSpore train image data of different sizes by batch?**</font>

A: You can refer to the usage of YOLOv3 which contains the resizing of different images. For details about the script, see [yolo_dataset](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/yolov3_darknet53/src/yolo_dataset.py).

<br/>

<font size=3>**Q: Must data be converted into MindRecords when MindSpore is used for segmentation training?**</font>

A: [build_seg_data.py](https://github.com/mindspore-ai/mindspore/blob/master/model_zoo/official/cv/deeplabv3/src/data/build_seg_data.py) is used to generate MindRecords based on a dataset. You can directly use or adapt it to your dataset. Alternatively, you can use `GeneratorDataset` if you want to read the dataset by yourself.

[GenratorDataset example](https://www.mindspore.cn/doc/programming_guide/en/master/dataset_loading.html#loading-user-defined-dataset)

[GeneratorDataset API description](https://www.mindspore.cn/doc/api_python/en/master/mindspore/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset)

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
For details, see [Converting Dataset to MindRecord](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/convert_dataset.html#id3).

<br/>

<font size=3>**Q: What can I do if an error message `wrong shape of image` is displayed when I use a model trained by MindSpore to perform prediction on a `28 x 28` digital image with white text on a black background?**</font>

A: The MNIST gray scale image dataset is used for MindSpore training. Therefore, when the model is used, the data must be set to a `28 x 28` gray scale image, that is, a single channel.

<br/>

<font size=3>**Q: Can you introduce the dedicated data processing framework?**</font>

A: MindData provides the heterogeneous hardware acceleration function for data processing. The high-concurrency data processing `pipeline` supports `NPU`, `GPU` and `CPU`. The `CPU` usage is reduced by 30%. For details, see [Optimizing Data Processing](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/optimize_data_processing.html).
