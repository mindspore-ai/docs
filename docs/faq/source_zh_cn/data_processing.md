# 数据处理

`Linux` `Windows` `Ascend` `GPU` `CPU` `环境准备` `初级` `中级`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_zh_cn/data_processing.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

<font size=3>**Q：MindSpore中`model.train`的`dataset_sink_mode`参数该如何理解？**</font>

A：当`dataset_sink_mode=True`时，数据处理会和网络计算构成Pipeline方式，即：数据处理在逐步处理数据时，处理完一个`batch`的数据，会把数据放到一个队列里，这个队列用于缓存已经处理好的数据，然后网络计算从这个队列里面取数据用于训练，那么此时数据处理与网络计算就`Pipeline`起来了，整个训练耗时就是数据处理/网络计算耗时最长的那个。

当`dataset_sink_mode=False`时，数据处理会和网络计算构成串行的过程，即：数据处理在处理完一个`batch`后，把这个`batch`的数据传递给网络用于计算，在计算完成后，数据处理再处理下一个`batch`，然后把这个新的`batch`数据传递给网络用于计算，如此的循环往复，直到训练完。该方法的总耗时是数据处理的耗时+网络计算的耗时=训练总耗时。

<br/>

<font size=3>**Q：MindSpore能否支持按批次对不同尺寸的图片数据进行训练？**</font>

A：你可以参考yolov3对于此场景的使用，里面有对于图像的不同缩放,脚本见[yolo_dataset](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/yolov3_darknet53/src/yolo_dataset.py)。

<br/>

<font size=3>**Q：使用MindSpore做分割训练，必须将数据转为MindRecords吗？**</font>

A：[build_seg_data.py](https://github.com/mindspore-ai/mindspore/blob/master/model_zoo/official/cv/deeplabv3/src/data/build_seg_data.py)是将数据集生成MindRecord的脚本，可以直接使用/适配下你的数据集。或者如果你想尝试自己实现数据集的读取，可以使用`GeneratorDataset`自定义数据集加载。

[GenratorDataset 示例](https://www.mindspore.cn/doc/programming_guide/zh-CN/master/dataset_loading.html#id5)

[GenratorDataset API说明](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset)

<br/>

<font size=3>**Q：如何不将数据处理为MindRecord格式，直接进行训练呢？**</font>

A：可以使用自定义的数据加载方式 `GeneratorDataset`，具体可以参考[数据集加载](https://www.mindspore.cn/doc/programming_guide/zh-CN/master/dataset_loading.html)文档中的自定义数据集加载。

<br/>

<font size=3>**Q：MindSpore在NPU硬件平台进行多卡训练，自定义数据集如何给不同NPU传递不同数据？**</font>

A：使用`GeneratorDataset`的时候，可以使用`num_shards=num_shards`,`shard_id=device_id`参数来控制不同卡读取哪个分片的数据，`__getitem__`和`__len__`按全量数据集处理即可。

举例：

```python
# 卡0：
ds.GeneratorDataset(..., num_shards=8, shard_id=0, ...)
# 卡1：
ds.GeneratorDataset(..., num_shards=8, shard_id=1, ...)
# 卡2：
ds.GeneratorDataset(..., num_shards=8, shard_id=2, ...)
...
# 卡7：
ds.GeneratorDataset(..., num_shards=8, shard_id=7, ...)
```

<br/>

<font size=3>**Q：如何构建图像的多标签MindRecord格式数据集？**</font>

A：数据Schema可以按如下方式定义：`cv_schema_json = {"label": {"type": "int32", "shape": [-1]}, "data": {"type": "bytes"}}`

说明：label是一个数组，numpy类型，这里面可以存你说的 1， 1，0，1， 0， 1 这么多label值，这些label值对应同一个data，即：同一个图像的二进制值。
可以参考[将数据集转换为MindRecord](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/convert_dataset.html#将数据集转换为MindRecord)教程。

<br/>

<font size=3>**Q：请问自己制作的黑底白字`28*28`的数字图片，使用MindSpore训练出来的模型做预测，报错提示`wrong shape of image`是怎么回事？**</font>

A：首先MindSpore训练使用的灰度图MNIST数据集。所以模型使用时对数据是有要求的，需要设置为`28*28`的灰度图，就是单通道才可以。

<br/>

<font size=3>**Q：第一次看到有专门的数据处理框架，能介绍下么？**</font>

A：MindData提供数据处理异构硬件加速功能，高并发数据处理`pipeline`同时支持`NPU/GPU/CPU`，`CPU`占用降低30%，点击查询[优化数据处理](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/optimize_data_processing.html)。
