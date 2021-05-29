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

<br/>

<font size=3>**Q：网络训练时出现报错提示数据下发失败“TDT Push data into device Failed”，如何定位原因？**</font>

A：首先上述报错指的是通过训练数据下发通道（TDT，train data transfer)发送数据到卡（device）上失败，导致这一报错的原因可能有多种，因此日志中给出了相应的检查建议，具体而言：

  1. 通常我们会找到日志中最先抛出的错误（第一个ERROR级别的错误）或报错堆栈（TraceBack)，并尝试从中找到有助于定位错误原因的信息。

  2. **在图编译阶段，训练还没开始报错时**（例如日志中还没打印loss)，请先检查下报错（ERROR）日志中是否有网络中涉及的相关算子报错或涉及环境没配置好导致的报错（如hccl.json不对导致多卡通信初始化异常）

  3. **在中间训练过程中报错时**，通常为host侧下发的数据量（batch数）与网络训练需要的数据量（step数）不匹配导致的，可以通过`get_dataset_size`接口打印一个epoch中包含的batch数，并检查host下发的数据量和device侧收到的数据量(检查方式如下)：

     ```bash
      # 注意：数据量都指的列数，如一个batch包含image, label两列，5个batch则包含10列数据
      # 注意：如果开启了环境变量“export ASCEND_SLOG_PRINT_TO_STDOUT=1"则下面plog中的日志将在屏幕上直接打印出来或在重定向的日志文件中。

      # 数据处理队列发给host tdt的数据量，pid为训练任务的进程id
      # 文件名如：plog-64944-20210531165504682.log，统计数据量可以加`|wc -l`统计has got日志的条数
      grep -rn "has got" ~/ascend/log/plog/plog-pid_timestamp0.log

      # host tdt发到device tdt的数据量，进行如下搜索，日志中关键字眼如“index is"后面的值即为host发下去的数据量
      grep -rn "has sent" ~/ascend/log/plog/plog-pid_timestamp0.log

      # 查看device侧队列的数据量，pid为训练任务的进程id，与上述host侧的进程id一致
      # 进行如下搜索，日志中关键字眼如“index=",等号后面的值即为device侧收到的数据量
      grep -rn "enqueue data" ~/ascend/log/device-id/device-pid_timestamp1.log
      ```

      - 如果host侧下发的与device侧收到的数据量相等，且该值小于网络正常训练完成的数据量，则数据下发失败主要为host侧数据处理异常导致供应不上网络训练，有三种可能的定位思路：
          - 如果数据量刚好为一个epoch中batch数的整数倍，则可能是数据处理部分涉及epoch的处理存在问题，如下面这场景：

          ```python
          ...
          dataset = dataset.create_tuple_iteator(num_epochs=-1) # 此处如果要返回一个迭代器则num_epochs应该给1, 但建议直接返回dataset
          return dataset
          ```

          - 数据处理性能较慢，跟不上网络训练的速度，针对这一场景，可借助profiler工具和MindInsight看一下是否存在明显的迭代间隙，或手动遍历一下dataset，并打印计算下平均单batch的耗时，是否比网络正反向加起来的时间更长，如果是则大概率需要对数据处理部分进行性能优化。
          - 训练过程中出现异常数据抛出异常导致下发数据失败，同常这种情况会有其他报错（ERROR）日志会提示数据处理哪个环节出现了异常及检查建议。如果不明显，也可以通过遍历dataset每条数据的方式尝试找出异常的数据（如关闭shuffle, 然后进行二分法）。
      - 如果host侧与device侧的数据量不相等（通常为host发的数据量更多）, 则可能为tdt模块存在一点问题（如反压等）需找模块开发人员协助定位。

  4. 如果**在训练结束后**打印这条日志（大抵是强制释放资源导致），可忽略这个报错。

  5. 如果仍不能定位具体原因，请开启mindspore和CANN的info级别日志，并检查日志看报错位置上下文寻找有帮助的信息，CANN host日志文件路径为：~/ascend/log/plog/plog-pid-timestamp.log

      ```bash
      export GLOG_v=1                  # set mindspore log level into info level
      export GLOBAL_ASCEND_LOG_LEVEL=1 # set CANN log level into info level
      ```
