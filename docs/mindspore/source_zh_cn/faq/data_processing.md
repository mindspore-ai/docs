# 数据处理

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/faq/data_processing.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

<font size=3>**Q: 请问如果不使用高阶API，怎么实现数据下沉？**</font>

A: 可以参考此手动下沉方式的[test_tdt_data_transfer.py](https://gitee.com/mindspore/mindspore/blob/r1.7/tests/st/data_transfer/test_tdt_data_transfer.py)示例实现，不用借助`model.train`接口，目前支持：GPU和Ascend硬件使用。

<br/>

<font size=3>**Q: 在使用`Dataset`处理数据过程中内存占用高，怎么优化？**</font>

A: 可以参考如下几个步骤来降低内存占用，同时也可能会降低数据处理的效率。

  1. 在定义数据集`**Dataset`对象前，设置`Dataset`数据处理预取的大小，`ds.config.set_prefetch_size(2)`。

  2. 在定义`**Dataset`对象时，设置其参数`num_parallel_workers`为1。

  3. 如果对`**Dataset`对象进一步使用了`.map(...)`操作，可以设置`.map(...)`的参数`num_parallel_workers`为1。

  4. 如果对`**Dataset`对象进一步使用了`.batch(...)`操作，可以设置`.batch(...)`的参数`num_parallel_workers`为1。

  5. 如果对`**Dataset`对象进一步使用了`.shuffle(...)`操作，可以把参数`buffer_size`设置减少。

<br/>

<font size=3>**Q: 在使用`Dataset`处理数据过程中CPU占用高，表现为sy占用高而us占用低，怎么优化？**</font>

A: 可以参考如下几个步骤来降低CPU占用，进一步提升性能，其主要原因是三方库多线程与数据处理多线程存在资源竞争。

  1. 如果数据处理阶段有opencv的`cv2`操作，那么通过`cv2.setNumThreads(2)`设置`cv2`全局线程数。

  2. 如果数据处理阶段有`numpy`操作，那么通过`export OPENBLAS_NUM_THREADS=1`设置`OPENBLAS`线程数。

<br/>

<font size=3>**Q: 在`GeneratorDataset`中，看到有参数`shuffle`，在跑任务时发现`shuffle=True`和`shuffle=False`，两者没有区别，这是为什么？**</font>

A: 开启`shuffle`,需要传入的`Dataset`是支持随机访问的（例如自定义的`Dataset`有`getitem`方法），如果是在自定义的`Dataset`里面通过`yeild`方式返回回来的数据，是不支持随机访问的，具体可查看教程中的[数据集加载](https://www.mindspore.cn/tutorials/zh-CN/r1.7/advanced/dataset.html#id5)章节。

<br/>

<font size=3>**Q: 请问`Dataset`如何把两个`columns`合并成一个`column`？**</font>

A: 可以添加如下操作把 两个字段合成一个。

```python
def combine(x, y):
    x = x.flatten()
    y = y.flatten()
    return np.append(x, y)

dataset = dataset.map(operations=combine, input_columns=["data", "data2"], output_columns=["data"])
```

注：因为两个`columns`是不同的`shape`，需要先`flatten`下，然后再合并。

<br/>

<font size=3>**Q: 请问`GeneratorDataset`支持`ds.PKSampler`采样吗？**</font>

A: 自定义数据集`GeneratorDataset`不支持`PKSampler`采样逻辑。主要原因是自定义数据操作灵活度太大了，内置的`PKSampler`难以做到通用性，所以选择在接口层面直接提示不支持。但是对于`GeneratorDataset`，可以方便的定义自己需要的`Sampler`逻辑，即在`ImageDataset`类的`__getitem__`函数中定义具体的`sampler`规则，返回自己需要的数据即可。

<br/>

<font size=3>**Q: MindSpore如何加载已有的预训练词向量？**</font>

A: 可以在定义EmbedingLookup或者Embedding时候，把预训练的词向量传进来，把预训练的词向量封装成一个Tensor作为EmbeddingLookup初始值。

<br/>

<font size=3>**Q: 请问`c_transforms`和`py_transforms`有什么区别，比较推荐使用哪个？**</font>

A: 推荐使用`c_transforms`，因为纯C层执行，所以性能会更好。

原理:`c_transform`底层使用的是C版本`opencv/jpeg-turbo`进行的数据处理，`py_transform`使用的是Python版本的`Pillow`进行数据处理。

<br/>

<font size=3>**Q: 由于我一条数据包含多个图像，并且每个图像的宽高都不一致，需要对转成mindrecord格式的数据进行`map`操作。可是我从`record`读取的数据是`np.ndarray`格式的数据，我的数据处理的`operations`是针对图像格式的。我应该怎么样才能对所生成的mindrecord的格式的数据进行预处理呢？**</font>

A: 建议你按照如下操作进行:

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

<font size=3>**Q: 我的自定义图像数据集转为mindrecord格式时，我的数据是`numpy.ndarray`格式的，且`shape`为[4,100,132,3]，这个`shape`的含义是四幅三通道的帧，且每个值都在0~255。可是当我查看转化成mindrecord的格式的数据时，发现是`[19800]`的`shape`，我原数据的维度全部展开有`[158400]`，请问这是为什么？**</font>

A: 可能是你数据中`ndarray`的`dtype`是`int8`，因为`[158400]`和`[19800]`刚好相差了8倍，建议将数据中`ndarray`的`dtype`指定为`float64`。

<br/>

<font size=3>**Q: 想要保存生成的图片，代码运行完毕以后在相应目录找不到图片。相似的，在JupyterLab中生成数据集用于训练，训练时可以在相应路径读取到数据，但是自己却无法在路径中找到图片或数据集？**</font>

A: 可能是JumperLab生成的图片或者数据集都是在Docker内，`moxing`下载的数据只能训练进程的Docker内看见，训练完成后这些数据就随着Docker释放了。 可以试试在训练任务中将需要`download`的数据再通过`moxing`传回`obs`，然后再在`obs`里面下载到你本地。

<br/>

<font size=3>**Q: MindSpore中`model.train`的`dataset_sink_mode`参数该如何理解？**</font>

A: 当`dataset_sink_mode=True`时，数据处理会和网络计算构成Pipeline方式，即: 数据处理在逐步处理数据时，处理完一个`batch`的数据，会把数据放到一个队列里，这个队列用于缓存已经处理好的数据，然后网络计算从这个队列里面取数据用于训练，那么此时数据处理与网络计算就`Pipeline`起来了，整个训练耗时就是数据处理/网络计算耗时最长的那个。

当`dataset_sink_mode=False`时，数据处理会和网络计算构成串行的过程，即: 数据处理在处理完一个`batch`后，把这个`batch`的数据传递给网络用于计算，在计算完成后，数据处理再处理下一个`batch`，然后把这个新的`batch`数据传递给网络用于计算，如此的循环往复，直到训练完。该方法的总耗时是数据处理的耗时+网络计算的耗时=训练总耗时。

<br/>

<font size=3>**Q: MindSpore能否支持按批次对不同尺寸的图片数据进行训练？**</font>

A: 你可以参考yolov3对于此场景的使用，里面有对于图像的不同缩放,脚本见[yolo_dataset](https://gitee.com/mindspore/models/blob/r1.7/official/cv/yolov3_darknet53/src/yolo_dataset.py)。

<br/>

<font size=3>**Q: 使用MindSpore做分割训练，必须将数据转为MindRecord吗？**</font>

A: [build_seg_data.py](https://gitee.com/mindspore/models/blob/r1.7/official/cv/deeplabv3/src/data/build_seg_data.py)是将数据集生成MindRecord的脚本，可以直接使用/适配下你的数据集。或者如果你想尝试自己实现数据集的读取，可以使用`GeneratorDataset`自定义数据集加载。

[GenratorDataset 示例](https://www.mindspore.cn/tutorials/zh-CN/r1.7/advanced/dataset.html#id5)

[GenratorDataset API说明](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset)

<br/>

<font size=3>**Q: MindSpore在Ascend硬件平台进行多卡训练，自定义数据集如何给不同卡传递不同数据？**</font>

A: 使用`GeneratorDataset`的时候，可以使用`num_shards=num_shards`,`shard_id=device_id`参数来控制不同卡读取哪个分片的数据，`__getitem__`和`__len__`按全量数据集处理即可。

举例:

```python
# 卡0:
ds.GeneratorDataset(..., num_shards=8, shard_id=0, ...)
# 卡1:
ds.GeneratorDataset(..., num_shards=8, shard_id=1, ...)
# 卡2:
ds.GeneratorDataset(..., num_shards=8, shard_id=2, ...)
...
# 卡7:
ds.GeneratorDataset(..., num_shards=8, shard_id=7, ...)
```

<br/>

<font size=3>**Q: 如何构建图像的多标签MindRecord格式数据集？**</font>

A: 数据Schema可以按如下方式定义: `cv_schema_json = {"label": {"type": "int32", "shape": [-1]}, "data": {"type": "bytes"}}`

说明: label是一个数组，numpy类型，这里面可以存 1， 1，0，1， 0， 1 这么多label值，这些label值对应同一个data，即: 同一个图像的二进制值。
可以参考[将数据集转换为MindRecord](https://www.mindspore.cn/tutorials/zh-CN/r1.7/advanced/dataset/record.html#将数据集转换为MindRecord)教程。

<br/>

<font size=3>**Q: 请问自己制作的黑底白字`28*28`的数字图片，使用MindSpore训练出来的模型做预测，报错提示`wrong shape of image`是怎么回事？**</font>

A: 首先MindSpore训练使用的灰度图MNIST数据集。所以模型使用时对数据是有要求的，需要设置为`28*28`的灰度图，就是单通道才可以。

<br/>

<font size=3>**Q: 第一次看到有专门的数据处理框架，能介绍下么？**</font>

A: MindData提供数据处理异构硬件加速功能，高并发数据处理`pipeline`同时支持`Ascend/GPU/CPU`，`CPU`占用降低30%，点击查询[优化数据处理](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.7/dataset/optimize.html)。

<br/>

<font size=3>**Q: 网络训练时出现报错提示数据下发失败“TDT Push data into device Failed”，如何定位原因？**</font>

A: 首先上述报错指的是通过训练数据下发通道（TDT，train data transfer)发送数据到卡（device）上失败，导致这一报错的原因可能有多种，因此日志中给出了相应的检查建议，具体而言:

  1. 通常我们会找到日志中最先抛出的错误（第一个ERROR级别的错误）或报错堆栈（TraceBack)，并尝试从中找到有助于定位错误原因的信息。

  2. **在图编译阶段，训练还没开始报错时**（例如日志中还没打印loss)，请先检查下报错（ERROR）日志中是否有网络中涉及的相关算子报错或涉及环境没配置好导致的报错（如hccl.json不对导致多卡通信初始化异常）。

  3. **在中间训练过程中报错时**，通常为下发的数据量（batch数）与网络训练需要的数据量（step数）不匹配导致的，可以通过`get_dataset_size`接口打印一个epoch中包含的batch数，导致异常的部分可能原因如下：

      - 通过查看打印loss次数的等方式判断如果数据量（step数）刚好为一个epoch中batch数的整数倍，则可能是数据处理部分涉及epoch的处理存在问题，如下面这场景:

          ```python
          ...
          dataset = dataset.create_tuple_iteator(num_epochs=-1) # 此处如果要返回一个迭代器则num_epochs应该给1, 但建议直接返回dataset
          return dataset
          ```

      - 考虑是否是数据处理性能较慢，跟不上网络训练的速度，针对这一场景，可借助profiler工具和MindInsight看一下是否存在明显的迭代间隙，或手动遍历一下dataset，并打印计算下平均单batch的耗时，是否比网络正反向加起来的时间更长，如果是则大概率需要对数据处理部分进行性能优化。

      - 训练过程中出现异常数据抛出异常导致下发数据失败，通常这种情况会有其他报错（ERROR）日志会提示数据处理哪个环节出现了异常及检查建议。如果不明显，也可以通过遍历dataset每条数据的方式尝试找出异常的数据（如关闭shuffle, 然后进行二分法）。

  4. 如果**在训练结束后**打印这条日志（大抵是强制释放资源导致），可忽略这个报错。

  5. 如果仍不能定位具体原因，请通过提issue或论坛提问等方式找模块开发人员协助定位。

<br/>

<font size=3>**Q: py_transforms 和 c_transforms 算子能否混合使用，如果混合使用具体需要怎么使用？**</font>

A: 出于高性能考虑，通常不建议将py_transforms 与 c_transforms算子混合使用，[文档](https://www.mindspore.cn/tutorials/zh-CN/r1.7/advanced/dataset.html#%E4%BD%BF%E7%94%A8%E6%B3%A8%E6%84%8F%E4%BA%8B%E9%A1%B9)也对此进行了说明。但若不追求极致的性能，主要考虑打通流程，在无法全部使用c_transforms算子（缺少对应的c_transforms算子）的情况下，可使用py_transforms算子替代，此时即存在混合使用。
对此我们需要注意c_transforms 算子的输出通常是numpy array，py_transforms算子的输出是PIL Image，具体可查看算子说明，为此通常的混合使用方法为：

- c_transforms 算子 + ToPIL 算子 + py_transforms 算子 + ToTensor算子
- py_transforms 算子 + ToTensor 算子 + c_transforms 算子

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

<font size=3>**Q: 当错误提示 "The data pipeline is not a tree (i.e., one node has 2 consumers)" 应该怎么检查？**</font>

A: 上述错误通常是脚本书写错误导致，具体发生在下面这种场景；正常情况下数据处理pipeline中的操作是依次串联的，下面的异常场景中dataset1有两个消费节点 dataset2和dataset3，就会出现上述错误。

```python
 dataset2 = dataset1.map(***)
 dataset3 = dataset1.map(***)
```

正确的写法如下所示，dataset3是由dataset2进性数据增强得到的，而不是在dataset1基础上进行数据增强操作得到。

```python
 dataset2 = dataset1.map(***)
 dataset3 = dataset2.map(***)
```

<br/>

<font size=3>**Q: MindSpore中和Dataloader对应的算子是什么？**</font>

A：如果将Dataloader考虑为接收自定义Dataset的API接口，MindSpore数据处理API中和Dataloader较为相似的是GeneratorDataset，可接收用户自定义的Dataset，具体使用方式参考[GeneratorDataset 文档](https://www.mindspore.cn/tutorials/zh-CN/r1.7/advanced/dataset.html#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86%E5%8A%A0%E8%BD%BD)，差异对比也可查看[API算子映射表](https://www.mindspore.cn/docs/zh-CN/r1.7/note/api_mapping/pytorch_api_mapping.html)。

<br/>

<font size=3>**Q: 自定义的Dataset出现错误时，应该如何调试？**</font>

A：自定义的Dataset通常会传入到GeneratorDataset，在使用过程中错误指向了自定义的Dataset时，可通过一些方式进行调试（如增加打印信息，打印返回值的shape、dtype等），自定义Dataset通常要保持中间处理结果为numpy array，且不建议与MindSpore网络计算的算子混合使用。此外针对自定义的Dataset如下面的MyDataset，初始化后也可直接进行如下遍历（主要为简化调试，分析原始Dataset中的问题，可不传入GeneratorDataset)，调试遵循常规的Python语法规则。

```python
Dataset = MyDataset()
for item in Dataset:
   print("item:", item)
```

<br/>

<font size=3>**Q: 数据处理算子与网络计算算子能否混合使用？**</font>

A：通常数据处理算子与网络计算算子混合使用会导致性能有所降低，在缺少对应的数据处理算子且自定义py_transforms算子不合适时可进行尝试。需要注意的是，因为二者需要的输入不一致，数据处理算子通常输入为numpy array 或 PIL Image，但网络计算算子输入需要是MindSpore.Tensor;
将二者混合使用需要使上一个算子的输出格式和下一个算子所需的输入格式一致。数据处理算子指的是官网API文档中mindspore.dataset开头的算子，如 mindspore.dataset.vision.c_transforms.CenterCrop，网络计算算子包含 mindspore.nn、 mindspore.ops等目录下的算子。

<br/>

<font size=3>**Q: MindRecord为何会生成.db文件？ 缺少.db文件时加载数据集会有什么报错？**</font>

A：.db文件为MindRecord文件对应的索引文件，缺少.db文件通常会在获取数据集总的数据量时报错，错误提示如：`MindRecordOp Count total rows failed`。

<br/>

<font size=3>**Q: 自定义Dataset中如何进行图像读取并进行Decode操作？**</font>

A：传入GeneratorDataset的自定义Dataset，在接口内部（如`__getitem__`函数）进行图像读取后可以直接返回bytes类型的数据、numpy array类型的数组或已经做了解码操作的numpy array, 具体如下所示：

- 读取图像后直接返回bytes类型的数据

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

- 读取图像后返回numpy array

    ```python
    # 在上面的用例中，对__getitem__函数可进行如下修改, Decode操作同上述用例一致
    def __getitem__(self, index):
        # use np.fromfile to read image
        img_np = np.fromfile(self.data[index])

        # return Numpy array directly
        return (img_np, )
    ```

- 读取图像后直接进行Decode操作

    ```python
    # 依据上面的用例，对__getitem__函数可进行如下修改, 直接返回Decode之后的数据，此后可以不需要通过map算子接Decode操作
    def __getitem__(self, index):
        # use Image.Open to open file, and convert to RGC
        img_rgb = Image.Open(self.data[index]).convert("RGB")
        return (img_rgb, )
    ```

<br/>

<font size=3>**Q: 在使用`Dataset`处理数据过程中，报错`RuntimeError: can't start new thread`，怎么解决？**</font>

A: 主要原因是在使用`**Dataset`、`.map(...)`和`.batch(...)`时，参数`num_parallel_workers`配置过大，用户进程数达到最大，可以通过`ulimit -u 最大进程数`来增加用户最大进程数范围，或者将`num_parallel_workers`配置减小。
