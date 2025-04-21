# 数据处理

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/faq/data_processing.md)

## Q: 请问如果不使用高阶API，怎么实现数据下沉？

A: 可以参考此手动下沉方式的[test_tdt_data_transfer.py](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/tests/st/data_transfer/test_tdt_data_transfer.py)示例实现，不用借助`model.train`接口，目前支持：GPU和Ascend硬件使用。

<br/>

## Q: 在使用`Dataset`处理数据过程中内存占用高，怎么优化？

A: 可以参考如下几个步骤来降低内存占用，同时也可能会降低数据处理的效率。

1. 在定义数据集`**Dataset`对象前，设置`Dataset`数据处理预取的大小，`ds.config.set_prefetch_size(2)`。

2. 在定义`**Dataset`对象时，设置其参数`num_parallel_workers`为1。

3. 如果对`**Dataset`对象进一步使用了`.map(...)`操作，可以设置`.map(...)`的参数`num_parallel_workers`为1。

4. 如果对`**Dataset`对象进一步使用了`.batch(...)`操作，可以设置`.batch(...)`的参数`num_parallel_workers`为1。

5. 如果对`**Dataset`对象进一步使用了`.shuffle(...)`操作，可以把参数`buffer_size`设置减少。

<br/>

## Q: 在使用`Dataset`处理数据过程中CPU占用高，表现为sy占用高而us占用低，怎么优化？

A: 可以参考如下几个步骤来降低CPU占用，进一步提升性能，其主要原因是三方库多线程与数据处理多线程存在资源竞争。

1. 如果数据处理阶段有opencv的`cv2`操作，那么通过`cv2.setNumThreads(2)`设置`cv2`全局线程数。

2. 如果数据处理阶段有`numpy`操作，那么通过`export OPENBLAS_NUM_THREADS=1`设置`OPENBLAS`线程数。

3. 如果数据处理阶段有`numba`操作，那么通过`numba.set_num_threads(1)`设置并行度来减少线程竞争。

<br/>

## Q: 在`GeneratorDataset`中，看到有参数`shuffle`，在跑任务时发现`shuffle=True`和`shuffle=False`，两者没有区别，这是为什么？

A: 开启`shuffle`，需要传入的`Dataset`是支持随机访问的（例如自定义的`Dataset`有`getitem`方法），如果是在自定义的`Dataset`里面通过`yeild`方式返回回来的数据，是不支持随机访问的，具体可查看[GeneratorDataset 示例](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset/mindspore.dataset.GeneratorDataset.html)章节。

<br/>

## Q: 请问`Dataset`如何把两个`columns`合并成一个`column`？

A: 可以添加如下操作把两个字段合成一个。

```python
def combine(x, y):
    x = x.flatten()
    y = y.flatten()
    return np.append(x, y)

dataset = dataset.map(operations=combine, input_columns=["data", "data2"], output_columns=["data"])
```

注：因为两个`columns`是不同的`shape`，需要先`flatten`下，然后再合并。

<br/>

## Q: 请问`GeneratorDataset`支持`ds.PKSampler`采样吗？

A: 自定义数据集`GeneratorDataset`不支持`PKSampler`采样逻辑。主要原因是自定义数据操作灵活度太大了，内置的`PKSampler`难以做到通用性，所以选择在接口层面直接提示不支持。但是对于`GeneratorDataset`，可以方便的定义自己需要的`Sampler`逻辑，即在`ImageDataset`类的`__getitem__`函数中定义具体的`sampler`规则，返回自己需要的数据即可。

<br/>

## Q: MindSpore如何加载已有的预训练词向量？

A: 可以在定义EmbedingLookup或者Embedding时候，把预训练的词向量传进来，把预训练的词向量封装成一个Tensor作为EmbeddingLookup初始值。

<br/>

## Q: 请问`c_transforms`和`py_transforms`有什么区别，比较推荐使用哪个？

A: 推荐使用`c_transforms`，因为纯C层执行，所以性能会更好。

原理:`c_transform`底层使用的是C版本`opencv/jpeg-turbo`进行的数据处理，`py_transform`使用的是Python版本的`Pillow`进行数据处理。

在MindSpore1.8开始，数据增强API进行了合并，用户无需显式感知`c_transforms`和`py_transforms`，MindSpore将根据传入数据增强API的数据类型决定使用何种后端，默认使用`c_transforms`，因其性能更佳。详细可以参考[最新API文档与import说明](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/docs/api/api_python/mindspore.dataset.transforms.rst#视觉)。

<br/>

## Q: 由于我一条数据包含多个图像，并且每个图像的宽高都不一致，需要对转成mindrecord格式的数据进行`map`操作。可是我从`record`读取的数据是`np.ndarray`格式的数据，我的数据处理的`operations`是针对图像格式的。我应该怎么样才能对所生成的mindrecord的格式的数据进行预处理呢？

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

## Q: 我的自定义图像数据集转为mindrecord格式时，我的数据是`numpy.ndarray`格式的，且`shape`为[4,100,132,3]，这个`shape`的含义是四幅三通道的帧，且每个值都在0~255。可是当我查看转化成mindrecord的格式的数据时，发现是`[19800]`的`shape`，我原数据的维度全部展开有`[158400]`，请问这是为什么？

A: 可能是你数据中`ndarray`的`dtype`是`int8`，因为`[158400]`和`[19800]`刚好相差了8倍，建议将数据中`ndarray`的`dtype`指定为`float64`。

<br/>

## Q: 想要保存生成的图片，代码运行完毕以后在相应目录找不到图片。相似的，在JupyterLab中生成数据集用于训练，训练时可以在相应路径读取到数据，但是自己却无法在路径中找到图片或数据集？

A: 可能是JumperLab生成的图片或者数据集都是在Docker内，`moxing`下载的数据只能训练进程的Docker内看见，训练完成后这些数据就随着Docker释放了。可以试试在训练任务中将需要`download`的数据再通过`moxing`传回`obs`，然后再在`obs`里面下载到你本地。

<br/>

## Q: MindSpore中`model.train`的`dataset_sink_mode`参数该如何理解？

A: 当`dataset_sink_mode=True`时，数据处理会和网络计算构成Pipeline方式，即: 数据处理在逐步处理数据时，处理完一个`batch`的数据，会把数据放到一个队列里，这个队列用于缓存已经处理好的数据，然后网络计算从这个队列里面取数据用于训练，那么此时数据处理与网络计算就`Pipeline`起来了，整个训练耗时就是数据处理/网络计算耗时最长的那个。

当`dataset_sink_mode=False`时，数据处理会和网络计算构成串行的过程，即: 数据处理在处理完一个`batch`后，把这个`batch`的数据传递给网络用于计算，在计算完成后，数据处理再处理下一个`batch`，然后把这个新的`batch`数据传递给网络用于计算，如此的循环往复，直到训练完。该方法的总耗时是数据处理的耗时+网络计算的耗时=训练总耗时。

<br/>

## Q: MindSpore能否支持按批次对不同尺寸的图片数据进行训练？

A: 你可以参考yolov3对于此场景的使用，里面有对于图像的不同缩放，脚本见[yolo_dataset](https://gitee.com/mindspore/models/blob/master/official/cv/YOLOv3/src/yolo_dataset.py)。

<br/>

## Q: 使用MindSpore做分割训练，必须将数据转为MindRecord吗？

A: [build_seg_data.py](https://gitee.com/mindspore/models/blob/master/research/cv/FCN8s/src/data/build_seg_data.py)是将数据集生成MindRecord的脚本，可以直接使用/适配下你的数据集。或者如果你想尝试自己实现数据集的读取，可以使用`GeneratorDataset`自定义数据集加载。

[GeneratorDataset 示例](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset/mindspore.dataset.GeneratorDataset.html)

<br/>

## Q: MindSpore在Ascend硬件平台进行多卡训练，自定义数据集如何给不同卡传递不同数据？

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

## Q: 如何构建图像的多标签MindRecord格式数据集？

A: 数据Schema可以按如下方式定义: `cv_schema_json = {"label": {"type": "int32", "shape": [-1]}, "data": {"type": "bytes"}}`

说明: label是一个数组，numpy类型，这里面可以存 1，1，0，1，0，1 这么多label值，这些label值对应同一个data，即: 同一个图像的二进制值。
可以参考[将数据集转换为MindRecord](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/dataset/record.html#转换成record格式)教程。

<br/>

## Q: 请问自己制作的黑底白字`28*28`的数字图片，使用MindSpore训练出来的模型做预测，报错提示`wrong shape of image`是怎么回事？

A: 首先MindSpore训练使用的灰度图MNIST数据集。所以模型使用时对数据是有要求的，需要设置为`28*28`的灰度图，就是单通道才可以。

<br/>

## Q: MindSpore设计了专门用于数据处理的框架，有相关的设计和用法介绍？

A: MindSpore Dataset模块使得用户很简便地定义数据预处理Pipeline，并以高效（多进程/多线程）的方式处理数据集中样本，同时MindSpore Dataset也提供了多样化的API加载和处理数据集，详细介绍请参阅[数据处理Pipeline介绍](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mindspore.dataset.loading.html#%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86pipeline%E4%BB%8B%E7%BB%8D)。如果想进一步对数据处理Pipeline进行性能调优，请参阅[数据处理性能优化](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/dataset/optimize.html)。

<br/>

## Q: 网络训练时出现报错提示数据下发失败“TDT Push data into device Failed”，如何定位原因？

A: 首先上述报错指的是通过训练数据下发通道（TDT，train data transfer）发送数据到卡（device）上失败，导致这一报错的原因可能有多种，因此日志中给出了相应的检查建议，具体而言:

1. 通常我们会找到日志中最先抛出的错误（第一个ERROR级别的错误）或报错堆栈（TraceBack），并尝试从中找到有助于定位错误原因的信息。

2. **在图编译阶段，训练还没开始报错时**（例如日志中还没打印loss），请先检查下报错（ERROR）日志中是否有网络中涉及的相关算子报错或涉及环境没配置好导致的报错（如hccl.json不对导致多卡通信初始化异常）。

3. **在中间训练过程中报错时**，通常为下发的数据量（batch数）与网络训练需要的数据量（step数）不匹配导致的，可以通过`get_dataset_size`接口打印一个epoch中包含的batch数，导致异常的部分可能原因如下：

    - 通过查看打印loss次数的等方式判断如果数据量（step数）刚好为一个epoch中batch数的整数倍，则可能是数据处理部分涉及epoch的处理存在问题，如下面这场景:

        ```python
        ...
        dataset = dataset.create_tuple_iteator(num_epochs=-1) # 此处如果要返回一个迭代器则num_epochs应该给1，但建议直接返回dataset
        return dataset
        ```

    - 考虑是否是数据处理性能较慢，跟不上网络训练的速度，针对这一场景，可借助profiler工具看一下是否存在明显的迭代间隙，或手动遍历一下dataset，并打印计算下平均单batch的耗时，是否比网络正反向加起来的时间更长，如果是则大概率需要对数据处理部分进行性能优化。

    - 训练过程中出现异常数据抛出异常导致下发数据失败，通常这种情况会有其他报错（ERROR）日志会提示数据处理哪个环节出现了异常及检查建议。如果不明显，也可以通过遍历dataset每条数据的方式尝试找出异常的数据（如关闭shuffle, 然后进行二分法）。

4. 如果**在训练结束后**打印这条日志（大抵是强制释放资源导致），可忽略这个报错。

5. 如果仍不能定位具体原因，请通过提issue或论坛提问等方式找模块开发人员协助定位。

<br/>

## Q: py_transforms 和 c_transforms 增强操作能否混合使用，如果混合使用具体需要怎么使用？

A: 出于高性能考虑，通常不建议将py_transforms 与 c_transforms增强操作混合使用，但若不追求极致的性能，主要考虑打通流程，在无法全部使用c_transforms增强模块（缺少对应的c_transforms增强操作）的情况下，可使用py_transforms模块中的增强操作替代，此时即存在混合使用。
对此我们需要注意c_transforms 增强模块的输出通常是numpy array，py_transforms增强模块的输出是PIL Image，具体可查看对应的模块说明，为此通常的混合使用方法为：

- c_transforms 增强操作 + ToPIL操作 + py_transforms 增强操作 + ToNumpy操作
- py_transforms 增强操作 + ToNumpy操作 + c_transforms 增强操作

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

在MindSpore1.8之后，由于数据增强API的合并，写作上会更简洁，如：

```python
import mindspore.vision as vision

transforms = [
    vision.Decode(),         # c_transforms 数据增强
    vision.ToPIL(),          # 切换下一个增强输入为PIL
    vision.CenterCrop(375),  # py_transforms 数据增强
]

data1 = data1.map(operations=transforms, input_columns=["image"])
```

<br/>

## Q: 当错误提示 "The data pipeline is not a tree (i.e., one node has 2 consumers)" 应该怎么检查？

A: 上述错误通常是脚本书写错误导致。正常情况下数据处理pipeline中的操作是依次串联的，如下列定义：

```python
# pipeline结构：
# dataset1 -> map -> shuffle -> batch
dataset1 = XXDataset()
dataset1 = dataset1.map(...)
dataset1 = dataset1.shuffle(...)
dataset1 = dataset1.batch(...)
```

然而在下列异常场景中，假如dataset1有两个分叉节点，即dataset2和dataset3，就会出现上述错误。
因为dataset1节点产生了分支，其数据流向是未定义的，所以不允许出现此种情况。

```python
# pipeline结构：
# dataset1 -> dataset2 -> map
#          |
#          --> dataset3 -> map
dataset1 = XXDataset()
dataset2 = dataset1.map(***)
dataset3 = dataset1.map(***)
```

正确的写法如下所示，dataset3是由dataset2进性数据增强得到的，而不是在dataset1基础上进行数据增强操作得到。

```python
dataset2 = dataset1.map(***)
dataset3 = dataset2.map(***)
```

<br/>

## Q: MindSpore中和DataLoader对应的接口是什么？

A：如果将DataLoader考虑为接收自定义Dataset的API接口，MindSpore数据处理API中和Dataloader较为相似的是GeneratorDataset，可接收用户自定义的Dataset，具体使用方式参考[GeneratorDataset 示例](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset/mindspore.dataset.GeneratorDataset.html)，差异对比也可查看[API算子映射表](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/note/api_mapping/pytorch_api_mapping.html)。

<br/>

## Q: 自定义的Dataset出现错误时，应该如何调试？

A：自定义的Dataset通常会传入到GeneratorDataset，在使用过程中错误指向了自定义的Dataset时，可通过一些方式进行调试（如增加打印信息，打印返回值的shape、dtype等），自定义Dataset通常要保持中间处理结果为numpy array，且不建议与MindSpore网络计算的算子混合使用。此外针对自定义的Dataset如下面的MyDataset，初始化后也可直接进行如下遍历（主要为简化调试，分析原始Dataset中的问题，可不传入GeneratorDataset），调试遵循常规的Python语法规则。

```python
Dataset = MyDataset()
for item in Dataset:
   print("item:", item)
```

<br/>

## Q: 数据处理操作与网络计算算子能否混合使用？

A：通常数据处理操作与网络计算算子混合使用会导致性能有所降低，在缺少对应的数据处理操作且自定义Python操作不合适时可进行尝试。需要注意的是，因为二者需要的输入不一致，数据处理操作通常输入为numpy array 或 PIL Image，但网络计算算子输入需要是MindSpore.Tensor;
将二者混合使用需要使上一个的输出格式和下一个所需的输入格式一致。数据处理操作指的是官网API文档中mindspore.dataset模块下的接口，如 mindspore.dataset.vision.CenterCrop，网络计算算子包含 mindspore.nn、 mindspore.ops等模块下的算子。

<br/>

## Q: MindRecord为何会生成.db文件？ 缺少.db文件时加载数据集会有什么报错？

A：.db文件为MindRecord文件对应的索引文件，缺少.db文件通常会在获取数据集总的数据量时报错，错误提示如：`MindRecordOp Count total rows failed`。

<br/>

## Q: 自定义Dataset中如何进行图像读取并进行Decode操作？

A：传入GeneratorDataset的自定义Dataset，在接口内部（如`__getitem__`函数）进行图像读取后可以直接返回bytes类型的数据、numpy array类型的数组或已经做了解码操作的numpy array，具体如下所示：

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
    # 依据上面的用例，对__getitem__函数可进行如下修改，直接返回Decode之后的数据，此后可以不需要通过map执行Decode操作
    def __getitem__(self, index):
        # use Image.Open to open file, and convert to RGC
        img_rgb = Image.Open(self.data[index]).convert("RGB")
        return (img_rgb, )
    ```

<br/>

## Q: 在使用`Dataset`处理数据过程中，报错`RuntimeError: can't start new thread`，怎么解决？

A: 主要原因是在使用`**Dataset`、`.map(...)`和`.batch(...)`时，参数`num_parallel_workers`配置过大，用户进程数达到最大，可以通过`ulimit -u 最大进程数`来增加用户最大进程数范围，或者将`num_parallel_workers`配置减小。

<br/>

## Q: 在使用`GeneratorDataset`加载数据时，报错`RuntimeError: Failed to copy data into tensor.`，怎么解决？

A: 在使用`GeneratorDataset`加载Pyfunc返回的Numpy array时，MindSpore框架将执行Numpy array到MindSpore Tensor的转换，假设Numpy array所指向的内存被释放，可能会发生内存拷贝的错误。举例如下：

- 在`__getitem__`函数中执行Numpy array - MindSpore Tensor - Numpy array的就地转换。其中Tensor `tensor`和Numpy array `ndarray_1`共享同一块内存，Tensor `tensor`在`__getitem__`函数退出时超出作用域，其所指向的内存将被释放。

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

- 忽略上面例子中的循环转换，在`__getitem__`函数退出时，Tensor对象`tensor`被释放，和其共享同一块内存的Numpy array对象`ndarray_1`变成未知状态，为了规避此问题可以直接使用`deepcopy`函数为将返回的Numpy array对象`ndarray_2`申请独立的内存。

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

## Q: 如何根据数据预处理退出状态判断GetNext超时原因？

A: 在使用数据下沉模式（此时 `数据预处理` -> `发送队列` -> `网络计算` 三者构成Pipeline模式）进行训练时，当出现GetNext超时报错，数据预处理模块会输出状态信息，帮助用户分析出错原因。用户可以通过环境变量 `export MS_SUBMODULE_LOG_v={MD:1}` 来开启日志输出。其中： `channel_name` 代表host侧向设备侧发送数据通道的名称， `have_sent` 代表已经向设备发送的数据总条数， `host_queue` 代表最近10次dataset host侧队列的大小， `device_queue` 代表最近10次设备侧队列大小， `push_first_start_time` 和 `push_first_end_time` 代表host侧向设备发送第一条数据的起始时间， `push_start_time` 和 `push_end_time` 代表最近10次host侧向设备侧发送数据的起始时间。用户可以在日志中看到如下几种情况，具体原因及改进方法可参考：

1. 当日志输出类似如下时，表示数据预处理没有产生任何可用于训练的数据。

    ```
    channel_name: 29475464-f51b-11ee-b72b-8feb6783b0c3
    have_sent: 0;
    host_queue: ;
    device_queue: ;
          push_first_start_time -> push_first_end_time
                             -1 -> -1
                push_start_time -> push_end_time
    ```

    改进方法：可以先循环数据集对象，确认数据集预处理是否正常。

2. 当日志输出类似如下时，表示数据预处理产生了一条数据，但是仍未发送到设备侧。

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

    改进方法：可以查看设备plog是否有报错信息。

3. 当日志输出类似如下时，表示数据预处理产生了三条数据，并且都已经发送到设备侧，同时正在预处理第4条数据。

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

    改进方法：查看最后一条 `push_end_time` 时间与GetNext报错时间，如果超过默认GetNext超时时间（默认：1900s，且可通过 `mindspore.device_context.ascend.op_debug.execute_timeout(xx)`来进行修改），说明数据预处理性能差，可参考 [数据处理性能优化](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/dataset/optimize.html) 对数据预处理部分进行优化。

4. 当日志输出类似如下时，表示数据预处理产生了182条数据，正在向设备发送第183条数据，并且 `device_queue` 显示设备侧有充足的数据缓存。

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

    改进方法：可以查看设备plog是否有报错信息。

5. 当日志输出类似如下时，`device_queue` 出现很多个0，表示数据预处理太慢，会导致网络训练变慢。

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

    改进方法：可参考 [数据处理性能优化](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/dataset/optimize.html) 对数据预处理部分进行优化。

<br/>

## Q: 数据处理阶段报错 `Malloc device memory failed, free memory size is less than half of total memory size.Device 0 Device MOC total size:65464696832 Device MOC free size:3596279808 may be other processes occupying this card, ...` 怎么办？

A：通常是使用了自定义数据增强操作（其中包含了基于Ascend的数据增强操作）且使用了多进程模式，导致多进程使用同一个卡资源出现设备内存不足。

报错信息如下：

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

可以通过如下方式解决，在自定义函数中设置 `ms.runtime.set_memory(max_size="2GB")` 减少多进程的设备内存占用。

出错的脚本如下：

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

修复的脚本如下：

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

## Q: GeneratorDataset和map在哪些场景下支持调用dvpp算子？

A: 对于GeneratorDataset和map来说：

<table>
    <tr>
        <td rowspan="2"></td>
        <td rowspan="2" style="text-align: center">多线程</td>
        <td colspan="2" style="text-align: center">多进程</td>
    </tr>
    <tr>
        <td style="text-align: center">spawn</td>
        <td style="text-align: center">fork</td>
    </tr>
    <tr>
        <td>独立进程</td>
        <td>数据处理：支持<br>数据处理 + 网络训练：不支持</td>
        <td>数据处理：支持<br>数据处理 + 网络训练：支持</td>
        <td>数据处理：支持<br>数据处理 + 网络训练：不支持</td>
    </tr>
    <tr>
        <td>非独立进程</td>
        <td>数据处理：支持<br>数据处理 + 网络训练：支持</td>
        <td>数据处理：支持<br>数据处理 + 网络训练：支持</td>
        <td>数据处理：支持<br>数据处理 + 网络训练：不支持</td>
    </tr>
</table>

不支持场景说明：可能会出现scoped acquire::dec_ref(): internal error、nullptr、coredump、out of memory、卡住等报错行为。

1. 不支持独立进程（其中使用多线程的方式执行数据处理）下执行数据处理 + 网络训练：因为独立进程以fork方式创建出来，同时运行网络时，会在主进程中先设置device，那么fork出来的dataset独立进程中不能重新设置device，创建流会失败。

    部分报错信息如下：

    ```text
    terminate called after throwing an instance of 'std::runtime_error'
      what():  scoped acquire::dec_ref(): internal error:
    Fatal Python error: Aborted

    Current thread 0x0000fffd90b18120 (most recent call first):
    <no Python frame>
    ```

2. 不支持fork模式启动多进程的方式执行数据处理 + 网络训练：因为fork方式创建出的数据处理子进程中调用dvpp操作时，在网络运行过程中可能会出现gil抢占的问题。

    部分报错信息如下：

    ```text
    Fatal Python error: Segmentation fault

    Thread 0x0000fffef36cd120 (most recent call first):
    File "/opt/buildtools/python-3.9.11/lib/python3.9/threading.py", line 312 in wait
    File "/opt/buildtools/python-3.9.11/lib/python3.9/multiprocessing/queue.py", line 233 in _feed
    File "/opt/buildtools/python-3.9.11/lib/python3.9/threading.py", line 910 in run
    File "/opt/buildtools/python-3.9.11/lib/python3.9/threading.py", line 973 in _bootstrap_inner
    File "/opt/buildtools/python-3.9.11/lib/python3.9/threading.py", line 930 in _bootstrap
    ```

建议：替换成上面已支持的场景。

<br/>
