# 数据处理调试方法与常见问题分析

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_zh_cn/debug/error_analysis/minddata_debug.md)&nbsp;&nbsp;

## 数据处理调试方法

### 方法1：数据处理执行出错，添加打印或调试点到代码中调试

使用 `GeneratorDataset` 或 `map` 进行加载/处理数据时，可能会因为语法错误、计算溢出等问题导致数据报错，一般可以按如下步骤进行排查和调试：

1. 观察报错栈信息，根据报错栈信息大概定位到出错代码块。

2. 在出错的代码块附近添加打印或调试点，进一步调试。

以下展示一个存在语法/数值问题数据pipeline，并如何按照上述的方案修复报错。

```python
import mindspore.dataset as ds

class Loader:
    def __init__(self):
        self.data = [1, 6, 0, 1, 2]
        self.dividend = 1
    def __getitem__(self, index):
        a = self.dividend
        b = self.data[index]
        return a / b
    def __len__(self):
        return len(self.data)


dataloader = ds.GeneratorDataset(Loader(), column_names=["data"])
for data in dataloader:
    print("data", data)
```

运行后报错如下，可以观察到错误提示分为3块：

* Dataset Pipeline Error Message：错误综述，此处提示由于Python代码执行出错导致报错退出。
* Python Call Stack：Python代码的调用信息，展示产生Python异常前的调用栈。
* C++ Call Stack：C++代码的调用信息，用于框架开发者调试。

```text
------------------------------------------------------------------
- Python Call Stack:
------------------------------------------------------------------
Traceback (most recent call last):
  File "/.../mindspore/dataset/engine/datasets_user_defined.py", line 99, in _cpp_sampler_fn
    val = dataset[i]
  File "test_cv.py", line 11, in __getitem__
    return a / b
ZeroDivisionError: division by zero

------------------------------------------------------------------
- Dataset Pipeline Error Message:
------------------------------------------------------------------
[ERROR] Execute user Python code failed, check 'Python Call Stack' above.

------------------------------------------------------------------
- C++ Call Stack: (For framework developers)
------------------------------------------------------------------
mindspore/ccsrc/minddata/dataset/engine/datasetops/source/generator_op.cc(247).
```

Dataset Pipeline Error Message提示在运行用户的Python脚本出现异常，继续查看Python Call Stack。
根据Python栈信息，异常从 `__getitem__` 函数中抛出，并且提示了相关的代码在 `return a / b` 附近。因此，可以对日志中提示报错的代码附近添加打印或者调试点。

```python
import mindspore.dataset as ds

class Loader:
    def __init__(self):
        self.data = [1, 6, 0, 1, 2]
        self.dividend = 1

    def __getitem__(self, index):
        try:
            print(">>> debug: come into __getitem__", flush=True)
            a = self.dividend
            b = self.data[index]
            print(">>> debug: a is", a, flush=True)
            print(">>> debug: b is", b, flush=True)
            return a / b
        except Exception as e:
            print("exception occurred", str(e))
            import pdb
            pdb.set_trace()
            # do anything you want to check variable

    def __len__(self):
        return len(self.data)


dataloader = ds.GeneratorDataset(Loader(), column_names=["data"])
# Make the pipeline single-threaded before you run it
ds.config.set_num_parallel_workers(1)
for i, data in enumerate(dataloader):
    print("data count", i)
```

加入相关的调试信息重新运行数据pipeline，可以查看到捕获到了异常，并且进入了pdb的调试器。
这个时候就可以按照需要（遵循pdb的语法）打印查看相关的变量进行调试，找到报错的地方为 1/0 导致了除零错误。

```text
>>> debug: come into __getitem__
>>> debug: a is 1
>>> debug: b is 2
>>> debug: come into __getitem__
data count 0
>>> debug: a is 1
>>> debug: b is 0
exception occurred division by zero
--Return--
> /test_cv.py(19)__getitem__()->None
-> pdb.set_trace()
(Pdb)
```

### 方法2：数据增强map操作出错，调试map操作中各个数据处理算子

将数据增强变换嵌入到数据pipeline的 `map` 操作中时，有时候会导致报错后不容易调试。
以下例子展示了一个嵌入 `RandomResize` 和 `Crop` 增强到 `map` 操作中对数据进行裁剪的例子，但由于输入对象的shape经过变换后有误导致报错。

#### 方式一：通过单个算子执行的方式调试

```python
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

class MyDataset:
    def __init__(self):
        self.data = [np.ones((32, 32, 3)), np.ones((3, 48, 48))]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)


dataset = ds.GeneratorDataset(MyDataset(), column_names=["data"])
transforms_list = [vision.RandomResize((3, 16)), vision.Crop(coordinates=(0, 0), size=(8, 8))]
dataset = dataset.map(operations=transforms_list)
for data in dataset:
    print("data", data)
```

当执行上述示例时会得到如下报错，但根据报错提示，较难获取输入的对象是什么内容，以及shape是什么。

```text
------------------------------------------------------------------
- Dataset Pipeline Error Message:
------------------------------------------------------------------
[ERROR] map operation: [Crop] failed. Crop: Crop height dimension: 8 exceeds image height: 3.

------------------------------------------------------------------
- C++ Call Stack: (For framework developers)
------------------------------------------------------------------
mindspore/ccsrc/minddata/dataset/kernels/image/crop_op.cc(33).
```

从Dataset Pipeline Error Message的提示可以看到错误是由 `Crop` 在计算时抛出。因此可以稍微改写一下数据pipeline，对 `Crop` 的输入输出进行打印，并添加打印进行调试。

```python
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

class MyDataset:
    def __init__(self):
        self.data = [np.ones((32, 32, 3)), np.ones((3, 48, 48))]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def MyWrapper(data):
    transforms_list = [vision.RandomResize((3, 16)), vision.Crop(coordinates=(0, 0), size=(8, 8))]
    for transforms in transforms_list:
        print(">>> debug: apply transforms: ", type(transforms))
        print(">>> debug: before apply transforms, data shape", data.shape)
        data = transforms(data)
        print(">>> debug: after apply transforms, data shape", data.shape)
    return data


dataset = ds.GeneratorDataset(MyDataset(), column_names=["data"], shuffle=False)
dataset = dataset.map(MyWrapper)
ds.config.set_num_parallel_workers(1)
for data in dataset:
    print("data", data[0].shape)
```

再次运行得到以下相关内容:

```text
>>> debug: apply transforms:  <class 'mindspore.dataset.vision.transforms.RandomResize'>
>>> debug: before apply transforms, data shape (32, 32, 3)
>>> debug: after apply transforms, data shape (3, 16, 3)
>>> debug: apply transforms:  <class 'mindspore.dataset.vision.transforms.Crop'>
>>> debug: before apply transforms, data shape (3, 16, 3)

RuntimeError: Exception thrown from user defined Python function in dataset.

------------------------------------------------------------------
- Dataset Pipeline Error Message:
------------------------------------------------------------------
[ERROR] Crop: Crop height dimension: 8 exceeds image height: 3.

------------------------------------------------------------------
- C++ Call Stack: (For framework developers)
------------------------------------------------------------------
mindspore/ccsrc/minddata/dataset/kernels/image/crop_op.cc(33).
```

根据打印的信息可以看到 `Crop` 处理第一个样本时报错，第一个样本的shape(32, 32, 3)，被 `RandomResize` 变换为(3, 16, 3)，但是没有打印 `Crop` 变换后的shape就报错了。因此正是此时的shape不能被 `Crop` 处理导致错误发生。进一步根据Dataset Pipeline Error Message的提示，输入样本的高只有3，但是期望裁剪出高维8的区域，所以报错。

查看 `Crop` 的 [API说明](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/dataset_vision/mindspore.dataset.vision.Crop.html#mindspore.dataset.vision.Crop) ，`Crop` 要求输入样本的shape为 <H, W> 或 <H, W, C>，所以 `Crop` 会把(3, 16, 3)当成<H, W, C>，当H=3, W=16，C=3时自然裁剪不出H=8, W=8的区域。

为了快速修复此问题，我们只需要把 `RandomResize` 的参数size由原来的(3, 16)改为(16, 16)，再次执行就会发现用例通过。

```text
>>> debug: apply transforms:  <class 'mindspore.dataset.vision.transforms.RandomResize'>
>>> debug: before apply transforms, data shape (32, 32, 3)
>>> debug: after apply transforms, data shape (16, 16, 3)
>>> debug: apply transforms:  <class 'mindspore.dataset.vision.transforms.Crop'>
>>> debug: before apply transforms, data shape (16, 16, 3)
>>> debug: after apply transforms, data shape (8, 8, 3)
data (8, 8, 3)
>>> debug: apply transforms:  <class 'mindspore.dataset.vision.transforms.RandomResize'>
>>> debug: before apply transforms, data shape (3, 48, 48)
>>> debug: after apply transforms, data shape (16, 16, 48)
>>> debug: apply transforms:  <class 'mindspore.dataset.vision.transforms.Crop'>
>>> debug: before apply transforms, data shape (16, 16, 48)
>>> debug: after apply transforms, data shape (8, 8, 48)
data (8, 8, 48)
```

#### 方式二：通过数据管道调试模式调试map操作

我们还可以调用 [set_debug_mode](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/dataset/mindspore.dataset.config.set_debug_mode.html) 方法开启数据集管道调试模式来进行调试。
当启用调试模式时，如果随机种子没有被设置，则会将随机种子设置为1，以便在调试模式下执行数据集管道可以获得确定性的结果。

流程如下:

1. 在 `map` 算子中打印每个变换op的输入输出数据的形状和类型。
2. 启用数据集管道调试模式，并使用MindData提供的预定义调试钩子或者用户定义的调试钩子，它必须定义继承自 [DebugHook](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/dataset/mindspore.dataset.debug.DebugHook.html) 类。

以下是在 `方式一` 的用例上做修改，使用MindData提供的预定义调试钩子。

```python
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.debug as debug
import mindspore.dataset.vision as vision

class MyDataset:
    def __init__(self):
        self.data = [np.ones((32, 32, 3)), np.ones((3, 48, 48))]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Enable dataset pipeline debug mode and use pre-defined debug hook provided by MindData.
ds.config.set_debug_mode(True)

# Define dataset pipeline
dataset = ds.GeneratorDataset(MyDataset(), column_names=["data"])

transforms_list = [vision.RandomResize((3, 16)), vision.Crop(coordinates=(0, 0), size=(8, 8))]
dataset = dataset.map(operations=transforms_list)
for i, data in enumerate(dataset):
    print("data count", i)
```

运行得到以下相关内容：

```text
[Dataset debugger] Print the [INPUT] of the operation [RandomResize].
Column 0. The dtype is [float64]. The shape is [(32, 32, 3)].
[Dataset debugger] Print the [OUTPUT] of the operation [RandomResize].
Column 0. The dtype is [float64]. The shape is [(3, 16, 3)].
    ......
E           RuntimeError: Exception thrown from dataset pipeline. Refer to 'Dataset Pipeline Error Message'.
E
E           ------------------------------------------------------------------
E           - Dataset Pipeline Error Message:
E           ------------------------------------------------------------------
E           [ERROR] map operation: [Crop] failed. Crop: Crop height dimension: 8 exceeds image height: 3.
E
E           ------------------------------------------------------------------
E           - C++ Call Stack: (For framework developers)
E           ------------------------------------------------------------------
E           mindspore/ccsrc/minddata/dataset/kernels/image/crop_op.cc(33).
```

根据打印的信息我们就能很清楚的知道 `Crop` 在处理输入shape为(3, 16, 3)的时候出现了报错，同样查看 `Crop` 的 [API说明](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/dataset_vision/mindspore.dataset.vision.Crop.html#mindspore.dataset.vision.Crop)。我们只需要把 `RandomResize` 的参数size由原来的(3, 16)改为(16, 16)，再次执行就会发现用例通过。

```text
[Dataset debugger] Print the [INPUT] of the operation [RandomResize].
Column 0. The dtype is [float64]. The shape is [(32, 32, 3)].
[Dataset debugger] Print the [OUTPUT] of the operation [RandomResize].
Column 0. The dtype is [float64]. The shape is [(16, 16, 3)].
[Dataset debugger] Print the [OUTPUT] of the operation [Crop].
Column 0. The dtype is [float64]. The shape is [(8, 8, 3)].
******data count 0
[Dataset debugger] Print the [INPUT] of the operation [RandomResize].
Column 0. The dtype is [float64]. The shape is [(3, 48, 48)].
[Dataset debugger] Print the [OUTPUT] of the operation [RandomResize].
Column 0. The dtype is [float64]. The shape is [(16, 16, 48)].
[Dataset debugger] Print the [OUTPUT] of the operation [Crop].
Column 0. The dtype is [float64]. The shape is [(8, 8, 48)].
******data count 1
```

另外还可以使用自定义的调试钩子手动插入，在 `MyHook` 类的 `compute` 函数中添加断点，可以打印日志查看数据的类型和形状等。

```python
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.debug as debug
import mindspore.dataset.vision as vision

# Enable dataset pipeline debug mode and use user-defined debug hook. It must define a
# class inherited from DebugHook.
class MyHook(debug.DebugHook):
    def __init__(self):
        super().__init__()

    def compute(self, *args):
        print("come into my hook function, block with pdb", flush=True)
        import pdb
        print("the input shape is: ", args[0][0].shape, flush=True)
        pdb.set_trace()
        return args


class MyDataset:
    def __init__(self):
        self.data = [np.ones((32, 32, 3)), np.ones((3, 48, 48))]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Enable dataset pipeline debug mode and use pre-defined debug hook provided by MindData.
ds.config.set_debug_mode(True, debug_hook_list=[MyHook()])

# Define dataset pipeline.
dataset = ds.GeneratorDataset(MyDataset(), column_names=["data"])

# Insert debug hook before `Crop` operation.
transforms_list = [vision.RandomResize((3, 16)), MyHook(), vision.Crop(coordinates=(0, 0), size=(8, 8))]
dataset = dataset.map(operations=transforms_list)
for i, data in enumerate(dataset):
    print("data count", i)
```

同上可知，可以通过一步步的查看输入的shape来定位问题，接下来就可以开始你的调试了：

```text
[Dataset debugger] Print the [INPUT] of the operation [RandomResize].
come into my hook function, block with pdb
the input shape is:  (3, 48, 48)

>>>>>>>>>>>>>>>>>>>>>PDB set_trace>>>>>>>>>>>>>>>>>>>>>
> /test_demo.py(18)compute
-> return args
(Pdb)
```

### 方法3：测试数据处理的性能

当使用MindSpore启动训练，训练日志一直打印，出现了很多条，很可能是数据处理较慢的问题。

```text
[WARNING] MD(90635,fffdf0ff91e0,python):2023-03-25-15:29:14.801.601 [mindspore/ccsrc/minddata/dataset/engine/datasetops/source/generator_op.cc:220] operator()] Bad performance attention,
it takes more than 25 seconds to generator.__next__ new row, which might cause `GetNext` timeout problem when sink_mode=True.
You can increase the parameter num_parallel_workers in GeneratorDataset / optimize the efficiency of obtaining samples in the user-defined generator function.
```

下面介绍一种调试数据集性能的方法，即使没有出现上述的WARNING信息，也可以调试数据性能，作为参考：
构造一个简单的lenet训练网络，简单修改一点代码，使运行结果出现WARNING信息。

```python
import time
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore.dataset.vision import Inter

def create_dataset(data_path, num_parallel_workers=1):
    mnist = ds.MnistDataset(data_path, num_samples=1000, shuffle=False)

    class udf:
        def __init__(self, dataset):
            self.dataset = dataset
            self.cnt = 0

            self.iterator = self.dataset.create_tuple_iterator(num_epochs=1)
            self.data = []
            for i in range(1000):
                self.data.append(self.iterator.__next__())

        def __len__(self):
            return 1000

        def __getitem__(self, index):
            if index >= 7:
                time.sleep(60)
            return self.data[index]


    mnist_ds = ds.GeneratorDataset(udf(mnist), ["image", "label"])

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml * rescale, shift_nml)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # use map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label")
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image")
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image")
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image")
    mnist_ds = mnist_ds.batch(4, drop_remainder=True)
    return mnist_ds


class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


dataset_train = create_dataset("mnist/train")

ms.set_context(mode=ms.GRAPH_MODE)
network = LeNet5(num_class=10)
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
model = ms.Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'accuracy'})
model.train(10, dataset_train, callbacks=[ms.LossMonitor()])
```

在训练的时候，我们会获得非常多warning提示数据集性能较慢，但是观察到有Epoch time，per step time信息，因此训练其实也在进行，只是较慢。

```text
[WARNING] MD(90635,fffdf0ff91e0,python):2023-03-25-15:29:14.801.601 [mindspore/ccsrc/minddata/dataset/engine/datasetops/source/generator_op.cc:220] operator()] Bad performance attention, it takes more than 25 seconds to generator.__next__ new row, which might cause `GetNext` timeout problem when sink_mode=True. You can increase the parameter num_parallel_workers in GeneratorDataset / optimize the efficiency of obtaining samples in the user-defined generator function.
[WARNING] MD(90635,fffd72ffd1e0,python):2023-03-25-15:29:14.802.398 [mindspore/ccsrc/minddata/dataset/engine/datasetops/data_queue_op.cc:903] DetectPerBatchTime] Bad performance attention, it takes more than 25 seconds to fetch a batch of data from dataset pipeline, which might result `GetNext` timeout problem. You may test dataset processing performance(with creating dataset iterator) and optimize it.
Epoch time: 60059.685 ms, per step time: 30029.843 ms, avg loss: 2.301
```

此时，可以单独迭代数据集，查看每条数据的处理时间，以此判断数据集的性能如何：
在上述代码的 `dataset_train = create_dataset("mnist/train")` 后面，可以加入以下代码用于调试数据集：

```python
import time

st = time.time()
for i, data in enumerate(dataset_train):
    print("data step", i, ", time", time.time() - st, flush=True)
    st = time.time()
    if i > 50:
        break
```

加入代码后再次运行，将会看到数据集的处理时间：

```text
data step 0 , time 0.0055468082427978516
data step 1 , time 60.034635634525
data step 2 , time 480.046234134121
data step 3 , time 480.023415324343
data step 4 , time 480.051423635473
```

可以看到，从第2条数据开始，每一条数据都要等到60s以上才处理完成, 对于上述“修改过的代码”其实是好解决的，检查一下代码会发现：

```python
def __getitem__(self, index):
    if index >= 7:
        time.sleep(60)
    return self.data[index]

```

从第7条数据开始，每一条都会sleep60秒才会输出，也正是这里导致了数据处理变慢。
由于batch size是4，所以第一个batch只包含前4条数据（0,1,2,3），自然处理时间没有问题，然后到第二个batch，由于包含（4,5,6,7）4条数据，所以在第7条的时候会额外等待60s才会输出，从而导致整体在第2个batch的时候，数据时间延长到了60s，后面第三个第四个batch同理。所以只需要把sleep的逻辑去掉，即可把数据处理拉回到正常的水平。

在真实训练场景中，也会有不同的原因导致网络训练变慢，但是分析方法也是类似的。我们可以先单独迭代数据，以定界是否为数据处理慢导致训练性能较低。

### 方法4：检查数据处理中的异常数据

在对数据进行处理的过程中，可能会因为计算错误、数值溢出等因素，产生了异常的结果数值，从而导致训练网络时算子计算溢出、权重更新异常等问题。此方案介绍如何调试和检查异常的数据行为/数据结果。

#### 关闭混洗，固定随机种子，确保可重现性

在一些数据处理的场景中，我们会使用随机的函数，作为数据运算的一部分。由于随机运算本身的特性，每一次运行的数据结果都不相同，这样很有可能会出现上一次运行的结果中存在异常的数值，但在下一次运行的时候，却没有检查到异常的数值，那么很有可能是因为随机索引/随机计算的影响。这种情况下，可以关闭数据集的混洗选项，并固定不同的随机种子，通过多次运行寻找可能引入的随机问题。

以下例子把一个随机值当做是一个除数，在偶然的情况下会出现除0的情况。

```python
import numpy as np
import mindspore as ms

class Gen():
    def __init__(self):
        self.data = [np.array(i) / np.random.randint(0, 3) for i in range(1, 4)]
    def __getitem__(self, index):
        data = self.data[index]
        return data
    def __len__(self):
        return len(self.data)


dataset = ms.dataset.GeneratorDataset(Gen(), ["data"])
for data in dataset:
    print(data)
```

通过 `set_seed` 设置随机种子，产生固定的随机数来达到确定的结果，可以进一步排查代码中随机运算是否符合预期。

```python
ms.set_seed(1)
ms.dataset.GeneratorDataset(Loader(), ["data"], shuffle=False)
```

多次运行结果保持一致，可以看到第1条数据和第3条数据出现的除零的结果。间接可以说明，在第1条和第3条数据的计算上存在异常导致出现了inf的数值。

```text
[Tensor(shape=[], dtype=Float64, value= inf)]
[Tensor(shape=[], dtype=Float64, value= 1)]
[Tensor(shape=[], dtype=Float64, value= inf)]
```

#### 利用NumPy等工具快速校验结果

上一个例子中的数据量较少，基本上可以通过检查代码发现出现异常数值的位置。对于一些大型的高维数组，代码检查或者打印数值就不太方便了。这个时候，可以配置MindSpoer的数据集以NumPy的形式返回数据，并借助NumPy的一些常用检查数组内容的手段去检查数组中是否存在异常数值。

以下例子构造了一个大型的高维数组，并对其中的数值进行随机的运算。

```python
import numpy as np
import mindspore as ms

class Gen():
    def __init__(self):
        self.data = np.random.randint(0, 255, size=(16, 50, 50))
    def __getitem__(self, index):
        data = self.data[index] / np.random.randint(0, 2)
        return data
    def __len__(self):
        return 16


dataset = ms.dataset.GeneratorDataset(Gen(), ["data"])
for data in dataset:
    print(data)
```

为了检查在数据运算时存在异常数值如nan、inf等，可以在遍历数据集对象时，指定其输出为NumPy类型。

指定了输出类型后，打印的data对象中各个元素均为NumPy类型，基于此可以采用NumPy中的一些非常方便的函数校验其中的数值是否异常

```python
for data_index, data in enumerate(dataset.create_tuple_iterator(output_numpy=True)):
    if(np.isinf(data).any()):             # Checking for inf values
        print("np.isinf index: ", data_index) # Prints the index of the sample if there is an inf value
    if(np.isnan(data).any()):             # Checking for nan values
        print("np.isinf index: ", data_index) # Prints an index of samples with nan values
```

## 数据处理常见问题分析

### 数据准备

数据准备阶段可能存在的问题有数据集路径问题以及MindRecord 文件读写问题，包括数据读取路径和保存路径问题、文件读写问题等。

#### 数据集路径有中文

错误日志：

```text
RuntimeError: Unexpected error. Failed to open file, file path E:\深度学习\models-master\official\cv\ssd\MindRecord_COCO\test.mindrecord
```

解决方法有两种：

① 将 MindRecord 格式数据集的输出路径指定在纯英文路径下；

② MindSpore 升级到 1.6.0 之后版本。

参考实例链接：

[MindRecord 数据准备 - Unexpected error. Failed to open file_MindSpore](https://www.hiascend.com/developer/blog/details/0231107679243990127)

#### MindRecord文件问题

* 未删除重名文件

    错误日志：

    ```text
    MRMOpenError: [MRMOpenError]: MindRecord File could not open successfully.
    ```

    参考解决方法：

    ① 代码中添加删除文件逻辑，保证每次保存文件前删除目录下的重名 MindRecord 文件。

    ② MindSpore 1.6.0 之后版本，定义`FileWriter`对象时，可以加上`overwrite=True`来实现覆盖写。

    参考实例链接：

    [MindSpore 数据准备 - MindRecord File could not open successfully](https://www.hiascend.com/developer/blog/details/0231107679243990127)

* 文件被移动

    错误日志：

    ```text
    RuntimeError: Thread ID 1 Unexpected error. Fail to open ./data/cora
    RuntimeError: Unexpected error. Invalid file, DB file can not match file
    ```

    使用MindSpore 1.4及之前版本时，在Windows环境下，生成MindRecord格式数据集文件后移动位置，文件不能被正常加载到MindSpore中使用。

    参考解决方法：

    ① Windows 环境下生成的 MindRecord 格式文件不要移动位置。

    ② 将 MindSpore 升级到 1.5.0 以及之后版本，重新生成 MindRecord 格式数据集，即可正常拷贝移动。

    参考实例链接：

    [MindSpore 数据准备 - Invalid file,DB file can not match_MindSpore](https://www.hiascend.com/developer/blog/details/0229106992212728097)

* 自定义数据时类型设置错误

    错误日志：

    ```text
    RuntimeError: Unexpected error. Invalid data, the number of schema should be positive but got: 0. Please check the input schema.
    ```

    参考解决方法：

    修改数据输入类型，使其与脚本中的类型定义保持一致。

    参考实例链接：

    [MindSpore 数据准备 - Unexpected error. Invalid data](https://www.hiascend.com/developer/blog/details/0231107678315400125)

### 数据加载

数据加载阶段可能存在的问题：资源配置问题、`GeneratorDataset`相关问题以及迭代器问题等。

#### 资源配置问题

* CPU核数设置问题

    错误日志：

    ```text
    RuntimeError: Thread ID 140706176251712 Unexpected error. GeneratorDataset's num_workers=8, this value is not within the required range of [1, cpu_thread_cnt=2].
    ```

    参考解决方法：

    ① 添加代码手动配置 CPU 核数：`ds.config.set_num_parallel_workers()`

    ② 使用更高版本的 MindSpore,目前的 MindSpore 1.6.0 版本会根据硬件中CPU的核数自适应配置，避免出现CPU核数过低导致报错。

    参考实例链接：

    [MindSpore 数据加载 - Unexpected error. GeneratorDataset's num_workers=8, this value is not within the required range of](https://www.hiascend.com/forum/thread-0215121940801939033-1-1.html)

* PageSize 设置问题

    错误日志：

    ```text
    RuntimeError: Syntax error. Invalid data, Page size: 1048576 is too small to save a blob row.
    ```

    参考解决方法：

    使用MindSpore的 set_page_size API，将 pagesize 设置大一点,设置方法如下：

    ```python
    from mindspore.mindrecord import FileWriter
    writer = FileWriter(file_name="test.mindrecord", shard_num=1)
    writer.set_page_size(1 << 26) # 128MB
    ```

    参考实例链接：

    [MindSpore 数据加载 - Invalid data,Page size is too small"](https://www.hiascend.com/developer/blog/details/0231107680001698128)

#### `GeneratorDataset` 相关问题

* `GeneratorDataset` 线程卡死

    无错误日志，线程卡死

    在自定义的数据处理中，存在 ```numpy.ndarray, mindspore.Tensor```数据类型混用过程，并且错误地使用 `numpy.array(Tensor)`做转换，导致 GIL(Global Interpreter Lock) 锁得不到释放，`GeneratorDataset` 不能正常工作。

    参考解决方法：

    ① 在定义`GeneratorDataset`的第一个入参 `source` 时，如果涉及调用 Python function，则使用`numpy.ndarray`数据类型。

    ② 使用 `Tensor.asnumpy()`方法将`Tensor`转成`numpy.ndarray`。

    参考实例链接：

    [MindSpore 数据加载 - GeneratorDataset 线程卡死](https://www.hiascend.com/developer/blog/details/0232106992052900089)

* 自定义数据返回类型不正确

    错误日志：

    ```python
    Unexpected error. Invalid data type.
    ```

    错误描述：

    自定义的 `Dataset` 或 `map` 操作中返回的是一个dict类型数据等，不是 numpy array或numpy array组成的元组（tuple）。因为其他类型（dict、object等）不是一个可控的数据类型同时也不明确其中的数据存储方式，因此报出 `Invalid type` 的错误。

    参考解决方法：

    ① 检查自定义的数据处理部分的数据返回类型，需要返回 numpy array。

    ② 检查自定义加载数据时，`__getitem__`函数的数据返回类型，需要返回 tuple，且 tuple 的元素是 numpy 类型。

    参考实例链接：

    [MindSpore 数据集加载 - Unexpected error. Invalid data type_MindSpore](https://www.hiascend.com/developer/blog/details/0231107678315400125)

* 自定义采样器初始化错误

    错误日志：

    ```text
    AttributeError: 'IdentitySampler' object has no attribute 'child_sampler'
    ```

    参考解决方法：

    在自定义的采样器初始化方法'\_\_init\_\_()'中需要使用'super().\_\_init\_\_()'调用父类的构造函数。

    参考实例链接：

    [MindSpore 数据集加载 - 'IdentitySampler' has no attribute child_sampler](https://www.hiascend.com/developer/blog/details/0229107679386960150)

* 重复定义访问方式

    错误日志：

    ```python
    For 'Tensor', the type of "input_data" should be one of ...
    ```

    参考解决方法：

    选择合适的数据输入：随机访问（`__getitem__`），顺序访问（iter，next）两者选其一即可。

    参考实例链接：

    [MindSpore 数据集加载 - the type of `input_data` should be one of](https://www.hiascend.com/developer/blog/details/0229107683010760153)

* 自定义数据返回字段与定义数目不一致

    错误日志：

    ```text
    RuntimeError: Exception thrown from PyFunc. Invalid python function, the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in column_names
    ```

    参考解决方法：

    检查 `GeneratorDataset` 返回与定义的`columns`字段是否一致。

    参考实例链接：

    [MindSpore 数据集加载 - Exception thrown from PyFunc](https://www.hiascend.com/developer/blog/details/0232107680321371137)

* 用户脚本问题

    错误日志：

    ```text
    TypeError: parse() missing 1 required positionnal argument: 'self'
    ```

    参考解决方法：

    单步调试代码，检查脚本中的语法，看是否缺少'()'等。

    参考实例链接：

    [MindSpore 数据集加载 - parse() missing 1 required positional](https://www.hiascend.com/developer/blog/details/0235121940704650030)

* 自定义数据集使用了算子或Tensor操作

    错误日志：

    ```text
    RuntimeError: Exception thrown from PyFunc. RuntimeError: mindspore/ccsrc/pipeline/pynative/pynative_execute.cc:1116 GetOpOutput] : The pointer[cnode] is null.
    ```

    错误描述：

    在自定义数据集里面使用了算子或Tensor操作，而数据处理时采用多线程并行处理，但算子或Tensor操作并不支持多线程执行，因此报错。

    参考解决方法：

    用户自定义的 Pyfunc 中，在数据集中的`__getitem__` 中不使用 MindSpore的Tensor操作或算子，建议先把入参转为 Numpy 类型，再通过 Numpy 相关操作实现相关功能。

    参考实例链接：

    [MindSpore 数据集加载 - The pointer[cnode] is null](https://www.hiascend.com/developer/blog/details/0230106992306834091)

* 迭代初始化错误导致下标越界

    错误日志：

    ```python
    list index out of range
    ```

    参考解决方法：

    移除非必要的`index`成员变量，或者在每次迭代前对`index`赋值为 0 进行复位操作。

    参考实例链接：

    [MindSpore 数据集加载 - list index out of range](https://www.hiascend.com/developer/blog/details/0232107679694236136)

* 未进行迭代初始化

    错误日志：

    ```python
    Unable to fetch data from GeneratorDataset, try iterate the source function of GeneratorDataset or check value of num_epochs when create iterator.
    ```

    未进行迭代初始化导致`len`和`iter`数量不一致

    参考解决方法：

    在 iter 中加入清零操作

    参考实例链接：

    [MindSpore 数据集加载 - Unable to fetch data from GeneratorDataset](https://www.hiascend.com/developer/blog/details/0232107679694236136)

#### 迭代器相关问题

* 重复创建迭代器

    错误日志：

    ```python
    oserror: [errno 24] too many open files
    ```

    错误描述：

    重复调用`iter()`会重复创建迭代器，而 `GeneratorDataset` 加载数据集时默认为多进程加载，每次打开的句柄在主进程停止前得不到释放，导致打开句柄数一直在增长。

    参考解决方法：

    使用 MindSpore 提供的dict迭代器 `create_dict_iterator()`和 tuple 迭代器 `create_tuple_iterator()`。

    参考实例链接：

    [MindSpore 数据加载 - too many open files](https://www.hiascend.com/developer/blog/details/0231107678973789126)

* 错误使用从迭代器中获取数据的方法

    错误日志：

    ```python
    'DictIterator' has no attribute 'get_next'
    ```

    参考解决方法：

    可通过以下两种方式从迭代器中获取下一条数据：

    ```python
    item = next(ds_test.create_dict_iterator())

    for item in ds_test.create_dict_iterator():
    ```

    参考实例链接：

    [MindSpore 数据集加载- 'DictIterator' has no attribute 'get_next'](https://www.hiascend.com/developer/blog/details/0230107679565465123)

### 数据增强

数据增强阶段是对读取的数据进行数据处理，MindSpore目前支持如数据清洗shuffle、数据分批batch、数据重复repeat、数据拼接concat等常用数据处理操作。该阶段可能存在的问题有：数据类型问题、接口参数类型问题、消费节点冲突问题、数据分批问题以及内存资源问题等。

#### 自定义数据增强操作调用第三方库API时数据类型错误

错误日志：

```text
TypeError: Invalid object with type'<class 'PIL.Image.Image'>' and value'<PIL.Image.Image image mode=RGB size=180x180 at 0xFFFF6132EA58>'.
```

参考解决方法：

检查自定义函数中用到的第三方库API的数据类型要求，将输入的数据类型转换为该API期望的数据类型。

参考实例链接：

[MindSpore 数据增强 - TypeError: Invalid with type](https://www.hiascend.com/developer/blog/details/0229107679078336149)

#### 自定义数据增强操作参数类型错误

错误日志：

```text
Exception thrown from PyFunc. TypeError: args should be Numpy narray. Got <class 'tuple'>.
```

参考解决方法：

修改 `call` 的入参为个数（且类型为 numpy.ndarray），除 `self` 外入参个数需要与 `input_columns` 中的参数个数保持一致，忽略 `input_columns` 时默认为全部的数据列。

参考实例链接：

[MindSpore 数据增强 - args should be Numpy narray](https://www.hiascend.com/developer/blog/details/0230107678833189122)

#### 数据集有两个消费节点发生冲突

错误日志：

```text
ValueError: The data pipeline is not a tree (i.e. one node has 2 consumers)
```

错误描述：

dataset 定义上出现了分叉，导致 dataset 无法确定分叉的走向。

参考解决方法：

检查数据集名称，通常一直保持同一个数据集名称即可。

参考实例链接：

[MindSpore 数据增强 - The data pipeline is not a tree](https://www.hiascend.com/developer/blog/details/0230107678474985121)

#### 数据 shape 不一致导致的 batch 操作问题

错误日志：

```text
RuntimeError: Unexpected error. Inconsistent batch shapes, batch operation expect same shape for each data row, but got inconsistent shape in column 0, expected shape for this column is:, got shape:
```

参考解决方法：

① 检查需要进行 batch 操作的数据 shape，不一致时放弃进行 batch 操作。

② 如果一定要对 shape 不一致的数据进行 batch 操作，需要整理数据集，通过 pad 补全等方式进行输入数据 shape 的统一。

参考实例链接：

[MindSpore 数据增强 - Unexpected error. Inconsistent batch](https://www.hiascend.com/developer/blog/details/0254121940499220038)

#### 数据增强操作占用内存高

错误描述：

MindSpore 进行数据增强过程中，如果内存不足，可能会自动退出。 MindSpore 1.7及以后版本在内存占用超过80%时会进行告警，用户在进行大数据训练时，需要注意内存占用率，防止内存占用过高导致直接退出。

参考实例链接：

[MindSpore 数据增强 - 内存不足，自动退出](https://www.hiascend.com/developer/blog/details/0230107679768460124)
