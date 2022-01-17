# PyNative模式应用

`Ascend` `GPU` `CPU` `模型运行`

<a href="https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL21pbmRzcG9yZV9kZWJ1Z2dpbmdfaW5fcHluYXRpdmVfbW9kZS5pcHluYg==&imageid=65f636a0-56cf-49df-b941-7d2a07ba8c8c" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_modelarts.png"></a>&nbsp;&nbsp;
<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/programming_guide/source_zh_cn/debug_in_pynative_mode.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## 概述

MindSpore支持两种运行模式，在调试或者运行方面做了不同的优化:

- PyNative模式：也称动态图模式，将神经网络中的各个算子逐一下发执行，方便用户编写和调试神经网络模型。
- Graph模式：也称静态图模式或者图模式，将神经网络模型编译成一整张图，然后下发执行。该模式利用图优化等技术提高运行性能，同时有助于规模部署和跨平台运行。

默认情况下，MindSpore处于Graph模式，可以通过`context.set_context(mode=context.PYNATIVE_MODE)`切换为PyNative模式；同样地，MindSpore处于PyNative模式时，可以通过`context.set_context(mode=context.GRAPH_MODE)`切换为Graph模式。

PyNative模式下，支持执行单算子、普通函数和网络，以及单独求梯度的操作。下面将详细介绍使用方法和注意事项。

> PyNative模式下为了提升性能，算子在device上使用了异步执行方式，因此在算子执行错误的时候，错误信息可能会在程序执行到最后才显示。因此在PyNative模式下，增加了一个pynative_synchronize的设置来控制算子device上是否使用异步执行。
>
> 下述例子中，参数初始化使用了随机值，在具体执行中输出的结果可能与本地执行输出的结果不同；如果需要稳定输出固定的值，可以设置固定的随机种子，设置方法请参考[mindspore.set_seed()](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore/mindspore.set_seed.html)。

## 设置模式

```python
context.set_context(mode=context.PYNATIVE_MODE)
```

## 执行单算子

```python
import numpy as np
import mindspore.ops as ops
from mindspore import context, Tensor

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

x = Tensor(np.ones([1, 3, 5, 5]).astype(np.float32))
y = Tensor(np.ones([1, 3, 5, 5]).astype(np.float32))
z = ops.add(x, y)
print(z.asnumpy())
```

输出：

```text
[[[[2. 2. 2. 2. 2.]
   [2. 2. 2. 2. 2.]
   [2. 2. 2. 2. 2.]
   [2. 2. 2. 2. 2.]
   [2. 2. 2. 2. 2.]]

  [[2. 2. 2. 2. 2.]
   [2. 2. 2. 2. 2.]
   [2. 2. 2. 2. 2.]
   [2. 2. 2. 2. 2.]
   [2. 2. 2. 2. 2.]]

  [[2. 2. 2. 2. 2.]
   [2. 2. 2. 2. 2.]
   [2. 2. 2. 2. 2.]
   [2. 2. 2. 2. 2.]
   [2. 2. 2. 2. 2.]]]]
```

## 执行函数

```python
import numpy as np
from mindspore import context, Tensor
import mindspore.ops as ops

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

def add_func(x, y):
    z = ops.add(x, y)
    z = ops.add(z, x)
    return z

x = Tensor(np.ones([3, 3], dtype=np.float32))
y = Tensor(np.ones([3, 3], dtype=np.float32))
output = add_func(x, y)
print(output.asnumpy())
```

输出：

```text
[[3. 3. 3.]
 [3. 3. 3.]
 [3. 3. 3.]]
```

## 执行网络

在construct中定义网络结构，在具体运行时，下例中，执行net(x, y)时，会从construct函数中开始执行。

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()

    def construct(self, x, y):
        return self.mul(x, y)

x = Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
y = Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))

net = Net()
print(net(x, y))
```

输出：

```text
[ 4. 10. 18.]
```

## 构建网络

可以在网络初始化时，明确定义网络所需要的各个部分，在construct中定义网络结构。

```python
import mindspore.nn as nn
from mindspore.common.initializer import Normal

class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
            self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))


    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## 设置Loss函数及优化器

在PyNative模式下，通过优化器针对每个参数对应的梯度进行参数更新。

```python
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
net_opt = nn.Momentum(network.trainable_params(), config.lr, config.momentum)
```

## 保存模型参数

保存模型可以通过定义CheckpointConfig来指定模型保存的参数。

save_checkpoint_steps：每多少个step保存一下参数；keep_checkpoint_max：最多保存多少份模型参数。详细使用方式请参考[保存模型](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/save_model.html)。

```python
config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", directory=config.ckpt_path, config=config_ck)
```

## 训练网络

```python
context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)
ds_train = create_dataset(os.path.join(config.data_path, "train"), config.batch_size)
network = LeNet5(config.num_classes)
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
net_opt = nn.Momentum(network.trainable_params(), config.lr, config.momentum)
time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                keep_checkpoint_max=config.keep_checkpoint_max)
ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", directory=config.ckpt_path, config=config_ck)

model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()}, amp_level="O2")
```

完整的运行代码可以到ModelZoo下载[lenet](https://gitee.com/mindspore/models/tree/r1.6/official/cv/lenet)，在train.py中修改为context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)。

## 提升PyNative性能

为了提高PyNative模式下的前向计算任务执行速度，MindSpore提供了ms_function功能，该功能可以在PyNative模式下将Python函数或者Python类的方法编译成计算图，通过图优化等技术提高运行速度，如下例所示。

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore import ms_function

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

class TensorAddNet(nn.Cell):
    def __init__(self):
        super(TensorAddNet, self).__init__()
        self.add = ops.Add()

    @ms_function
    def construct(self, x, y):
        res = self.add(x, y)
        return res

x = Tensor(np.ones([4, 4]).astype(np.float32))
y = Tensor(np.ones([4, 4]).astype(np.float32))
net = TensorAddNet()

z = net(x, y) # Staging mode
add = ops.Add()
res = add(x, z) # PyNative mode
print(res.asnumpy())
```

输出：

```text
[[3. 3. 3. 3.]
 [3. 3. 3. 3.]
 [3. 3. 3. 3.]
 [3. 3. 3. 3.]]
```

上述示例代码中，在`TensorAddNet`类的`construct`之前加装了`ms_function`装饰器，该装饰器会将`construct`方法编译成计算图，在给定输入之后，以图的形式下发执行，而上一示例代码中的`add`会直接以普通的PyNative的方式执行。

需要说明的是，加装了`ms_function`装饰器的函数中，如果包含不需要进行参数训练的算子（如`pooling`、`add`等算子），则这些算子可以在被装饰的函数中直接调用，如下例所示。

示例代码：

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore import ms_function

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

add = ops.Add()

@ms_function
def add_fn(x, y):
    res = add(x, y)
    return res

x = Tensor(np.ones([4, 4]).astype(np.float32))
y = Tensor(np.ones([4, 4]).astype(np.float32))
z = add_fn(x, y)
print(z.asnumpy())
```

输出：

```text
[[2. 2. 2. 2.]
 [2. 2. 2. 2.]
 [2. 2. 2. 2.]
 [2. 2. 2. 2.]]
```

如果被装饰的函数中包含了需要进行参数训练的算子（如`Convolution`、`BatchNorm`等算子），则这些算子必须在被装饰的函数之外完成实例化操作，如下例所示。

示例代码：

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore import ms_function

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

conv_obj = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=0)
conv_obj.init_parameters_data()
@ms_function
def conv_fn(x):
    res = conv_obj(x)
    return res

input_data = np.random.randn(2, 3, 6, 6).astype(np.float32)
z = conv_fn(Tensor(input_data))
print(z.asnumpy())
```

输出：

```text
[[[[ 0.10377571 -0.0182163 -0.05221086]
[ 0.1428334 -0.01216263 0.03171652]
[-0.00673915 -0.01216291 0.02872104]]

[[ 0.02906547 -0.02333629 -0.0358406 ]
[ 0.03805163 -0.00589525 0.04790922]
[-0.01307234 -0.00916951 0.02396654]]

[[ 0.01477884 -0.06549098 -0.01571796]
[ 0.00526886 -0.09617482 0.04676902]
[-0.02132788 -0.04203424 0.04523344]]

[[ 0.04590619 -0.00251453 -0.00782715]
[ 0.06099087 -0.03445276 0.00022781]
[ 0.0563223 -0.04832596 -0.00948266]]]

[[[ 0.08444098 -0.05898955 -0.039262 ]
[ 0.08322686 -0.0074796 0.0411371 ]
[-0.02319113 0.02128408 -0.01493311]]

[[ 0.02473745 -0.02558945 -0.0337843 ]
[-0.03617039 -0.05027632 -0.04603915]
[ 0.03672804 0.00507637 -0.08433761]]

[[ 0.09628943 0.01895323 -0.02196114]
[ 0.04779419 -0.0871575 0.0055248 ]
[-0.04382382 -0.00511185 -0.01168541]]

[[ 0.0534859 0.02526264 0.04755395]
[-0.03438103 -0.05877855 0.06530266]
[ 0.0377498 -0.06117418 0.00546303]]]]
```

更多ms_function的功能可以参考[ms_function文档](https://mindspore.cn/docs/programming_guide/zh-CN/r1.6/ms_function.html)

## PyNative下同步执行

PyNative模式下算子默认为异步执行，可以通过设置context来控制是否异步执行，当算子执行失败时，可以方便地通过调用栈看到出错的代码位置。

设置为同步执行：

```python
context.set_context(pynative_synchronize=True)
```

示例代码:

```python
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.ops as ops

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", pynative_synchronize=True)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.get_next = ops.GetNext([mstype.float32], [(1, 1)], 1, "test")

    def construct(self, x1,):
        x = self.get_next()
        x = x + x1
        return x

context.set_context()
x1 = np.random.randn(1, 1).astype(np.float32)
net = Net()
output = net(Tensor(x1))
print(output.asnumpy())
```

输出：此时算子为同步执行，当算子执行错误时，可以看到完整的调用栈，找到出错的代码行。

```text
Traceback (most recent call last):
  File "test_pynative_sync_control.py", line 41, in <module>
    output = net(Tensor(x1))
  File "mindspore/mindspore/nn/cell.py", line 406, in <module>
    output = self.run_construct(cast_inputs, kwargs)
  File "mindspore/mindspore/nn/cell.py", line 348, in <module>
    output = self.construct(*cast_inputs, **kwargs)
  File "test_pynative_sync_control.py", line 33, in <module>
    x = self.get_next()
  File "mindspore/mindspore/ops/primitive.py", line 247, in <module>
    return _run_op(self, self.name, args)
  File "mindspore/mindspore/common/api.py", line 77, in <module>
    results = fn(*arg, **kwargs)
  File "mindspore/mindspore/ops/primitive.py", line 677, in _run_op
    output = real_run_op(obj, op_name, args)
RuntimeError: mindspore/ccsrc/runtime/device/kernel_runtime.cc:1006 DebugStreamSync] Op Default/GetNext-op0 run failed!
```

## Hook功能

调试深度学习网络是每一个深度学习领域的从业者需要面对，且投入大量精力的工作。由于深度学习网络隐藏了中间层算子的输入、输出梯度，只提供输入数据（特征量、权重）的梯度，导致无法准确地感知中间层算子的梯度变化，从而影响调试效率。为了方便用户准确、快速地对深度学习网络进行调试，MindSpore在PyNative模式下设计了Hook功能，使用Hook功能可以捕获中间层算子的输入、输出梯度。目前，PyNative模式下提供了两种形式的Hook功能，分别是：HookBackward算子和在nn.Cell对象上进行注册的register_backward功能。

### HookBackward算子

HookBackward将Hook功能以算子的形式实现。用户初始化一个HookBackward算子，将其安插到深度学习网络中需要捕获梯度的位置。在网络正向执行时，HookBackward算子将输入数据不做任何修改地原样输出；在网络反向传播梯度时，在HookBackward上注册的Hook函数将会捕获反向传播至此的梯度。用户可以在Hook函数中自定义对梯度的操作，比如打印梯度，或者返回新的梯度。

示例代码:

```python
import mindspore
from mindspore import ops
from mindspore import Tensor
from mindspore import context
from mindspore.ops import GradOperation

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

def hook_fn(grad_out):
    print(grad_out)

grad_all = GradOperation(get_all=True)
hook = ops.HookBackward(hook_fn)
def hook_test(x, y):
    z = x * y
    z = hook(z)
    z = z * y
    return z

def net(x, y):
    return grad_all(hook_test)(x, y)

output = net(Tensor(1, mindspore.float32), Tensor(2, mindspore.float32))
print(output)
```

输出：

```python
(Tensor(shape=[], dtype=Float32, value= 2),)
(Tensor(shape=[], dtype=Float32, value= 4), Tensor(shape=[], dtype=Float32, value= 4))
```

更多HookBackward算子的说明可以参考[API文档](https://mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.HookBackward.html)。

### nn.Cell对象的register_backward_hook功能

用户可以在nn.Cell对象上使用register_backward_hook接口，注册一个自定义的Hook函数，用来捕获网络反向传播时，与该nn.Cell对象相关联的梯度。与HookBackward算子所使用的自定义Hook函数有所不同，register_backward_hook接口使用的Hook函数的入参中包含了注册nn.Cell对象的id信息，反向传入的梯度以及反向输出的梯度。

示例代码:

```python
def cell_hook_function(cell_id, grad_input, grad_output):
    print(grad_input)
    print(grad_output)
```

这里的`grad_input`是梯度反向传播时，传入到nn.Cell对象的梯度，它对应正向过程中下一个算子的反向输出梯度；`grad_output`是nn.Cell对象反向输出的梯度。因此，用户可以使用register_backward_hook接口捕获网络中某一个nn.Cell对象的反向传入和反向输出梯度。用户可以在自定义的Hook函数中，自定义对梯度的操作，比如打印梯度，或者返回新的输出梯度。

示例代码:

```python
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import GradOperation

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

def cell_hook_function(cell_id, grad_input, grad_output):
    print(grad_input)
    print(grad_output)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.bn.register_backward_hook(cell_hook_function)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

grad_all = GradOperation(get_all=True)
output = grad_all(Net())(Tensor(np.ones([1, 1, 2, 2]).astype(np.float32)))
print(output)
```

输出：

```python
(Tensor(shape=[1, 2, 1, 1], dtype=Float32, value=
[[[[ 1.00000000e+00]],
  [[ 1.00000000e+00]]]]),)
(Tensor(shape=[1, 2, 1, 1], dtype=Float32, value=
[[[[ 9.99994993e-01]],
  [[ 9.99994993e-01]]]]),)
(Tensor(shape=[1, 1, 2, 2], dtype=Float32, value=
[[[[ 1.99998999e+00, 1.99998999e+00],
   [ 1.99998999e+00, 1.99998999e+00]]]]),)
```

更多关于nn.Cell对象的register_backward_hook功能的说明可以参考[API文档](https://mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_backward_hook)。

## 自定义bprop功能

用户可以自定义nn.Cell对象的反向传播（计算）函数，从而控制nn.Cell对象梯度计算的过程，定位梯度问题。自定义bprop函数的使用方法是：在定义的nn.Cell对象里面增加一个用户自定义的bprop函数。训练的过程中会使用用户自定义的bprop函数来生成反向图。

示例代码:

```python
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import GradOperation

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

    def construct(self, x, y):
        z = x * y
        z = z * y
        return z

    def bprop(self, x, y, out, dout):
        x_dout = x + y
        y_dout = x * y
        return x_dout, y_dout

grad_all = GradOperation(get_all=True)
output = grad_all(Net())(Tensor(1, mindspore.float32), Tensor(2, mindspore.float32))
print(output)
```

输出：

```python
(Tensor(shape=[], dtype=Float32, value= 3), Tensor(shape=[], dtype=Float32, value= 2))
```
