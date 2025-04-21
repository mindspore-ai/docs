# 网络编译

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/faq/network_compilation.md)

## Q: 静态图模式支持的语法集合是什么？

A: 静态图模式能够支持覆盖Python常用语法子集，以支持神经网络的构建和训练，部分Python语法暂不支持。具体支持的语法集合，请参考[静态图语法支持](https://www.mindspore.cn/tutorials/zh-CN/br_base/compile/static_graph.html)。静态图模式提供了JIT语法支持级别选项，便于用户选择是否扩展静态图语法，对于一些网络场景，推荐使用基础语法（nn/ops等）而非扩展语法（例如numpy三方库）。此外，推荐使用 [静态图高级编程技巧](https://www.mindspore.cn/tutorials/zh-CN/br_base/compile/static_graph_expert_programming.html) 优化编译性能。

<br/>

## Q: 编译时报错'self.xx' should be initialized as a 'Parameter' type in the '`__init__`' function怎么办？

A: 在 `construct` 函数内，如果想对类成员 `self.xx` 赋值，那么 `self.xx` 必须已经在 `__init__` 函数中被定义为 [Parameter](<https://www.mindspore.cn/docs/zh-CN/br_base/api_python/mindspore/mindspore.Parameter.html>) 类型，其他类型则不支持。局部变量 `xx` 不受这个限制。

<br/>

## Q: 编译时报错`For syntax like 'a is not b', b supports True, False and None`怎么办？

A: 对于语法 `is` 或 `is not` 而言，当前 `MindSpore` 仅支持与 `True`、`False` 和 `None` 的比较。暂不支持其他类型，如字符串等。

<br/>

## Q: 编译时报错`TypeError: For 'Cell', the function construct requires 1 positional argument and 0 default argument, total 1, but got 2`怎么办？

A: 网络的实例被调用时，会执行 `construct` 方法，然后会检查 `construct` 方法需要的参数个数和实际传入的参数个数，如果不一致则会抛出以上异常。
请检查脚本中调用网络实例时传入的参数个数，和定义的网络中 `construct` 函数需要的参数个数是否一致。

<br/>

## Q: 编译时报错`Unsupported expression 'Yield'`怎么办？

A: MindSpore在静态图模式下不支持 `yield` 语法。

<br/>

## Q: 编译时报错`Type Join Failed`怎么办？

A: 在前端编译的推理阶段，会对节点的抽象类型(包含 `type`、`shape` 等)进行推导，常见抽象类型包括 `AbstractScalar`、`AbstractTensor`、`AbstractFunction`、`AbstractTuple`、`AbstractList` 等。在一些场景比如多分支场景，会对不同分支返回值的抽象类型进行 `join` 合并，推导出返回结果的抽象类型。如果抽象类型不匹配，或者 `type`/`shape` 不一致，则会抛出以上异常。

当出现类似`Type Join Failed: dtype1 = Float32, dtype2 = Float16`的报错时，说明数据类型不一致，导致抽象类型合并失败。根据提供的数据类型和代码行信息，可以快速定位出错范围。此外，报错信息中提供了具体的抽象类型信息、节点信息，可以通过 `analyze_fail.ir` 文件查看MindIR信息，定位解决问题。关于MindIR的具体介绍，可以参考[MindSpore IR（MindIR）](https://www.mindspore.cn/docs/zh-CN/br_base/design/all_scenarios.html#中间表示mindir)。代码样例如下：

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn

ms.set_context(mode=ms.GRAPH_MODE)
class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = ops.ReLU()
        self.cast = ops.Cast()

    def construct(self, x, a, b):
        if a > b:    # if的两个分支返回值的type不一致
            return self.relu(x)    # shape: (2, 3, 4, 5), dtype:Float32
        else:
            return self.cast(self.relu(x), ms.float16)    # shape: (2, 3, 4, 5)， dtype:Float16

input_x = ms.Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
input_a = ms.Tensor(2, ms.float32)
input_b = ms.Tensor(6, ms.float32)
net = Net()
out_me = net(input_x, input_a, input_b)
```

执行结果如下：

```text
TypeError: Cannot join the return values of different branches, perhaps you need to make them equal.
Type Join Failed: dtype1 = Float32, dtype2 = Float16.
For more details, please refer to https://www.mindspore.cn/search?inputValue=Type%20Join%20Failed.

Inner Message:
The abstract type of the return value of the current branch is AbstractTensor(shape: (2, 3, 4, 5), element: AbstractScalar(Type: Float16, Value: AnyValue, Shape: NoShape), value_ptr: 0x55b9f289d090, value: AnyValue), and that of the previous branch is AbstractTensor(shape: (2, 3, 4, 5), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x55b9f289d090, value: AnyValue).
The node is construct.6:[CNode]13{[0]: construct.6:[CNode]12{[0]: ValueNode<Primitive> Switch, [1]: [CNode]11, [2]: ValueNode<FuncGraph> ✓construct.4, [3]: ValueNode<FuncGraph> ✗construct.5}}, true branch: ✓construct.4, false branch: ✗construct.5

The function call stack (See file 'analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir):
# 0 In file test.py(14)
        if a > b:
        ^
```

当出现如`Type Join Failed: abstract type AbstractTensor can not join with AbstractTuple`的报错时，说明抽象类型不匹配，导致抽象类型合并失败，代码样例如下：

```python
import mindspore.ops as ops
import mindspore as ms

x = ms.Tensor([1.0])
y = ms.Tensor([2.0])
grad = ops.GradOperation(get_by_list=False, sens_param=True)
sens = 1.0

def test_net(a, b):
    return a, b

@ms.jit()
def join_fail():
    sens_i = ops.Fill()(ops.DType()(x), ops.Shape()(x), sens)    # sens_i 是一个标量shape: (1), dtype:Float64, value:1.0
    # sens_i = (sens_i, sens_i)
    a = grad(test_net)(x, y, sens_i)    # 对输出类型为tuple(Tensor, Tensor)的test_net求梯度需要sens_i的类型同输出保持一致，但sens_i是个Tensor; 在grad前设置sens_i = (sens_i, sens_i)可以修复问题。
    return a

join_fail()
```

执行结果如下：

```text
TypeError: Type Join Failed: abstract type AbstractTensor cannot join with AbstractTuple.
For more details, please refer to https://www.mindspore.cn/search?inputValue=Type%20Join%20Failed.

Inner Message:
This: AbstractTensor(shape: (1), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x55c969c44c60, value: Tensor(shape=[1], dtype=Float32, value=[ 1.00000000e+00])), other: AbstractTuple{element[0]: AbstractTensor(shape: (1), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x55c96a9a3bd0, value: Tensor(shape=[1], dtype=Float32, value=[ 1.00000000e+00])), element[1]: AbstractTensor(shape: (1), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x55c96a5f06a0, value: Tensor(shape=[1], dtype=Float32, value=[ 2.00000000e+00])), sequence_nodes: {test_net.3:[CNode]4{[0]: ValueNode<PrimitivePy> MakeTuple, [1]: a, [2]: b}, elements_use_flags: {ptr: 0x55c96ae83400, value: [const vector][1, 1]}}}. Please check the node: test_net.5:a{[0]: a, [1]: test_net}

The function call stack (See file 'analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir):

The function call stack:
# 0 In file test.py(17)
    a = grad(test_net)(x, y, sens_i)
        ^
```

<br/>

## Q: 编译时报错`The params of function 'bprop' of Primitive or Cell requires the forward inputs as well as the 'out' and 'dout'`怎么办？

A: 用户自定义的Cell的反向传播函数 `bprop`，它的输入需要包含正向网络的输入，以及 `out` 和 `dout`，代码样例如下：

```python
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore import dtype as mstype

class BpropUserDefinedNet(nn.Cell):
        def __init__(self):
            super(BpropUserDefinedNet, self).__init__()
            self.zeros_like = ops.ZerosLike()

        def construct(self, x, y):
            return x + y

        # def bprop(self, x, y, out, dout):    # 正确写法
        def bprop(self, x, y, out):
            return self.zeros_like(out), self.zeros_like(out)

ms.set_context(mode=ms.GRAPH_MODE)
net = BpropUserDefinedNet()
x = Tensor(2, mstype.float32)
y = Tensor(6, mstype.float32)
grad_fn = ms.grad(net, grad_position=(0, 1))
output = grad_fn(x, y)
print(output)
```

执行结果如下：

```text
TypeError: The params of function 'bprop' of Primitive or Cell requires the forward inputs as well as the 'out' and 'dout'.
In file test.py(13)
        def bprop(self, x, y, out):
```

<br/>

## Q: 编译时报错`There isn't any branch that can be evaluated`怎么办？

A: 当出现There isn't any branch that can be evaluated 时，说明代码中可能出现了无穷递归或者死循环，导致if条件的每一个分支都无法推导出正确的类型和维度信息。

<br/>

## Q: 编译时报错`Exceed function call depth limit 1000`怎么办？

A: 当出现Exceed function call depth limit 1000 时，说明代码中出现了无穷递归死循环，或者是代码过于复杂，类型推导过程中导致栈深度超过设置的最大深度。
此时可以通过设置 `mindspore.set_recursion_limit(recursion_limit=value)` 更改栈的最大深度，并考虑简化代码逻辑或者检查代码中是否存在无穷递归或死循环。
需要注意的是，设置recursion_limit虽然可以改变MindSpore的递归深度，但是可能会超过系统栈的最大深度，进而出现段错误。此时可能还需要设置系统栈深度。

<br/>

## Q: 编译时报错`could not get source code`以及`MindSpore can not compile temporary source code in terminal. Please write source code to a python file and run the file.`是什么原因？

A: MindSpore编译网络时通过 `inspect.getsourcelines(self.fn)` 获取网络代码所在的文件，如果网络是编辑在命令行中的临时代码，那么会出现如标题所示的报错，需要将网络写在Python文件中去执行才能避免该错误。

<br/>

## Q: 报错提示中的`Corresponding forward node candidate:”或“Corresponding code candidate:`是什么意思？

A: `Corresponding forward node candidate:`为关联的正向网络中的代码，表示该反向传播算子与该正向代码对应。`Corresponding code candidate:`表示该算子是由这些代码融合而来，其中分符“-”用以区分不同的代码。

例如：

- 算子FusionOp_BNTrainingUpdate_ReLUV2报错，打印了如下的代码行：

    ```text
    Corresponding code candidate:
     - In file /home/workspace/mindspore/build/package/mindspore/nn/layer/normalization.py(212)/                return self.bn_train(x,/
       In file /home/workspace/mindspore/tests/st/tbe_networks/resnet.py(265)/        x = self.bn1(x)/
       In file /home/workspace/mindspore/build/package/mindspore/nn/wrap/cell_wrapper.py(109)/        out = self._backbone(data)/
       In file /home/workspace/mindspore/build/package/mindspore/nn/wrap/cell_wrapper.py(356)/        loss = self.network(*inputs)/
       In file /home/workspace/mindspore/build/package/mindspore/train/dataset_helper.py(98)/        return self.network(*outputs)/
     - In file /home/workspace/mindspore/tests/st/tbe_networks/resnet.py(266)/        x = self.relu(x)/
       In file /home/workspace/mindspore/build/package/mindspore/nn/wrap/cell_wrapper.py(109)/        out = self._backbone(data)/
       In file /home/workspace/mindspore/build/package/mindspore/nn/wrap/cell_wrapper.py(356)/        loss = self.network(*inputs)/
       In file /home/workspace/mindspore/build/package/mindspore/train/dataset_helper.py(98)/        return self.network(*outputs)/
    ```

    第一个分隔符的代码调用栈指向了网络脚本文件中第265行的“x = self.bn1(x)”，第二个分隔符的代码调用栈指向了网络脚本文件中第266行的“x = self.relu(x)”。可知，该算子FusionOp_BNTrainingUpdate_ReLUV2由这两行代码融合而来。

- 算子Conv2DBackpropFilter报错，打印了如下的代码行：

    ```text
    In file /home/workspace/mindspore/build/package/mindspore/ops/_grad_experimental/grad_nn_ops.py(65)/        dw = filter_grad(dout, x, w_shape)/
    Corresponding forward node candidate:
     - In file /home/workspace/mindspore/build/package/mindspore/nn/layer/conv.py(266)/        output = self.conv2d(x, self.weight)/
       In file /home/workspace/mindspore/tests/st/tbe_networks/resnet.py(149)/        out = self.conv1(x)/
       In file /home/workspace/mindspore/tests/st/tbe_networks/resnet.py(195)/        x = self.a(x)/
       In file /home/workspace/mindspore/tests/st/tbe_networks/resnet.py(270)/        x = self.layer2(x)/
       In file /home/workspace/mindspore/build/package/mindspore/nn/wrap/cell_wrapper.py(109)/        out = self._backbone(data)/
       In file /home/workspace/mindspore/build/package/mindspore/nn/wrap/cell_wrapper.py(356)/        loss = self.network(*inputs)/
       In file /home/workspace/mindspore/build/package/mindspore/train/dataset_helper.py(98)/        return self.network(*outputs)/
    ```

    第一行是该算子的相应源码，该算子是反向算子，故由MindSpore实现。第二行提示此算子有关联的正向节点，第四行则指向了网络脚本文件第149行的“out = self.conv1(x)”。综上可知，算子Conv2DBackpropFilter是一个反向算子，相应的正向节点是一个卷积算子。

<br/>

## Q: 为什么运行代码时屏幕中会出现`Start compiling and it will take a while. Please wait...`和`End compiling.`的打印？

A: 当需要加速执行时，MindSpore会将Python源码转换成一种基于图表示的函数式IR，并进行相关的优化。这个过程也被称为编译流程。
当出现“Start compiling and it will take a while. Please wait...”的打印时，MindSpore开始了图编译流程；当出现“End compiling.”则表明图编译流程结束。

当前主要有以下两种场景会有该打印：

- 静态图模式下运行网络。
- 动态图下执行被`@jit`装饰的函数（例如优化器`nn.Momentum`）。

> 一次任务中有可能会触发多次编译流程。

<br/>

## Q: 编译时报出告警:`On the Ascend platform, if you read-only access to the parameter, you can take the value of the parameter, so that the system can do more optimization.`，是什么意思？

A: 由于Ascend平台不能真正返回一个内存地址，导致在整图下沉模式下，对于控制流场景中返回值存在参数的情况，会存在一些问题。为了避免出现问题，会对这种场景切换到统一运行时模式，从整图下沉模式切换到统一运行时模式，网络性能可能会劣化。如果控制流子图的返回值仅使用参数的值，可以通过参数的value接口获取参数的值，从而避免模式切换导致的性能劣化。

例如下面的用例，在网络“Net”中仅使用“InnerNet”中的“self.param1”和“self.param2”的值，没有使用参数的属性，所以可以使用value接口来避免模式切换导致的性能劣化。

```python
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor, Parameter

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_device(device_target="Ascend")

class InnerNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param1 = Parameter(Tensor(1), name="param1")
        self.param2 = Parameter(Tensor(2), name="param2")

    def construct(self, x):
        if x > 0:
            return self.param1.value(), self.param2.value()
        return self.param2.value(), self.param1.value()

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.inner_net = InnerNet()
        self.addn = ops.AddN()

    def construct(self, x, y):
        inner_params = self.inner_net(x)
        out_res = self.addn(inner_params) + y
        return out_res, inner_params[0] + inner_params[1]

input_x = Tensor(3)
input_y = Tensor(5)
net = Net()
out = net(input_x, input_y)
print("out:", out)
```

执行结果如下：

```text
out: (Tensor(shape=[], dtype=Int64, value=8), Tensor(shape=[], dtype=Int64, value=3))
```

<br/>

## Q: load MindIR 时，出现 `The input number of parameters is not Compatible.`该怎么办？

A: 首先检查导出参数和导入执行的参数个数是否是匹配的。如果是匹配的，则需要检查一下导出时候的参数是不是存在非Tensor的场景。

因为导出数据输入为非Tensor时，该导出的输入将会变成常量固化到MindIR中，使MindIR中的输入要少于网络构建的Construct入参。

如果是标量类型，可以将标量转成Tensor类型导出。如果是Tuple或者List类型，可以使用[mutable](https://www.mindspore.cn/docs/zh-CN/br_base/api_python/mindspore/mindspore.mutable.html)接口进行包装后及进行导出。

<br/>

## Q: 编译时报错`ValueError: The shape of sense must not be dynamic shape.`怎么办？

A: 在图模式中，当调用GradOperation接口且参数sens_param=True时，通过nn.Cell.set_inputs传入动态shape的sense参数时会导致报错。代码样例如下：

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_device(device_target="CPU")

class Net(nn.Cell):
    """ReLU Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.relu = ops.ReLU()

    def construct(self, x):
        return self.relu(x)

class GradWithSense(nn.Cell):
    """Grad Net"""
    def __init__(self, network):
        super(GradWithSense, self).__init__()
        self.grad = ops.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_, sense):
        return self.grad(self.network)(input_, sense)

x = np.array([[1, 1], [1, -1]]).astype(np.float32)
sense = np.array([[2, 3], [4, 5]]).astype(np.float32)
dynamic_x = Tensor(shape=[2, None], dtype=ms.float32)
sense_x = Tensor(shape=[1, None], dtype=ms.float32)
net = GradWithSense(Net())
net.set_inputs(dynamic_x, sense_x)
print(net(Tensor(x), Tensor(sense_x))) # ValueError: The shape of sense must not be dynamic shape.
```

图模式下，不支持动态shape的sense，建议修改为以下代码：

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_device(device_target="CPU")

class Net(nn.Cell):
    """ReLU Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.relu = ops.ReLU()

    def construct(self, x):
        return self.relu(x)

class NetWithSense(nn.Cell):
    """ReLU Net"""
    def __init__(self, sense):
        super(NetWithSense, self).__init__()
        self.relu = ops.ReLU()
        self.sense = sense

    def construct(self, x):
        return self.relu(x) * self.sense  # 将sense加入正向计算网络中

class Grad(nn.Cell):
    """Grad Net"""
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation(get_all=True)
        self.network = network

    def construct(self, input_):
        return self.grad(self.network)(input_)

x = np.array([[1, 1], [1, -1]]).astype(np.float32)
sense = np.array([[2, 3], [4, 5]]).astype(np.float32)
dynamic_x = Tensor(shape=[2, None], dtype=ms.float32)
net = Grad(NetWithSense(Tensor(sense)))
net.set_inputs(dynamic_x)
print(net(Tensor(x)))
```

执行结果如下：

```text
(Tensor(shape=[2, 2], dtype=Float32, value=
[[ 2.00000000e+00,  3.00000000e+00],
 [ 4.00000000e+00,  0.00000000e+00]]),)
```

<br/>

## Q: 编译时报错 `'External' TypeError`怎么办？

A: “External” 类型表示在图模式中使用了无法原生支持的对象。例如：第三方库对象是 “External” 类型。

<br/>

## Q: 编译时报错`Nested execution during JIT execution for 'xxx' is not supported when 'xxx' compile and execute.`怎么办？

A: 当触发编译流程，即代码编译成静态计算图时，见[Graph模式执行原理](https://www.mindspore.cn/docs/zh-CN/br_base/features/program_form/overview.html)，同时在默认使用JIT Fallback特性时，再次进入编译流程时，则会抛出以上异常。

下面以JIT Fallback支持调用第三方库的对象和方法为例：

1) 再次调用@jit装饰器修饰函数或者类的成员方法，所修饰的函数或方法将会被编译成静态计算图。

```python
from mindspore import context, Tensor, jit, nn
import numpy as np
context.set_context(mode=context.GRAPH_MODE)

class UserDefinedNet: # 自定义普通Python类
    def __init__(self):
        self.value = 10

    @jit
    def func(self, x):  # jit装饰的方法
        return 2 * x + self.value

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.net = UserDefinedNet()

    def construct(self, x):
        x = self.net.value + self.net.func(x)
        return x

x = np.random.randn(2, 2, 3).astype(np.float32)
net = Net()
out = net(Tensor(x))
```

执行结果如下：

```text
Nested execution during JIT execution for 'UserDefinedNet.func' is not supported when 'Net.construct' compile and execute.
```

当前场景建议去掉@jit装饰器。

2) 使用Cell类并且在construct函数中编写执行代码，此时construct函数的代码将会被编译成静态计算图。

```python
from mindspore import context, Tensor, jit, nn
import numpy as np
context.set_context(mode=context.GRAPH_MODE)

class InnerNet(nn.Cell):
    def __init__(self):
        super(InnerNet, self).__init__()

    def construct(self, x):
        return x

class UserDefinedNet: # 自定义普通Python类
    def __init__(self):
        self.value = 10
        self.inner_net = InnerNet()

    def func(self, x):
        return 2 * x * self.inner_net(x) + self.value

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.net = UserDefinedNet()

    def construct(self, x):
        x = self.net.value + self.net.func(x)
        return x

x = np.random.randn(2, 2, 3).astype(np.float32)
net = Net()
out = net(Tensor(x))
```

执行结果如下：

```text
Nested execution during JIT execution for 'InnerNet.construct' is not supported when 'Net.construct' compile and execute.
```

建议修改为以下代码：

```python
from mindspore import context, Tensor, jit, nn
import numpy as np
context.set_context(mode=context.GRAPH_MODE)

class InnerNet(nn.Cell):
    def __init__(self):
        super(InnerNet, self).__init__()

    def construct(self, x):
        return x

class UserDefinedNet: # 自定义普通Python类
    def __init__(self):
        self.value = 10

    def func(self, x, y):
        return 2 * x * y + self.value

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.net = UserDefinedNet()
        self.inner_net = InnerNet()

    def construct(self, x):
        y = self.inner_net(x)
        x = self.net.value + self.net.func(x, y)
        return x

x = np.random.randn(2, 2, 3).astype(np.float32)
net = Net()
out = net(Tensor(x))
```

3) 自定义类中调用了使用@jit装饰器修饰的函数，将会报错。这种场景建议将网络中的自定义类加上@jit_class装饰器，避免使用JIT Fallback特性。自定义类的更多使用可参考[自定义类的使用](https://www.mindspore.cn/tutorials/zh-CN/br_base/compile/static_graph.html#支持自定义类的使用)。jit_class装饰器的使用可参考[使用jit_class](https://www.mindspore.cn/tutorials/zh-CN/br_base/compile/static_graph_expert_programming.html#使用jit-class)。

```python
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE)

class InnerNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.value = 10

    @ms.jit
    def construct(self, x):
        return self.value + x

class CustomNet():
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, x):
        return self.model(2 * x)

class OutNet(ms.nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, x):
        return self.net(x)

x = ms.Tensor(2)
call_net = InnerNet()
custom_net = CustomNet(call_net)
out_net = OutNet(custom_net)
out = out_net(x)
print("out:", out)
```

执行结果如下：

```text
Nested execution during JIT execution for 'InnerNet.construct' is not supported when 'OuterNet.construct' compile and execute.
```

建议修改为以下代码：

```python
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE)

class InnerNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.value = 10

    @ms.jit
    def construct(self, x):
        return self.value + x

@ms.jit_class
class CustomNet():
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, x):
        return self.model(2 * x)

class OutNet(ms.nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, x):
        return self.net(x)

x = ms.Tensor(2)
call_net = InnerNet()
custom_net = CustomNet(call_net)
out_net = OutNet(custom_net)
out = out_net(x)
print("out:", out)
```

<br/>

## Q: 编译时报错 `ValueError: The value Parameter (name=name_a, shape=(1,), dtype=Float32, requires_grad=True) , its name 'name_a' already exists. Please set a unique name for the parameter.`，是什么含义？应该怎么处理？

A: 图模式下要求Parameter的name拥有唯一性，如果存在同名的两个或者多个Parameter，网络中区分不出不同的对象，将造成错误。我们可以从下面几个角度来排查脚本中的同名的Parameter，对其中的Parameter设置唯一的name。

```python
import mindspore as ms
from mindspore.nn import Cell
from mindspore import Tensor, context, ParameterTuple, Parameter

context.set_context(mode=context.GRAPH_MODE)


class ParamNet(Cell):
    def __init__(self):
        super(ParamNet, self).__init__()
        self.res1 = ParameterTuple((Parameter(Tensor([2], ms.float32), name="name_a"),
                                    Parameter(Tensor([4], ms.float32), name="name_a")))
        self.param_tuple = (Parameter(Tensor([1], ms.float32), name="name_b"),
                            Parameter(Tensor([2], ms.float32)))
        self.param_list = [Parameter(Tensor([3], ms.float32), name="name_b"),
                           Parameter(Tensor([4], ms.float32))]

    def construct(self):
        out1 = self.res1[0] + self.res1[1]
        out2 = self.param_tuple[0] + self.param_tuple[1] + self.param_list[0] + self.param_listp[1]
        return out1, out2


net = ParamNet()
res = net()
```

如上脚本，ParameterTuple中定义了两个同名name_a的Parameter，是不允许的。param_tuple和param_list中定义了同名name_b的Parameter，也是不允许的。还有一种情况是脚本中在同一个Cell中实例化某个网络，如下面例子，也将报错`its name 'name_a' already exists.`。

```python
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context, ParameterTuple, Parameter


context.set_context(mode=context.GRAPH_MODE)


class InnerNet(nn.Cell):
    def __init__(self):
        super(InnerNet, self).__init__()
        self.param = Parameter(Tensor([1], ms.float32), name="name_a")

    def construct(self, x):
        return x + self.param


class OutNet1(nn.Cell):
    def __init__(self, net1, net2):
        super(OutNet1, self).__init__()
        self.param1 = ParameterTuple(net1.get_parameters())
        self.param2 = ParameterTuple(net2.get_parameters())

    def construct(self, x):
        return x + self.param1[0] + self.param2[0]


net1 = InnerNet()
net2 = InnerNet()
out_net = OutNet1(net1, net2)
res = out_net(Tensor([1], ms.float32))
print("res:", res)
```

针对这种情况，我们可以使用CellList来管理同一个网络的多个实例。

```python
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context, ParameterTuple, Parameter


context.set_context(mode=context.GRAPH_MODE)


class InnerNet(nn.Cell):
    def __init__(self):
        super(InnerNet, self).__init__()
        self.param = Parameter(Tensor([1], ms.float32), name="name_a")

    def construct(self, x):
        return x + self.param


class OutNet1(nn.Cell):
    def __init__(self, net1, net2):
        super(OutNet1, self).__init__()
        self.cell_list = nn.CellList()
        self.cell_list.append(net1)
        self.cell_list.append(net2)
        self.param1 = ParameterTuple(self.cell_list[0].get_parameters())
        self.param2 = ParameterTuple(self.cell_list[1].get_parameters())

    def construct(self, x):
        return x + self.param1[0] + self.param2[0]


net1 = InnerNet()
net2 = InnerNet()
out_net = OutNet1(net1, net2)
res = out_net(Tensor([1], ms.float32))
print("res:", res)
```

如下场景，网络Net中已经定义了bias的Parameter和网络实例化再次定义同名的Parameter，在图模式下是不允许。

```python
import mindspore as ms
ms.set_context(mode=context.GRAPH_MODE)

class Net(ms.nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.bias = ms.Parameter(ms.Tensor(2.), name='bias')

class SubNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()

sub = SubNet()
net = Net(sub)

net.net.bias = ms.Parameter(ms.Tensor(2.), name='bias')
a = ms.Tensor(3.)
grad_fn = ms.value_and_grad(net, None, net.trainable_params())
print(grad_fn(a))
```

<br/>

## Q: 多次调用同一个网络时，什么情况会重新编译？

A: 以下场景会触发重新编译：

- Tensor的shape发生改变。

- 标量值发生改变。

- Tuple或List的长度发生改变。

- 网络的输入是tuple[Tensor]、list[Tensor]或Dict[Tensor]，即使里面Tensor的shape和dtype没有发生变化。详情请参考 [mutable](https://www.mindspore.cn/docs/zh-CN/br_base/api_python/mindspore/mindspore.mutable.html)。

<br/>

## Q: 静态图模式如何判断有几张图？什么情况会切分子图？多子图有什么影响？如何避免出现多子图？

A: 1、子图数量可以通过查看IR文件并搜索"Total subgraphs"获取。关于如何查看分析IR文件，请参考 [IR文件分析](https://www.mindspore.cn/tutorials/zh-CN/br_base/debug/error_analysis/mindir.html)。

2、图模式切分子图，常见于控制流场景，如if/while等。除了用户手动编写，MindSpore框架内部实现的控制流语法也可能会切分出多张子图。

3、多子图可能影响网络执行性能。

4、为避免出现多张子图，尽量避免出现if/while的条件依赖Tensor计算结果。

<br/>
