# 网络编译

<a href="https://gitee.com/mindspore/docs/blob/r1.10/docs/mindspore/source_zh_cn/faq/network_compilation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source.png"></a>

<font size=3>**Q: 编译时报错“'self.xx' should be initialized as a 'Parameter' type in the '`__init__`' function”怎么办？**</font>

A: 在 `construct` 函数内，如果想对类成员 `self.xx` 赋值，那么 `self.xx` 必须已经在 `__init__` 函数中被定义为 [Parameter](<https://www.mindspore.cn/docs/zh-CN/r1.10/api_python/mindspore/mindspore.Parameter.html>) 类型，其他类型则不支持。局部变量 `xx` 不受这个限制。

<br/>

<font size=3>**Q: 编译时报错“For syntax like 'a is not b', b supports True, False and None”怎么办？**</font>

A: 对于语法 `is` 或 `is not` 而言，当前 `MindSpore` 仅支持与 `True`、`False` 和 `None` 的比较。暂不支持其他类型，如字符串等。

<br/>

<font size=3>**Q: 编译时报错“Only support comparison with 1 operator, but got 2”怎么办？**</font>

A: 对于比较语句，`MindSpore` 最多支持一个操作数。例如不支持语句 `1 < x < 3`，请使用 `1 < x and x < 3` 的方式代替。

<br/>

<font size=3>**Q: 编译时报错“TypeError: For 'Cell', the function construct requires 1 positional argument and 0 default argument, total 1, but got 2”怎么办？**</font>

A: 网络的实例被调用时，会执行 `construct` 方法，然后会检查 `construct` 方法需要的参数个数和实际传入的参数个数，如果不一致则会抛出以上异常。
请检查脚本中调用网络实例时传入的参数个数，和定义的网络中 `construct` 函数需要的参数个数是否一致。

<br/>

<font size=3>**Q: 编译时报错“Unsupported expression 'Yield'”怎么办？**</font>

A: MindSpore在静态图模式下不支持 `yield` 语法。另外，在静态图模式下，如果代码中错误使用了 `net.trainable_params()` 不支持语法，也会触发该报错，因为其内部实现使用了 `list(filter(iterator))` 语法，隐式调用了 `yield` 语法。代码样例如下：

```python
import mindspore as ms
from mindspore import set_context, nn

class Net(nn.Cell):
    def construct(self):
        return True

class TestNet(nn.Cell):
    def __init__(self):
        super(TestNet, self).__init__()
        self.net = Net()

    def construct(self):
        return net.trainable_params()

set_context(mode=ms.GRAPH_MODE)
net = TestNet()
out = net()
```

执行结果如下：

```text
RuntimeError: Unsupported expression 'Yield'.
More details please refer to syntax support at https://www.mindspore.cn

----------------------------------------------------
- The Traceback of Net Construct Code:
----------------------------------------------------
The function call stack (See file 'analyze_fail.dat' for more details. Get instructions about `analyze_fail.dat` at https://www.mindspore.cn/search?inputValue=analyze_fail.dat):
# 0 In file test.py:13
        return net.trainable_params()
               ^
# 1 In file /home/workspace/mindspore/build/package/mindspore/nn/cell.py:1257
        return list(filter(lambda x: x.requires_grad, self.get_parameters(expand=recurse)))
                                                      ^
```

<br/>

<font size=3>**Q: 编译时报错“Type Join Failed”或“Shape Join Failed”怎么办？**</font>

A: 在前端编译的推理阶段，会对节点的抽象类型(包含 `type`、`shape` 等)进行推导，常见抽象类型包括 `AbstractScalar`、`AbstractTensor`、`AbstractFunction`、`AbstractTuple`、`AbstractList` 等。在一些场景比如多分支场景，会对不同分支返回值的抽象类型进行 `join` 合并，推导出返回结果的抽象类型。如果抽象类型不匹配，或者 `type`/`shape` 不一致，则会抛出以上异常。

当出现类似“Type Join Failed: dtype1 = Float32, dtype2 = Float16”的报错时，说明数据类型不一致，导致抽象类型合并失败。根据提供的数据类型和代码行信息，可以快速定位出错范围。此外，报错信息中提供了具体的抽象类型信息、节点信息，可以通过 `analyze_fail.dat` 文件查看MindIR信息，定位解决问题。关于MindIR的具体介绍，可以参考[MindSpore IR（MindIR）](https://www.mindspore.cn/docs/zh-CN/r1.10/design/mindir.html)。代码样例如下：

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

The function call stack (See file 'analyze_fail.dat' for more details. Get instructions about `analyze_fail.dat` at https://www.mindspore.cn/search?inputValue=analyze_fail.dat):
# 0 In file test.py(14)
        if a > b:
        ^
```

当出现类似“Shape Join Failed: shape1 = (2, 3, 4, 5), shape2 = ()”的报错时，说明 `shape` 不一致，导致抽象类型合并失败。代码样例如下:

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
        self.reducesum = ops.ReduceSum()

    def construct(self, x, a, b):
        if a > b:    # if的两个分支返回值的shape不一致
            return self.relu(x)    # shape: (2, 3, 4, 5), dtype:Float32
        else:
            return self.reducesum(x)    # shape:(), dype: Float32

input_x = ms.Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
input_a = ms.Tensor(2, ms.float32)
input_b = ms.Tensor(6, ms.float32)
net = Net()
out = net(input_x, input_a, input_b)
```

执行结果如下：

```text
ValueError: Cannot join the return values of different branches, perhaps you need to make them equal.
Shape Join Failed: shape1 = (2, 3, 4, 5), shape2 = ().
For more details, please refer to https://www.mindspore.cn/search?inputValue=Type%20Join%20Failed.

Inner Message:
The abstract type of the return value of the current branch is AbstractTensor(shape: (), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x55658aa9b090, value: AnyValue), and that of the previous branch is AbstractTensor(shape: (2, 3, 4, 5), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x55658aa9b090, value: AnyValue).
The node is construct.6:[CNode]13{[0]: construct.6:[CNode]12{[0]: ValueNode<Primitive> Switch, [1]: [CNode]11, [2]: ValueNode<FuncGraph> ✓construct.4, [3]: ValueNode<FuncGraph> ✗construct.5}}, true branch: ✓construct.4, false branch: ✗construct.5

The function call stack (See file 'analyze_fail.dat' for more details. Get instructions about `analyze_fail.dat` at https://www.mindspore.cn/search?inputValue=analyze_fail.dat):
# 0 In file test.py(14)
        if a > b:
        ^
```

当出现如“Type Join Failed: abstract type AbstractTensor can not join with AbstractTuple”的报错时，说明抽象类型不匹配，导致抽象类型合并失败，代码样例如下：

```python
import mindspore.ops as ops
import mindspore as ms
from mindspore import ms_function

x = ms.Tensor([1.0])
y = ms.Tensor([2.0])
grad = ops.GradOperation(get_by_list=False, sens_param=True)
sens = 1.0

def test_net(a, b):
    return a, b

@ms_function()
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

The function call stack (See file 'analyze_fail.dat' for more details. Get instructions about `analyze_fail.dat` at https://www.mindspore.cn/search?inputValue=analyze_fail.dat):

The function call stack:
# 0 In file test.py(17)
    a = grad(test_net)(x, y, sens_i)
        ^
```

<br/>

<font size=3>**Q: 编译时报错“The params of function 'bprop' of Primitive or Cell requires the forward inputs as well as the 'out' and 'dout'”怎么办？**</font>

A: 用户自定义的Cell的反向传播函数 `bprop`，它的输入需要包含正向网络的输入，以及 `out` 和 `dout`，代码样例如下：

```python
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

grad_fn = ops.GradOperation(get_all=True)
net = BpropUserDefinedNet()
x = Tensor(2, mstype.float32)
y = Tensor(6, mstype.float32)
output = grad_fn(net)(x, y)
print(output)
```

执行结果如下：

```text
TypeError: The params of function 'bprop' of Primitive or Cell requires the forward inputs as well as the 'out' and 'dout'.
In file test.py(13)
        def bprop(self, x, y, out):
```

<br/>

<font size=3>**Q: 编译时报错“There isn't any branch that can be evaluated”怎么办？**</font>

A: 当出现There isn't any branch that can be evaluated 时，说明代码中可能出现了无穷递归或者时死循环，导致if条件的每一个分支都无法推导出正确的类型和维度信息。代码样例如下：

```python
import mindspore as ms
from mindspore import ms_function

ZERO = ms.Tensor([0], ms.int32)
ONE = ms.Tensor([1], ms.int32)
@ms_function
def f(x):
    y = ZERO
    if x < 0:
        y = f(x - 3)
    elif x < 3:
        y = x * f(x - 1)
    elif x < 5:
        y = x * f(x - 2)
    else:
        y = f(x - 4)
    z = y + 1
    return z

ms.set_context(mode=ms.GRAPH_MODE)
x = ms.Tensor([5], ms.int32)
f(x)
```

<br/>

<font size=3>**Q: 编译时报错"Exceed function call depth limit 1000"怎么办？**</font>

A: 当出现Exceed function call depth limit 1000 时，说明代码中出现了无穷递归死循环，或者是代码过于复杂，类型推导过程中导致栈深度超过设置的最大深度。
此时可以通过设置 `set_context(max_call_depth = value)` 更改栈的最大深度，并考虑简化代码逻辑或者检查代码中是否存在无穷递归或死循环。
需要注意的是，设置max_call_depth虽然可以改变MindSpore的递归深度，但是可能会超过系统栈的最大深度，进而出现段错误。此时可能还需要设置系统栈深度。

<br/>

<font size=3>**Q: 编译时报错“could not get source code”以及“Mindspore can not compile temporary source code in terminal. Please write source code to a python file and run the file.”是什么原因？**</font>

A: MindSpore编译网络时通过 `inspect.getsourcelines(self.fn)` 获取网络代码所在的文件，如果网络是编辑在命令行中的临时代码，那么会出现如标题所示的报错，需要将网络写在Python文件中去执行才能避免该错误。

<br/>

<font size=3>**Q: 报错提示中的“Corresponding forward node candidate:”或“Corresponding code candidate:”是什么意思？**</font>

A: “Corresponding forward node candidate:”为关联的正向网络中的代码，表示该反向传播算子与该正向代码对应。“Corresponding code candidate:”表示该算子是由这些代码融合而来，其中分符“-”用以区分不同的代码。

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
    In file /home/workspace/mindspore/build/package/mindspore/ops/_grad/grad_nn_ops.py(65)/        dw = filter_grad(dout, x, w_shape)/
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

<font size=3>**Q: 什么是“JIT Fallback”？编译时报错“Should not use Python object in runtime”怎么办？**</font>

A: JIT Fallback从静态图的角度出发考虑静态图和动态图的统一。通过JIT Fallback特性，静态图可以支持尽量多的动态图语法，使得静态图提供接近动态图的语法使用体验。JIT Fallback的环境变量开关是 `DEV_ENV_ENABLE_FALLBACK`，默认使用JIT Fallback。

当出现“Should not use Python object in runtime”和“We suppose all nodes generated by JIT Fallback would not return to outside of graph”的报错信息时，说明静态图模式代码中出现了错误使用语法。JIT Fallback处理不支持的语法表达式时，将会生成相应的节点，并在编译时阶段完成推导和执行，否则这些节点传递到运行时后会引发报错。当前JIT Fallback有条件地支持Graph模式的部分常量场景，编写代码时请参考[静态图语法支持](https://www.mindspore.cn/docs/zh-CN/r1.10/note/static_graph_syntax_support.html)和[JIT Fallback](https://www.mindspore.cn/docs/zh-CN/r1.10/design/jit_fallback.html)。

例如，在调用第三方库NumPy时，JIT Fallback支持使用 `np.add(x, y)` 和 `Tensor(np.add(x, y))` 语法，但MindSpore不支持NumPy类型作为返回值，否则将会出现报错。代码样例如下：

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
    def construct(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        return np.add(x, y)

net = Net()
out = net()
```

执行结果如下：

```text
RuntimeError: Should not use Python object in runtime, node: ValueNode<InterpretedObject> InterpretedObject: '[4 6]'.
Line: In file test.py(11)
        return np.add(x, y)
        ^

We suppose all nodes generated by JIT Fallback not return to outside of graph. For more information about JIT Fallback, please refer to https://www.mindspore.cn/search?inputValue=JIT%20Fallback
```

出现JIT Fallback相关的报错时，请根据[静态图语法支持](https://www.mindspore.cn/docs/zh-CN/r1.10/note/static_graph_syntax_support.html)以及报错代码行，重新检视代码语法并修改。如果需要关闭JIT Fallback，可以设置 `export DEV_ENV_ENABLE_FALLBACK=0`。

<br/>

<font size=3>**Q: 为什么运行代码时屏幕中会出现“Start compiling and it will take a while. Please wait...”和“End compiling.”的打印？**</font>

A: 当需要加速执行时，MindSpore会将Python源码转换成一种基于图表示的函数式IR，并进行相关的优化。这个过程也被称为编译流程。
当出现“Start compiling and it will take a while. Please wait...”的打印时，MindSpore开始了图编译流程；当出现“End compiling.”则表明图编译流程结束。

当前主要有以下两种场景会有该打印：

- 静态图模式下运行网络。
- 动态图下执行被`@ms_function`装饰的函数（例如优化器`nn.Momentum`）。

> 一次任务中有可能会触发多次编译流程。

<font size=3>**Q: 编译时报出告警:“On the Ascend platform, when the return value of the control flow subgraph is parameter, the performance may be degraded. The value of the parameter can be returned to improve the performance. ”，是什么意思？**</font>

A: 由于Ascend平台不能真正返回一个内存地址，导致在整图下沉模式下，对于控制流场景中返回值存在参数的情况，会存在一些问题。为了避免出现问题，会对这种场景切换到统一运行时模式，从整图下沉模式切换到统一运行时模式，网络性能可能会劣化。如果控制流子图的返回值仅使用参数的值，可以通过参数的value接口获取参数的值，从而避免模式切换导致的性能劣化。

例如下面的用例，在网络“Net”中仅使用“InnerNet”中的“self.param1”和“self.param2”的值，没有使用参数的属性，所以可以使用value接口来避免模式切换导致的性能劣化。

```python
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor, Parameter

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

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
