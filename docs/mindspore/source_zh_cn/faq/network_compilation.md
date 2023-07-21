# 网络编译

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.8/docs/mindspore/source_zh_cn/faq/network_compilation.md)

<font size=3>**Q: 编译时报错“'self.xx' should be defined in the class '__init__' function.”怎么办？**</font>

A: 如果在`construct`函数里，想对类成员`self.xx`赋值，那么`self.xx`必须已经在`__init__`函数中被定义为[Parameter](<https://www.mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.Parameter.html>)类型，其他类型则不支持。局部变量`xx`不受这个限制。

<br/>

<font size=3>**Q: 编译时报错“This comparator 'AnyValue' is not supported. For statement 'is', only support compare with 'None', 'False' or 'True'”怎么办？**</font>

A: 对于语法`is` 或 `is not`而言，当前`MindSpore`仅支持与`True`、`False`和`None`的比较。暂不支持其他类型，如字符串等。

<br/>

<font size=3>**Q: 编译时报错“MindSpore does not support comparison with operators more than one now, ops size =2”怎么办？**</font>

A: 对于比较语句，`MindSpore`最多支持一个操作数。例如不支持语句`1 < x < 3`，请使用`1 < x and x < 3`的方式代替。

<br/>

<font size=3>**Q: 编译时报错“TypeError: The function construct need 1 positional argument and 0 default argument, but provided 2”怎么办？**</font>

A: 网络的实例被调用时，会执行`construct`方法，然后会检查`construct`方法需要的参数个数和实际传入的参数个数，如果不一致则会抛出以上异常。
请检查脚本中调用网络实例时传入的参数个数，和定义的网络中`construct`函数需要的参数个数是否一致。

<br/>

<font size=3>**Q: 编译时报错“Type Join Failed”或“Shape Join Failed”怎么办？**</font>

A: 在前端编译的推理阶段，会对节点的抽象类型(包含`type`、`shape`等)进行推导，常见抽象类型包括`AbstractScalar`、`AbstractTensor`、`AbstractFunction`、`AbstractTuple`、`AbstractList`等。在一些场景比如多分支场景，会对不同分支返回值的抽象类型进行`join`合并，推导出返回结果的抽象类型。如果抽象类型不匹配，或者`type`/`shape`不一致，则会抛出以上异常。

当出现类似“Type Join Failed: dtype1 = Float32, dtype2 = Float16”的报错时，说明数据类型不一致，导致抽象类型合并失败。根据提供的数据类型和代码行信息，可以快速定位出错范围。此外，报错信息中提供了具体的抽象类型信息、节点信息，可以通过`analyze_fail.dat`文件查看MindIR信息，定位解决问题。关于MindIR的具体介绍，可以参考[MindSpore IR（MindIR）](https://www.mindspore.cn/docs/zh-CN/r1.8/design/mindir.html)。代码样例如下:

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn

ms.set_context(mode=GRAPH_MODE)
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

执行结果如下:

```text
TypeError: Cannot join the return values of different branches, perhaps you need to make them equal.
Type Join Failed: dtype1 = Float32, dtype2 = Float16.
For more details, please refer to the FAQ at https://www.mindspore.cn
The abstract type of the return value of the current branch is AbstractTensor(shape: (2, 3, 4, 5), element: AbstractScalar(Type: Float16, Value: AnyValue, Shape: NoShape), value_ptr: 0x55b9f289d090, value: AnyValue), and that of the previous branch is AbstractTensor(shape: (2, 3, 4, 5), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x55b9f289d090, value: AnyValue).
The node is construct.6:[CNode]13{[0]: construct.6:[CNode]12{[0]: ValueNode<Primitive> Switch, [1]: [CNode]11, [2]: ValueNode<FuncGraph> ✓construct.4, [3]: ValueNode<FuncGraph> ✗construct.5}}, true branch: ✓construct.4, false branch: ✗construct.5
The function call stack:
In file test.py(14)/        if a > b:

The function call stack (See file 'analyze_fail.dat' for more details):
# 0 In file test.py(14)
        if a > b:
        ^
```

当出现类似“Shape Join Failed: shape1 = (2, 3, 4, 5), shape2 = ()”的报错时，说明`shape`不一致，导致抽象类型合并失败。代码样例如下:

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn

ms.set_context(mode=GRAPH_MODE)
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

执行结果如下:

```text
ValueError: Cannot join the return values of different branches, perhaps you need to make them equal.
Shape Join Failed: shape1 = (2, 3, 4, 5), shape2 = ().
For more details, please refer to the FAQ at https://www.mindspore.cn
The abstract type of the return value of the current branch is AbstractTensor(shape: (), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x55658aa9b090, value: AnyValue), and that of the previous branch is AbstractTensor(shape: (2, 3, 4, 5), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x55658aa9b090, value: AnyValue).
The node is construct.6:[CNode]13{[0]: construct.6:[CNode]12{[0]: ValueNode<Primitive> Switch, [1]: [CNode]11, [2]: ValueNode<FuncGraph> ✓construct.4, [3]: ValueNode<FuncGraph> ✗construct.5}}, true branch: ✓construct.4, false branch: ✗construct.5
The function call stack:
In file test.py(14)/        if a > b:

The function call stack (See file 'analyze_fail.dat' for more details):
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
For more details, please refer to the FAQ at https://www.mindspore.cn.
This: AbstractTensor(shape: (1), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x56458a351ad0, value: Tensor(shape=[1], dtype=Float32, value=[ 1.00000000e+00])), other: AbstractTuple{element[0]: AbstractTensor(shape: (1), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x564583e3fa90, value: Tensor(shape=[1], dtype=Float32, value=[ 1.00000000e+00])), element[1]: AbstractTensor(shape: (1), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x564583cb00b0, value: Tensor(shape=[1], dtype=Float32, value=[ 2.00000000e+00])), sequence_nodes: {test_net.3:[CNode]4{[0]: ValueNode<PrimitivePy> MakeTuple, [1]: a, [2]: b}, elements_use_flags: {ptr: 0x5645cbc500c0, value: [const vector][1, 1]}}}
The function call stack (See file 'analyze_fail.dat' for more details):
# 0 In file test.py(16)
    a = grad(test_net)(x, y, sens_i)
        ^
```

<br/>

<font size=3>**Q: 编译时报错“The params of function 'bprop' of Primitive or Cell requires the forward inputs as well as the 'out' and 'dout'”怎么办？**</font>

A: 用户自定义的Cell的反向传播函数`bprop`，它的输入需要包含正向网络的输入，以及`out`和`dout`，例如：

```python
class BpropUserDefinedNet(nn.Cell):
        def __init__(self):
            super(BpropUserDefinedNet, self).__init__()
            self.zeros_like = P.ZerosLike()

        def construct(self, x, y):
            return x + y

        def bprop(self, x, y, out, dout):
            return self.zeros_like(out), self.zeros_like(out)
```

<br/>

<font size=3>**Q: 编译时报错“There isn't any branch that can be evaluated”怎么办？**</font>

当出现There isn't any branch that can be evaluated 时，说明代码中可能出现了无穷递归或者时死循环，导致if条件的每一个分支都无法推导出正确的类型和维度信息。
例如代码

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

def test_endless():
    ms.set_context(mode=ms.GRAPH_MODE)
    x = ms.Tensor([5], ms.int32)
    f(x)

```

其中f(x)由于每一个if分支都没办法推导出正确的类型信息导致失败。

<br/>

<font size=3>**Q: 编译时报错"Exceed function call depth limit 1000"怎么办？**</font>

当出现Exceed function call depth limit 1000 时，说明代码中出现了无穷递归死循环，或者是代码过于复杂，类型推导过程中导致栈深度超过设置的最大深度。
此时可以通过设置set_context(max_call_depth = value)这样的方式更改栈的最大深度，并考虑简化代码逻辑或者检查代码中是否存在无穷递归或死循环。
此外设置max_call_depth = value 虽然可以改变MindSpore的递归深度，但是此时也可能会超过系统栈的最大深度而出现段错误。此时可能还需要设置将系统栈深度进行设置。

<br/>

<font size=3>**Q: 编译时报错“could not get source code”以及“Mindspore can not compile temporary source code in terminal. Please write source code to a python file and run the file.”是什么原因？**</font>

A: MindSpore编译网络时通过`inspect.getsourcelines(self.fn)`获取网络代码所在的文件，如果网络是编辑在命令行中的临时代码，那么会出现如标题所示的报错，需要将网络写在Python文件中去执行才能避免该错误。

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

A: JIT Fallback从静态图的角度出发考虑静态图和动态图的统一。通过JIT Fallback特性，静态图可以支持尽量多的动态图语法，使得静态图提供接近动态图的语法使用体验。JIT Fallback的环境变量开关是`DEV_ENV_ENABLE_FALLBACK`，默认使用JIT Fallback。

当出现“Should not use Python object in runtime”和“We suppose all nodes generated by JIT Fallback would not return to outside of graph”的报错信息时，说明静态图模式代码中出现了错误使用语法。JIT Fallback处理不支持的语法表达式时，将会生成相应的节点，并在编译时阶段完成推导和执行，否则这些节点传递到运行时后会引发报错。当前JIT Fallback有条件地支持Graph模式的部分常量场景，同时需要符合MindSpore的编程语法，编写代码时请参考[静态图语法支持](https://www.mindspore.cn/docs/zh-CN/r1.8/note/static_graph_syntax_support.html)。

例如，在调用第三方库NumPy时，JIT Fallback支持`np.add(x, y)`和`Tensor(np.add(x, y))`的语法，但MindSpore不支持NumPy类型的返回值，将会出现报错。代码样例如下：

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
    def construct(self, x, y):
        out = np.add(x, y)
        return out

net = Net()
out = net(1, 1)
```

执行结果如下：

```text
RuntimeError: mindspore/ccsrc/pipeline/jit/validator.cc:139 ValidateValueNode] Should not use Python object in runtime, node: ValueNode<InterpretedObject> InterpretedObject: '2'

We suppose all nodes generated by JIT Fallback not return to outside of graph.

# In file test.py(9)
        out = np.add(x, y)
        ^
```

出现JIT Fallback相关的报错时，请根据[静态图语法支持](https://www.mindspore.cn/docs/zh-CN/r1.8/note/static_graph_syntax_support.html)以及报错代码行，重新检视代码语法并修改。如果需要关闭JIT Fallback，可以设置`export DEV_ENV_ENABLE_FALLBACK=0`。

<font size=3>**Q: 编译时报错“Operator[AddN]  input(kNumberTypeBool,kNumberTypeBool) output(kNumberTypeBool) is not support. This error means the current input type is not supported, please refer to the MindSpore doc for supported types.”怎么办？**</font>
A: MindSpore当前对数据类型为bool的Tensor[后续简称Tensor(bool)]支持能力较弱，仅有少量算子支持Tensor(bool)类型的数据参与运算。若在正向图中使用了支持Tensor(bool)类型的算子且正向图语法正确，由于反向图求解全导数会引入`AddN`，`AddN`不支持Tensor(bool)类型，反向图运行就会抛出该异常。

例如代码：

```python
from mindspore import ops, ms_function
import mindspore as ms

ms.set_context(save_graphs=True, save_graphs_path='graph_path')

@ms_function
def test_logic(x, y):
    z = x and y
    return z and x

x = ms.Tensor(True, ms.bool_)
y = ms.Tensor(True, ms.bool_)
grad = ops.GradOperation(get_all=True)
grad_net = grad(test_logic)
out = grad_net(x, y)
```

上述代码正向处理可以用公式表示为：`r = f(z, x), z = z(x, y)` 对应的全导数公式为：`dr/dx = df/dz * dz/dx + df/dx`， 函数`f(z,x)`和`z(x,y)`均为逻辑运算符`and`； 正向图中的`and`算子支持Tensor(bool)类型，反向图求全导数时引入的`AddN`不支持Tensor(bool) 类型， 且该错误无法对应到具体的正向代码行。

执行结果如下：

```text
Traceback (most recent call last):
  File "grad_fail.py", line 14, in <module>
    out = grad_net(x, y)
  File "/usr/local/python3.7/lib/python3.7/site-packages/mindspore/common/api.py", line 307, in staging_specialize
    out = _MindsporeFunctionExecutor(func, ms_create_time, input_signature, process_obj)(*args)
  File "/usr/local/python3.7/lib/python3.7/site-packages/mindspore/common/api.py", line 79, in wrapper
    results = fn(*arg, **kwargs)
  File "/usr/local/python3.7/lib/python3.7/site-packages/mindspore/common/api.py", line 221, in __call__
    phase = self.compile(args_list, arg_names, parse_method)
  File "/usr/local/python3.7/lib/python3.7/site-packages/mindspore/common/api.py", line 195, in compile
    self.enable_tuple_broaden)
TypeError: mindspore/ccsrc/runtime/device/cpu/kernel_select_cpu.cc:235 KernelNotSupportException] Operator[AddN]  input(kNumberTypeBool,kNumberTypeBool) output(kNumberTypeBool) is not support. This error means the current input type is not supported, please refer to the MindSpore doc for supported types.
Trace:
In file /usr/local/python3.7/lib/python3.7/site-packages/mindspore/ops/composite/multitype_ops/add_impl.py(287)/    return F.addn((x, y))/
```

若遇到这类问题请去除对Tensor(bool)类型的使用，本例中将Tensor(bool)替换为bool即可解决问题。

<br/>

<font size=3>**Q: 编译时报错“Side Effect Invalid: found unsupported syntax in graph mode, those side effect codes would be ignored:”怎么办？**</font>

A: 如果在`Cell.construct`或者`ms_function`函数以及其调用的子函数里，使用了副作用算子，则要求所在函数不能直接返回常量值，包括最终返回值经过推导是常量的情况。由于函数返回常量时，编译器会优先把常量值以外的操作优化掉，导致其它操作看起来无效。对于非副作用的算子操作，忽略掉一般不会影响最终结果的正确性。但是如果包含了副作用算子的操作，忽略掉副作用算子往往跟用户期望相左。因此，对于出现函数返回值为常量，同时又包含副作用算子操作的情况，编译器会抛出异常，提示用户代码执行有可能无法符合预期，需要调整代码实现。

例如代码：

```python
from mindspore.nn import Cell

class Demo(Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        print('print here...')
        y = x[1]
        y[1] = 9
        return y

x = [[1, 2, 3, 4], [5, 6, 7, 8]]
net = Demo()
output = net(x)
print(output)
```

上述代码`y`经过推导后是一个常量值，整个函数可以被优化为直接返回常量值。除此以外的操作全部被优化掉，包括`print('print here...')`也会在编译时被忽略掉。由于`print`算子是副作用算子，其行为被删除后不符合预期，因此编译器会抛出错误提示用户。

执行结果如下：

```text
Traceback (most recent call last):
  File "test_print_op.py", line 20, in <module>
    output = net(x)
  File "/usr/local/python3.7/lib/python3.7/site-packages/mindspore/nn/cell.py", line 586, in __call__
    out = self.compile_and_run(*args)
  File "/usr/local/python3.7/lib/python3.7/site-packages/mindspore/nn/cell.py", line 964, in compile_and_run
    self.compile(*inputs)
  File "/usr/local/python3.7/lib/python3.7/site-packages/mindspore/nn/cell.py", line 937, in compile
    _cell_graph_executor.compile(self, *inputs, phase=self.phase, auto_parallel_mode=self._auto_parallel_mode)
  File "/usr/local/python3.7/lib/python3.7/site-packages/mindspore/common/api.py", line 1086, in compile
    result = self._graph_executor.compile(obj, args_list, phase, self._use_vm_mode())
RuntimeError: mindspore/ccsrc/pipeline/jit/static_analysis/evaluator.cc:127 CheckSideEffectNodes] Side Effect Invalid: Found unsupported syntax in graph mode, those side effect codes would be ignored:
-----
# No. 1:
In file test_print_op.py(11)
         print('print here...')
         ^

-----

If a function return a const value or inferred const value, the side effect node would be ignored.
So the codes may not run as the user's expectation, please fix it.

In this case, the const value '[[1, 2, 3, 4], [5, 6, 7, 8]]' returns:
In file test_print_op.py(10)
     def construct(self, a):
     ^

For more information about this issue, please refer to https://www.mindspore.cn/search?inputValue=Side%20Effect%20Invalid
```

若遇到这类问题请去除副作用算子的调用，或者修改函数返回值不返回常量。

