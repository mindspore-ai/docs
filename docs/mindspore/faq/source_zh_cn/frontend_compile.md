# 前端编译

`Linux` `Windows` `Ascend` `GPU` `CPU` `环境准备` `初级` `中级`

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/faq/source_zh_cn/frontend_compile.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

<font size=3>**Q：运行时报错“Create python object \`<class 'mindspore.common.tensor.Tensor'>\` failed, only support create Cell or Primitive object.”怎么办？**</font>

A：当前在图模式下，`construct`函数(或`@ms_function`装饰器修饰的函数)仅支持构造`Cell`和`Primitive object`，不支持构造`Tensor`，即不支持语法`x = Tensor(args...)`。

如果是常量`Tensor`，请在`__init__`函数中定义。如果不是常量`Tensor`，可以通过`@constexpr`装饰器修饰函数，在函数里生成`Tensor`。

关于`@constexpr`的用法可参考：<https://www.mindspore.cn/doc/api_python/zh-CN/r1.3/mindspore/ops/mindspore.ops.constexpr.html>。

对于网络中需要用到的常量`Tensor`，可以作为网络的属性，在`init`的时候定义，即`self.x = Tensor(args...)`，然后在`construct`函数(或`@ms_function`装饰器修饰的函数)里使用。

如下示例，通过`@constexpr`生成一个`shape = (3, 4), dtype = int64`的`Tensor`。

```python
@constexpr
def generate_tensor():
    return Tensor(np.ones((3, 4)))
```

<br/>

<font size=3>**Q：运行时报错“'self.xx' should be defined in the class '__init__' function.”怎么办？**</font>

A：如果在`construct`函数里，想对类成员`self.xx`赋值，那么`self.xx`必须已经在`__init__`函数中被定义为[`Parameter`](<https://www.mindspore.cn/docs/api/zh-CN/r1.3/api_python/mindspore.html?highlight=parameter#mindspore.Parameter>)类型，其他类型则不支持。局部变量`xx`不受这个限制。

<br/>

<font size=3>**Q：运行时报错“This comparator 'AnyValue' is not supported. For statement 'is', only support compare with 'None', 'False' or 'True'”怎么办？**</font>

A：对于语法`is` 或 `is not`而言，当前`MindSpore`仅支持与`True`、`False`和`None`的比较。暂不支持其他类型，如字符串等。

<br/>

<font size=3>**Q：运行时报错“MindSpore does not support comparison with operators more than one now, ops size =2”怎么办？**</font>

A：对于比较语句，`MindSpore`最多支持一个操作数。例如不支持语句`1 < x < 3`，请使用`1 < x and x < 3`的方式代替。

<br/>

<font size=3>**Q：运行时报错“TypeError: The function construct need 1 positional argument and 0 default argument, but provided 2”怎么办？**</font>

A：网络的实例被调用时，会执行`construct`方法，然后会检查`construct`方法需要的参数个数和实际传入的参数个数，如果不一致则会抛出以上异常。
请检查脚本中调用网络实例时传入的参数个数，和定义的网络中`construct`函数需要的参数个数是否一致。

<br/>

<font size=3>**Q：运行时报错“Type Join Failed”或“Shape Join Failed”怎么办？**</font>

A：在前端编译的推理阶段，会对节点的抽象类型(包含`type`、`shape`等)进行推导，常见抽象类型包括`AbstractScalar`、`AbstractTensor`、`AbstractFunction`、`AbstractTuple`、`AbstractList`等。在一些场景比如多分支场景，会对不同分支返回值的抽象类型进行`join`合并，推导出返回结果的抽象类型。如果抽象类型不匹配，或者`type`/`shape`不一致，则会抛出以上异常。

当出现类似“Type Join Failed: dtype1 = Float32, dtype2 = Float16”的报错时，说明数据类型不一致，导致抽象类型合并失败。根据提供的数据类型和代码行信息，可以快速定位出错范围。此外，报错信息中提供了具体的抽象类型信息、节点信息，可以通过`analyze_fail.dat`文件查看MindIR信息，定位解决问题。关于MindIR的具体介绍，可以参考[MindSpore IR（MindIR）](https://www.mindspore.cn/docs/note/zh-CN/r1.3/design/mindspore/mindir.html)。代码样例如下：

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn, Tensor, context

context.set_context(mode=context.GRAPH_MODE)
class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = ops.ReLU()
        self.cast = ops.Cast()

    def construct(self, x, a, b):
        if a > b:
            return self.relu(x)
        else:
            return self.cast(self.relu(x), ms.float16)

input_x = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
input_a = Tensor(2, ms.float32)
input_b = Tensor(6, ms.float32)
net = Net()
out_me = net(input_x, input_a, input_b)
```

执行结果如下：

```text
TypeError: The return values of different branches do not match. Type Join Failed: dtype1 = Float32, dtype2 = Float16. The abstract type of the return value of the current branch is AbstractTensor(shape: (2, 3, 4, 5), element: AbstractScalar(Type: Float16, Value: AnyValue, Shape: NoShape), value_ptr: 0x32ed00e0, value: AnyValue), and that of the previous branch is AbstractTensor(shape: (2, 3, 4, 5), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x32ed00e0, value: AnyValue). Please check the node construct.4:[CNode]5{[0]: [CNode]6}, true branch: ✓construct.2, false branch: ✗construct.3. trace:
In file test_type_join_failed.py(14)/        if a > b:/

The function call stack (See file 'analyze_fail.dat' for more details):
# 0 In file test_type_join_failed.py(14)
        if a > b:
```

当出现类似“Shape Join Failed: shape1 = (2, 3, 4, 5), shape2 = ()”的报错时，说明`shape`不一致，导致抽象类型合并失败。代码样例如下：

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn, Tensor, context

context.set_context(mode=context.GRAPH_MODE)
class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = ops.ReLU()
        self.reducesum = ops.ReduceSum()

    def construct(self, x, a, b):
        if a > b:
            return self.relu(x)
        else:
            return self.reducesum(x)

input_x = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
input_a = Tensor(2, ms.float32)
input_b = Tensor(6, ms.float32)
net = Net()
out = net(input_x, input_a, input_b)
```

执行结果如下：

```text
ValueError: The return values of different branches do not match. Shape Join Failed: shape1 = (2, 3, 4, 5), shape2 = (). The abstract type of the return value of the current branch is AbstractTensor(shape: (), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x239b5120, value: AnyValue), and that of the previous branch is AbstractTensor(shape: (2, 3, 4, 5), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x239b5120, value: AnyValue). Please check the node construct.4:[CNode]5{[0]: [CNode]6}, true branch: ✓construct.2, false branch: ✗construct.3. trace:
In file test_shape_join_failed.py(14)/        if a > b:/

The function call stack (See file 'analyze_fail.dat' for more details):
# 0 In file test_shape_join_failed.py(14)
        if a > b:
```

当出现如“Type Join Failed: abstract type AbstractTensor can not join with AbstractTuple”的报错时，说明这两种抽象类型无法匹配，需要根据提供的代码行等报错信息，重新检视代码并修改。

<br/>
