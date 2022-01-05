# 使用流程控制语句

`Ascend` `GPU` `CPU` `模型开发`

<!-- TOC -->

- [使用流程控制语句](#使用流程控制语句)
    - [概述](#概述)
    - [使用if语句](#使用if语句)
        - [使用条件为变量的if语句](#使用条件为变量的if语句)
        - [使用条件为常量的if语句](#使用条件为常量的if语句)
    - [使用for语句](#使用for语句)
    - [使用while语句](#使用while语句)
        - [使用条件为常量的while语句](#使用条件为常量的while语句)
        - [使用条件为变量的while语句](#使用条件为变量的while语句)
    - [使用while语句等价替换for语句](#使用while语句等价替换for语句)
        - [简单示例](#简单示例)
        - [循环体内有权重](#循环体内有权重)
    - [约束](#约束)
        - [副作用约束](#副作用约束)
        - [死循环约束](#死循环约束)

<!-- TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/control_flow.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

MindSpore流程控制语句的使用与Python原生语法相似，尤其是在`PYNATIVE_MODE`模式下， 与Python原生语法基本一致，但是在`GRAPH_MODE`模式下，会有一些特殊的约束。鉴于以上原因，下述关于流程控制语句的使用指导均是指在`GRAPH_MODE`模式下运行。

使用流程控制语句时，MindSpore会依据条件是否为变量来决策是否在网络中生成控制流算子，只有条件为变量时网络中才会生成控制流算子。如果条件表达式的结果需要在图编译时确定，则条件为常量，否则条件为变量。需要特殊说明的是，当网络中存在控制流算子时，网络会被切分成多个执行子图，子图间进行流程跳转和数据传递会引起一定的性能损耗。

条件为变量的场景：

- 条件表达式中存在Tensor或者元素为Tensor类型的List、Tuple、Dict，并且条件表达式的结果受Tensor的值影响。

常见的变量条件：

- `(x < y).all()`，`x`或`y`为算子输出。此时条件是否为真取决于算子输出`x`、`y`，而算子输出是图在各个step执行时才能确定。

- `x in list`，`x`为算子输出。

条件为常量的场景：

- 条件表达式中不存在Tensor和元素为Tensor类型的List、Tuple、Dict。

- 条件表达式中存在Tensor或者元素为Tensor类型的List、Tuple、Dict，但是表达式结果不受Tensor的值影响。

常见的常量条件：

- `self.flag`，`self.flag`为标量。此处`self.flag`为一个bool类型标量，其值在构建Cell对象时已确定，因此该条件是一个常量条件。

- `x + 1 < 10`，`x`为标量。此处`x + 1`的值在构建Cell对象时是不确定的，但是在图编译时MindSpore会计算所有标量表达式的结果，因此该表达式的值也是在编译期确定的，该条件为常量条件。

- `len(my_list) < 10`，`my_list为`元素是Tensor类型的List对象。虽然该条件表达式包含Tensor，但是表达式结果不受Tensor的值影响，只与`my_list`中Tensor的数量有关，因此该条件为常量条件。

- `for i in range(0,10)`，`i`为标量，潜在的条件表达式`i < 10`为常量条件。

## 使用if语句

使用`if`语句需要注意在条件为变量时，在不同分支中的同一变量名应被赋予相同的数据类型。同时，网络最终生成的执行图的子图数量与`if`的数量成正比关系，过多的`if`会产生较大的控制流算子性能开销和子图间数据传递性能开销。

### 使用条件为变量的if语句

在例1中，`out`在true分支被赋值为[0]，在false分支被赋值为[0, 1]，条件`x < y`为变量，因此在`out = out + 1`这一句无法确定输入`out`的数据类型，会导致图编译出现异常。

例1：

```python
import numpy as np
from mindspore import context
from mindspore import Tensor, nn
from mindspore import dtype as ms

class SingleIfNet(nn.Cell):
    def construct(self, x, y, z):
        if x < y:
            out = x
        else:
            out = z
        out = out + 1
        return out

forward_net = SingleIfNet()
x = Tensor(np.array(0), dtype=ms.int32)
y = Tensor(np.array(1), dtype=ms.int32)
z = Tensor(np.array([0, 1]), dtype=ms.int32)
output = forward_net(x, y, z)
```

例1报错信息如下：

```text
ValueError: mindspore/ccsrc/pipeline/jit/static_analysis/static_analysis.cc:734 ProcessEvalResults] The return values of different branches do not match. Shape Join Failed: shape1 = (2), shape2 = ()..
```

### 使用条件为常量的if语句

在例2中，`out`在true分支被赋值为标量0，在false分支被赋值为[0, 1]，`x`和`y`均为标量，条件`x < y + 1`为常量，图编译阶段可以确定是走true分支，因此网络中只存在true分支的内容并且无控制流算子，`out = out + 1`的输入`out`数据类型是确定的，因此该用例可正常执行。

例2：

```python
import numpy as np
from mindspore import context
from mindspore import Tensor, nn
from mindspore import dtype as ms

class SingleIfNet(nn.Cell):
    def construct(self, z):
        x = 0
        y = 1
        if x < y + 1:
            out = x
        else:
            out = z
        out = out + 1
        return out

forward_net = SingleIfNet()
z = Tensor(np.array([0, 1]), dtype=ms.int32)
output = forward_net(z)
```

## 使用for语句

`for`语句会展开循环体内容。在例3中，`for`循环了3次，与例4最终生成的执行图结构是完全一致的，因此使用`for`语句的网络的子图数量、算子数量取决于`for`的迭代次数，算子数量过多或者子图过多会导致硬件资源受限。`for`语句导致出现子图过多的问题时，可参考`while`写作方式，尝试将`for`语句等价转换为条件是变量的`while`语句，示例见本文[使用while语句等价替换for语句](#whilefor)。

例3：

```python
import numpy as np
from mindspore import context
from mindspore import Tensor, nn
from mindspore import dtype as ms

class IfInForNet(nn.Cell):
    def construct(self, x, y):
        out = 0
        for i in range(0,3):
            if x + i < y :
                out = out + x
            else:
                out = out + y
            out = out + 1
        return out

forward_net = IfInForNet()
x = Tensor(np.array(0), dtype=ms.int32)
y = Tensor(np.array(1), dtype=ms.int32)
output = forward_net(x, y)
```

例4：

```python
import numpy as np
from mindspore import context
from mindspore import Tensor, nn
from mindspore import dtype as ms

class IfInForNet(nn.Cell):
    def construct(self, x, y):
        out = 0
        #######cycle 0
        if x + 0 < y :
            out = out + x
        else:
            out = out + y
        out = out + 1

         #######cycle 1
        if x + 1 < y :
            out = out + x
        else:
            out = out + y
        out = out + 1

         #######cycle 2
        if x + 2 < y :
            out = out + x
        else:
            out = out + y
        out = out + 1
        return out

forward_net = IfInForNet()
x = Tensor(np.array(0), dtype=ms.int32)
y = Tensor(np.array(1), dtype=ms.int32)
output = forward_net(x, y)
```

## 使用while语句

`while`语句相比`for`语句更为灵活。当`while`的条件为常量时，`while`对循环体的处理和`for`类似，会展开循环体里的内容。当`while`的条件为变量时，`while`不会展开循环体里的内容，则会在执行图产生控制流算子。

### 使用条件为常量的while语句

如例5所示，条件`i < 3`为常量， `while`的循环体的内容会被复制3份，因此最终生成的执行图和例4完全一致。`while`语句条件为常量时，算子数量和子图数量与while的循环次数成正比，算子数量过多或者子图过多会导致硬件资源受限。

例5：

```python
import numpy as np
from mindspore import context
from mindspore import Tensor, nn
from mindspore import dtype as ms

class IfInWhileNet(nn.Cell):
    def construct(self, x, y):
        i = 0
        out = x
        while i < 3:
            if x + i < y :
                out = out + x
            else:
                out = out + y
            out = out + 1
            i = i + 1
        return out

forward_net = IfInWhileNet()
x = Tensor(np.array(0), dtype=ms.int32)
y = Tensor(np.array(1), dtype=ms.int32)
output = forward_net(x, y)
```

### 使用条件为变量的while语句

如例6所示，把`while`条件变更为变量，`while`按照不展开处理，最终网络输出结果和例5一致，但执行图的结构不一致。例6不展开的执行图，有较少的算子和较多的子图，会使用较短的编译时间和占用较小的设备内存，但是会产生额外的控制流算子执行和子图间数据传递引起的性能开销。

例6：

```python
import numpy as np
from mindspore import context
from mindspore import Tensor, nn
from mindspore import dtype as ms

class IfInWhileNet(nn.Cell):
    def construct(self, x, y, i):
        out = x
        while i < 3:
            if x + i < y :
                out = out + x
            else:
                out = out + y
            out = out + 1
            i = i + 1
        return out

forward_net = IfInWhileNet()
i = Tensor(np.array(0), dtype=ms.int32)
x = Tensor(np.array(0), dtype=ms.int32)
y = Tensor(np.array(1), dtype=ms.int32)
output = forward_net(x, y, i)
```

当`while`的条件为变量时，`while`循环体不能展开，`while`循环体内的表达式都是在各个step运行时计算，因此循环体内部不能出现标量、List、Tuple等非Tensor类型的计算操作，这些类型计算操作需要在图编译时期完成，与`while`在执行期进行计算的机制是矛盾的。如例7所示，条件`i < 3`是变量条件，但是循环体内部存在`j = j + 1`的标量计算操作，最终会导致图编译出错。

例7：

```python
import numpy as np
from mindspore import context
from mindspore import Tensor, nn
from mindspore import dtype as ms

class IfInWhileNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.nums = [1, 2, 3]

    def construct(self, x, y, i):
        j = 0
        out = x
        while i < 3:
            if x + i < y :
                out = out + x
            else:
                out = out + y
            out = out + self.nums[j]
            i = i + 1
            j = j + 1
        return out

forward_net = IfInWhileNet()
i = Tensor(np.array(0), dtype=ms.int32)
x = Tensor(np.array(0), dtype=ms.int32)
y = Tensor(np.array(1), dtype=ms.int32)
output = forward_net(x, y, i)
```

例7报错信息如下：

```text
IndexError: mindspore/core/abstract/prim_structures.cc:178 InferTupleOrListGetItem] list_getitem evaluator index should be in range[-3, 3), but got 3.
```

`while`条件为变量时，循环体内部不能更改算子的输入shape。因为MindSpore要求网络的同一个算子的输入shape在图编译时是确定的，而在`while`的循环体内部改变算子输入shape的操作是在图执行时生效，两者是矛盾的。如例8所示，条件`i < 3`为变量条件，`while`不展开，循环体内部的`ExpandDims`算子会改变表达式`out = out + 1`在下一轮循环的输入shape，会导致图编译出错。

例8：

```python
import numpy as np
from mindspore import context
from mindspore import Tensor, nn
from mindspore.common import dtype as ms
from mindspore import ops

class IfInWhileNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.expand_dims = ops.ExpandDims()

    def construct(self, x, y, i):
        out = x
        while i < 3:
            if x + i < y :
                out = out + x
            else:
                out = out + y
            out = out + 1
            out = self.expand_dims(out, -1)
            i = i + 1
        return out

forward_net = IfInWhileNet()
i = Tensor(np.array(0), dtype=ms.int32)
x = Tensor(np.array(0), dtype=ms.int32)
y = Tensor(np.array(1), dtype=ms.int32)
output = forward_net(x, y, i)
```

例8报错信息如下：

```text
ValueError: mindspore/ccsrc/pipeline/jit/static_analysis/static_analysis.cc:734 ProcessEvalResults] The return values of different branches do not match. Shape Join Failed: shape1 = (1, 1), shape2 = (1)..
```

## 使用while语句等价替换for语句

`for`语句会展开循环体内容，带来执行性能提升的同时，也会带来编译问题，如增加编译时间、函数调用深度溢出、不同的权重被共享等等。为了解决该类问题，需要将`for`语句等价转换为`while`语句。

### 简单示例

如例9所示，实现了通过加法来完成两数相乘的计算。

例9：

```python
from mindspore import Tensor
from mindspore import ms_function

one = Tensor(1)
zero = Tensor(0)

@ms_function
def mul_by_for(x, y):
    r = zero
    for _ in range(y):
        r = r + x
    return r


a = Tensor(2)
b = 1000
out = mul_by_for(a, b)
print(out)
```

执行出错，报错信息如下所示：

```text
RuntimeError: mindspore/ccsrc/pipeline/jit/static_analysis/evaluator.cc:201 Eval] Exceed function call depth limit 1000, (function call depth: 1001, simulate call depth: 998).
It's always happened with complex construction of code or infinite recursion or loop.
Please check the code if it's has the infinite recursion or call 'context.set_context(max_call_depth=value)' to adjust this value.
If max_call_depth is set larger, the system max stack depth should be set larger too to avoid stack overflow.
```

这是因为`for`语句内的循环体会被展开。该例子中循环了1000次，展开后的子图过多，函数调用深度超出了允许的限制。为了不让子图数量展开太多，可等价替换成`while`实现，实现方式如例10所示。

例10：

```python
from mindspore import Tensor
from mindspore import ms_function

one = Tensor(1)
zero = Tensor(0)

@ms_function
def mul_by_while(x, y):
    y = Tensor(y)
    r = zero
    while y > 0:
        y = y - one
        r = r + x
    return r

a = Tensor(2)
b = 1000
out = mul_by_while(a, b)
print(out)
```

执行结果正确，如下：

```text
2000
```

### 循环体内有权重

如例11所示，实现了`1+2+3`的计算，同时用权重保存循环中每次迭代的计数器的值。

例11：

```python
import mindspore
from mindspore import nn, Tensor
from mindspore import Parameter

class AddIndexNet(nn.Cell):
    def __init__(self, index):
        super(AddIndexNet, self).__init__()
        self.weight = Parameter(Tensor(0, mindspore.float32), name="weight")
        self.idx = Tensor(index)

    def construct(self, x):
        self.weight = self.weight + self.idx
        x = x + self.weight
        return x

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.idx = Tensor(0)
        self.block_nums = 3
        self.nets = []
        for i in range(self.block_nums):
            self.nets.append(AddIndexNet(i + 1))

    def construct(self, x):
        for i in range(self.block_nums):
            x = self.nets[i](x)
        return x

x = Tensor(0, mindspore.float32)
net = Net()
out = net(x)
print(out)
```

执行结果：

```text
10.0
```

结果和预期不符，预期结果应该是`1.0+2.0+3.0=6.0`。这是因为循环体展开后，不同迭代中的算子共享了同一个权重（即self.weight），导致每次迭代，更新的都是同一个权重。

为了解决该问题，我们使用`while`语句等价替换该`for`语句，如例12所示。

例12：

```python
import numpy as np
import mindspore
from mindspore import nn, Tensor, ops
from mindspore import Parameter

class AddIndexNet(nn.Cell):
    def __init__(self, block_nums):
        super(AddIndexNet, self).__init__()
        self.weights = Parameter(Tensor(np.zeros((block_nums, 1)), mindspore.float32), name="weights")
        self.gather = ops.Gather()

    def construct(self, x, index):
        weight = self.gather(self.weights, index, 0)
        weight += (index + 1)
        x = x + weight
        return x


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.idx = Parameter(Tensor(0), name="index")
        self.iter_num = 3
        self.add_net = AddIndexNet(self.iter_num)

    def construct(self, x):
        while self.idx < self.iter_num:
            x = self.add_net(x, self.idx)
            self.idx += 1
        return x


x = Tensor([0], mindspore.float32)
net = Net()
out = net(x)
print(out)
```

执行结果如下：

```text
[6.]
```

在该例中，拓展了权重的维度为`[iter_num, 1]`，不同迭代中即使共享同一个权重，但第0维之外的数据相对独立，使用时再通过`Gather`算子取出对应的数据，完成相应的计算。预期的结果是`1+2+3=6`，本例中打印的结果值符合预期。

## 约束

当前使用流程语句除了条件变量场景下的约束，还有一些其他特定场景下的约束。

### 副作用约束

在使用条件为变量的流程控制语句时，图编译生成的网络模型中会包含控制流算子，在此场景下，正向图会执行两次。如果此时正向图中存在`Assign`等副作用算子并且是训练场景时，会导致反向图计算结果与预期不符。

控制流训练场景不支持的副作用算子列表如下：

| Side Effect List      |
| --------------------- |
| Print                 |
| Assign                |
| AssignAdd             |
| AssignSub             |
| ScalarSummary         |
| ImageSummary          |
| TensorSummary         |
| HistogramSummary      |
| ScatterAdd            |
| ScatterDiv            |
| ScatterMax            |
| ScatterMin            |
| ScatterMul            |
| ScatterNdAdd          |
| ScatterNdSub          |
| ScatterNdUpadte       |
| ScatterNonAliasingAdd |
| ScatterSub            |
| ScatterUpdate         |

### 死循环约束

当表达式`while cond`中`cond`的值恒为标量`True`时，无论循环内部是否存在`break`或`return`循环退出语句，均可能会出现未知异常。
