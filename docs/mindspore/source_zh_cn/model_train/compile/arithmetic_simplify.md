# 代数化简

在传统编译器中，**代数化简** 是一种编译器优化技术，旨在简化源代码中的代数表达式，消除多余计算，提高程序执行效率、减少内存占用等。

例如，在以下代码片段中：

```cpp
int a = x * 1;
int b = x + 0;
int c = x * 0 + y * 1;
```

传统编译器根据代数规则和恒等式对识别出的表达式进行等价替换。常见代数规则包括结合律、交换律和分配律等，编译器尽可能将表达式替换成更为简单的形式。通过对 AST（抽象语法树）或 SSA（静态单赋值形式）的分析来进行优化，识别并简化代码为：

```cpp
a = x;
b = x;
c = y;
```

在 MindSpore编译器中，**代数化简** 原理不同于传统编译器，进行处理的是计算图而非传统控制流图，通过调整计算图中算子的执行顺序，或者删除不必要的算子，以保持计算图的简洁性和提高计算效率。

例如，在如下Python代码片段中：

```python
import numpy as np
from mindspore.common import Tensor, jit

@jit
def func(x):
    return x + 0

m = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32))
out = func(m)
```

MindSpore图编译器会把 Python 程序转换为计算图，计算图由多个子图构成。源程序中的代数运算，转换为子图内部的算子调用，可以看到 PrimFunc_Add 算子调用了一次。

```txt
%para1_x: <Tensor[Int32], (2, 3)>

subgraph @1_func_14() {
    %0(CNode_7) = PrimFunc_Add(%para1_x, Tensor(shape=[], dtype=Int32, value=0))
        : (<Tensor[int32], (2, 3)>, <Tensor[Int32], (), value=...>) -> (<Tensor[int32], (2, 3)>)

    Return(%0)
        : (<Tensor[int32], (2, 3)>)
}
```

通过 **代数化简** ，可以直接删除 PrimFunc_Add 算子，简化计算图结构, 将 `x + 0` 简化成 `x`。

```txt
%para1_x: <Tensor[Int32], (2, 3)>

subgraph @1_func_14() {
    Return(%para1_x)
        : (<Tensor[int32], (2, 3)>)
}
```

**代数化简** 能更多地涉及对计算图结构的修改，它通常还与其他编译器优化技术（如常量折叠、常量传播等）结合使用，共同提高程序性能。