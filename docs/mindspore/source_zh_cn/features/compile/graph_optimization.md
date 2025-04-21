# 图优化

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/features/compile/graph_optimization.md)

与传统编译器类似，MindSpore 在进行完构图之后，也会进行编译优化。编译优化的主要目的是通过静态分析技术对 MindSpore 的中间表示 [MindIR](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/design/all_scenarios.html#%E4%B8%AD%E9%97%B4%E8%A1%A8%E7%A4%BAmindir) 进行分析和转换，以达成减小目标代码大小、提升代码执行效率、降低运行时资源开销或者提升其它性能指标的目的。编译优化是图编译系统中的重要一环，对提升整个神经网络模型的性能和资源利用率有着极其重要的意义，相较于未经过编译优化的原始代码，编译优化可能带来数倍甚至数十倍的性能提升。

本节主要介绍独立于特定硬件的前端编译优化技术，特定于硬件的后端编译优化技术不在本节的讨论范围之内。

## 前端编译优化技术原理

与传统编译优化技术类似，MindSpore 中的编译优化也是通过一个个 Pass 来完成的。将每个 Pass 的上一个 Pass 所产生的 MindIR 作为输入，经过本 Pass 优化之后，产生新的 MindIR 表示作为输出。一个大的 Pass 可以包含多个小的 Pass，每个小的 Pass 只负责单点的编译优化，如：代数化简、函数内联（inline）、冗余消除等。一个 Pass 产生的优化结果，可能会为其它的 Pass 带来优化机会，故可以循环运行这些 Pass，直到产生的 MindIR 不再发生变化为止。

编译优化过程中，选择运行哪些 Pass，如何安排这些 Pass 的执行顺序对生成的最终的编译结果有着非常重要的影响。可以按照实际情况，通过设定编译优化策略（如优化级别、次数等）来对即将执行的优化动作进行调整。

## 常见前端编译优化技术

前端编译优化技术有很多，如：代数化简、函数inline（内联）、冗余消除等。本节将介绍部分具有代表性的编译优化技术。

### 代数化简

在传统编译器中，代数化简是一种编译器优化技术，旨在简化源代码中的代数表达式，消除多余计算，提高程序执行效率、减少内存占用等。

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

在 MindSpore编译器中，代数化简原理不同于传统编译器，进行处理的是计算图而非传统控制流图，通过调整计算图中算子的执行顺序，或者删除不必要的算子，以保持计算图的简洁性和提高计算效率。

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

通过代数化简，可以直接删除 PrimFunc_Add 算子，简化计算图结构，将 `x + 0` 简化成 `x`。

```txt
%para1_x: <Tensor[Int32], (2, 3)>

subgraph @1_func_14() {
    Return(%para1_x)
        : (<Tensor[int32], (2, 3)>)
}
```

代数化简能更多地涉及对计算图结构的修改，它通常还与其他编译器优化技术（如常量折叠、常量传播等）结合使用，共同提高程序性能。

### 函数inline

在传统编译器中，inline（内联）是一种优化技术，可以把被调用函数的代码直接替换到调用该函数的位置，提高程序运行效率。假设我们有一个 C++ 函数`add`，用于对两个数求和：

```cpp
int add(int a, int b) {
    return a + b;
}

int main() {
    int x = add(3, 5);
    int y = add(x, 10);
    return y;
}
```

编译器通过 inline 将函数体直接替换到调用处，这消除了函数调用的开销，同时为后续优化（如消除冗余计算`3 + 5`，直接在编译期求值替换）创造了条件。这种**用代码替换调用**的思想，正是 inline 的核心。

```cpp
int main() {
    int x = 3 + 5;   // 替换第一次调用
    int y = x + 10;  // 替换第二次调用
    return y;
}
```

在 AI 框架的计算图编译器中，inline 的目标类似，但操作对象从“函数”变成了“子图”（subgraph）。假设我们有一个 Python 程序：

```python
from mindspore import Tensor, jit, ops

def f2(x: Tensor, y: Tensor):
    return x * 0.5 + y

@jit
def f1(a: Tensor, b: Tensor, c: Tensor):
    x = f2(a, b)
    y = f2(a, c)
    return x + y

# 创建3个shape=(2, 4)的随机值Tensor
a = ops.randn(2, 4)
b = ops.randn(2, 4)
c = ops.randn(2, 4)
out = f1(a, b, c)
```

首先，MindSpore 的计算图编译器会把 Python 程序转换为计算图。而 Python 程序中的函数调用，会转换为计算图之间的调用，得到类似于下面的原始计算图。其中，主图 f1 调用了 2 次子图 f2。

```txt
# Params:
%para1_a: <Tensor[Float32], (2, 4)>
%para2_b: <Tensor[Float32], (2, 4)>
%para3_c: <Tensor[Float32], (2, 4)>

subgraph @f2(%para1_x, %para2_y) {
    %0 = PrimFunc_Mul(%para1_x, Float32(0.5))

    %1 = PrimFunc_Add(%0, %para2_y)

    Return(%2)
}

subgraph @f1() {
  %0(x) = call @f2(%para1_a, %para2_b)  # 调用子图f2

  %1(y) = call @f2(%para1_a, %para3_c)  # 调用子图f2

  %2 = PrimFunc_Add(%1, %2)

  Return(%2)
}
```

通过 inline，可以将子图 f2 展开，合并到主图 f1。

```txt
subgraph @f1() {
  # 第一次子图inline
  %0 = PrimFunc_Mul(%para1_a, Float32(0.5))  # 重复计算步骤
  %1 = PrimFunc_Add(%0, %para2_b)

  # 第二次子图inline
  %2 = PrimFunc_Mul(%para1_a, Float32(0.5))  # 重复计算步骤
  %3 = PrimFunc_Add(%2, %para3_c)

  %4 = PrimFunc_Add(%1, %3)

  Return(%4)
}
```

在 inline 将子图展开之前，编译器可能无法识别到两次调用子图 f2 中的重复操作（此时子图通常被当作黑盒处理）。而通过 inline 将子图展开后，此时编译器可以清晰看到`x * 0.5`被计算了两次，就可以触发编译器进一步的优化：**公共子表达式消除** (CSE, Common Subexpression Elimination)，这样就降低了计算量。

```txt
subgraph @f1() {
  %0 = PrimFunc_Mul(%para1_a, Float32(0.5))  # CSE合并重复计算

  %1 = PrimFunc_Add(%0, %para2_b)

  %2 = PrimFunc_Add(%0, %para3_c)  # 直接复用%0

  %3 = PrimFunc_Add(%1, %2)

  Return(%3)
}
```

通过 inline 将子图展开，编译器能够更清晰地识别跨子图的优化机会，除了公共子表达式消除 (CSE)，还能够触发算子融合、内存管理等许多优化措施。因此 inline 是计算图编译器的一项重要优化机制，也是许多跨图优化的基础。

### 冗余消除

在传统编译器中，冗余消除包含了多种编译优化技术，旨在通过在编译期间识别出代码中存在冗余的部分并进行消除，达到减少不必要的计算，提高程序的执行效率的目的。

通常冗余代码可能是用户出于可读性等目的有意编写的，也可能仅仅是编码过程中的无心之举。此外，编译优化过程本身通过其它优化技术（如：代数化简、inline、公共子表达式消除等）产生的中间结果，也可能带来冗余消除的机会。

冗余消除的技术有很多，本节挑选了其中常见的无用代码消除、不可达代码消除进行介绍。

1. **无用代码消除**

    消除计算结果未被使用的代码。例如：下面的 C++ 代码中，变量 `c` 未被任何其它代码使用，编译器可以通过静态分析领域的数据流分析等技术，将计算 `int c = x * y` 的这行代码消除。

    ```cpp
    int func(x, y) {
        int a = x + y;
        int b = x - y;
        int c = x * y; // 无用代码
        int d = a / b;
        return d;
    }
    ```

2. **不可达代码消除**

    消除未被有效控制流路径包含的代码。例如：下面的 C++ 代码中，编译器可以通过静态分析领域的控制流分析技术，分析代码的控制流图，识别到表达式 `1 < 0` 恒不成立，从而控制流 `1 < 0` 包含的代码在实际运行期间必定不会被执行，故可将该分支的代码消除。

    ```cpp
    int func(x, y) {
        int a = x + y;

        int b;
        if 1 < 0 { // 不可达分支
            b = x + y;
        } else {
            b = x - y;
        }

        int d = a / b;
        return d;
    }
    ```

MindSpore 图模式下冗余消除的目的及使用的技术也类似。与传统编译器不同的是，这些冗余优化技术是在 MindIR 上完成的。类似的，MindSpore 中常见的冗余消除技术有：

1. **无用代码消除**

    假设有如下存在冗余计算的Python代码：

    ```python
    import mindspore as ms
    from mindspore.common import Tensor, jit

    @jit
    def func(x, y):
        a = x + y
        b = x - y
        c = x * y # 无用代码
        d = a / b
        return d

    x = Tensor(20, ms.float32)
    y = Tensor(10, ms.float32)
    out = func(x, y)
    ```

    MindSpore 图编译器会通过静态分析将 `@jit` 修饰的 Python 代码转换为 MindIR 的表示形式并消除其中冗余的 `c = x * y` 的计算，最终生成的 MindIR 如下：

    ```txt
    # Params:
    %para1_x: <Tensor[Float32], ()>
    %para2_y: <Tensor[Float32], ()>

    subgraph @func_1() {
    %0(a) = PrimFunc_Add(%para1_x, %para2_y)
        : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
    %1(b) = PrimFunc_Sub(%para1_x, %para2_y)
        : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
    %2(d) = PrimFunc_Div(%0, %1)
        : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
    Return(%2)
        : (<Tensor[Float32], ()>)
    }
    ```

2. **不可达代码消除**

    假设有如下存在不可达路径的Python代码：

    ```python
    import mindspore as ms
    from mindspore.common import Tensor, jit

    @jit
    def func(x, y):
        a = x + y
        if 1 < 0: # 不可达分支
            b = x + y
        else:
            b = x - y
        d = a / b
        return d

    x = Tensor(20, ms.float32)
    y = Tensor(10, ms.float32)
    out = func(x, y)
    ```

    MindSpore 图编译器会通过静态分析将 `@jit` 修饰的 Python 代码转换为 MindIR 的表示形式并消除其中冗余的控制流分支 `1 < 0` 的代码，最终生成的 MindIR 如下：

    ```txt
    # Params:
    %para1_x: <Tensor[Float32], ()>
    %para2_y: <Tensor[Float32], ()>

    subgraph @func_1() {
    %0(a) = PrimFunc_Add(%para1_x, %para2_y)
        : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
    %1(b) = PrimFunc_Sub(%para1_x, %para2_y)
        : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
    %2(d) = PrimFunc_Div(%0, %1)
        : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
    Return(%2) cnode_attrs: {checkpoint: Bool(1)}
        : (<Tensor[Float32], ()>)
    }
    ```

冗余消除在编译优化中扮演着重要的角色，在不改变程序原语义的前提下，能够显著提高程序的执行效率，通过减少不必要的运行时计算节省计算资源。冗余消除通常还与其它编译优化技术结合使用以获得更多消除冗余代码的机会。
