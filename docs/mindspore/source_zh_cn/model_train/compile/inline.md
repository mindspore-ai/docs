# 函数inline

在传统编译器中，**inline**（内联）是一种优化技术，可以把被调用函数的代码直接替换到调用该函数的位置，提高程序运行效率。假设我们有一个 C++ 函数`add`，用于对两个数求和：

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
