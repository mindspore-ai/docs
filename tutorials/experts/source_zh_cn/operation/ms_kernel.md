# MindSpore Hybrid 语法规范

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/tutorials/experts/source_zh_cn/operation/ms_kernel.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

MindSpore Hybrid DSL的语法与Python语法类似，例如函数定义、缩进和注释。把MindSpore Hybrid DSL书写的函数加上ms_hybrid装饰器后可以当做普通的`numpy`函数使用，也可以用于Custom的进行自定义算子。

```python
import numpy as np
import mindspore as ms
from mindspore.ops import ms_kernel

@ms_kernel
def outer_product(a, b):
    d = allocate(a.shape, a.dtype)
    c = output_tensor(a.shape, a.dtype)

    for i0 in range(a.shape[0]):
        for i1 in range(b.shape[1]):
            c[i0, i1] = 0.0
            for i2 in range(a.shape[1]):
                d[i0, i2] = 2 * a[i0, i2]
                c[i0, i1] = c[i0, i1] + sin(d[i0, i2] * b[i2, i1])
    return c

np_x = np.random.normal(0, 1, [4, 4]).astype(np.float32)
np_y = np.random.normal(0, 1, [4, 4]).astype(np.float32)

print(outer_product(np_x, np_y))

input_x = ms.Tensor(np_x)
input_y = ms.Tensor(np_y)

test_op_akg = ops.Custom(outer_product)
out = test_op_akg(input_x, input_y)
print(out)
```

## 语法规则

### 变量

MindSpore Hybrid DSL中的变量包括Tensor和Scalar两种形式。

对于Tensor类型的变量，除了在输入中提供的变量，其他变量都需要在使用前申明 `shape`和 `dtype`。

- 对于输出Tensor使用 `output_tensor`，用法为：`output_tensor(shape, dtype)`。
- 对于中间结果使用 `allocate`，用法为：`allocate(shape, dtype)`。

Tensor分配的示例代码如下：

```python
@ms_kernel
def kernel_func(a, b):
    # a和b作为输入tensor，可以直接使用

    # d为一个数据类型为fp16,形状为(2,)的Tensor，在下面的code中作为中间变量使用
    d = allocate((2,), "float16")
    # c为一个数据类型与b相同,形状与a相同的Tensor，在下面的code中作为函数输出使用
    c = output_tensor(a.shape, b.dtype)

    # d作为中间变量，给c赋值
    d[0] = b[0, 0]
    for i in range(4):
        for j in range(4):
            c[i, j] = d[0]

    # c作为输出
    return c
```

对于Scalar类变量，会将它第一次的赋值运算作为声明。赋值操作可以是一个立即数，也可以是一个计算表达式。Scalar类变量第一次赋值的地方决定了它的定义域（例如，某一个for loop之内），在定义域之外使用Scalar变量会报错。

Scalar变量使用的示例代码如下：

```python
@ms_kernel
def kernel_func(a):
    c = output_tensor(a.shape, a.dtype)

    for i in range(10): # i loop
        for j in range(5): # j loop
            # 用一个立即数给Scalar赋值
            d = 2.0
            # 用表达式给Scalar赋值
            e = a[i, j]
            # 正常使用scalar
            c[i, j] = d + e

    # Wrong: c[0, 0] = d
    # 不能在超出Scalar d的定义域（j loop）之外的范围使用

    return c
```

与原生Python语言不同的是，变量一旦创建，`shape`和 `dtype`就不能改变。

### 计算表达

MindSpore Hybrid DSL支持基本的四则运算表达，包括 `+, -, *, /`，及赋值运算符，包括 `=, +=, -=, *=, /=`。
用户可以像写Python表达一样书写计算表达式利用变量计算和为变量赋值。

**所有的计算需要基于标量计算，如果是Tensor对象那么写清楚所有index，即 `C[i, j] = A[i, j] + B[i, j]`。当前不支持 `C = A + B`这种向量化的写法。**

在书写计算表达式时，用户需要自行负责类型的合法性。表达式左右两边的类型需要保持一致，否则在**算子编译环节**会报错。计算式中的整数立即数会被认定为int32，而浮点立即数会被认定为float32。MindSpore Hybrid DSL不提供任何隐式的类型转化，所有类型转化都需要显式的书写出来。类型名即对应类型转换函数的名字，包括：

- int32
- float16
- float32
- (仅GPU后端)int8, int16, int64, float64

类型转换代码示例如下：

```python
@ms_kernel
def kernel_func(a):
    c = output_tensor((2,), "float16")

    # Wrong: c[0] = 0.1 此处c的类型为fp16, 而0.1的类型为fp32
    c[0] = float16(0.1) # float16(0.1)把表达式的类型转化为fp16
    c[1] = float16(a[0, 0]) # float16(a[0, 0])把表达式的类型转化为fp16

    return c
```

### 循环

当前只支持 `for` loop，不支持 `while`、 `break`、 `continue`关键词。

基本循环的写法和Python一样，循环维度的表达可以使用 `range`和 `grid`关键词。`range`表示一维的循环维度，接受一个参数表示循环的上限，例如：

```python
@ms_kernel
def kernel_func(a, b):
    c = output_tensor((3, 4, 5), "float16")

    for i in range(3):
        for j in range(4):
            for k in range(5):
                out[i, j, k] = a[i, j, k] + b[i, j, k]
    return  c
```

则循环表达的计算空间为 `0 <= i < 3, 0 <= j < 4, 0 <= k < 5`。

`grid`表示多维网格，接受的输入为 `tuple` ，例如上面的代码用 `grid`表达后如下：

```python
@ms_kernel
def kernel_func(a, b):
    c = output_tensor((3, 4, 5), "float16")

    for arg in grid((4,5,6)):
        out[arg] = a[arg] + b[arg]
    return  c
```

此时，参数 `arg`等价于一个三维index `(i,j,k)`，其上限分别为4，5，6。对参数 `arg`我们可以取其中的某个分量，例如

```python
@ms_kernel
def kernel_func(a, b):
    c = output_tensor((3, 4, 5), "float16")

    for arg in grid((4,5,6)):
        out[arg] = a[arg] + b[arg[0]]
    return  c
```

那么循环内的表达式等价于 `out[i, j, k] = a[i, j, k] + b[i]`。

### 调度原语

从1.8版本开始，MindSpore Hybrid DSL 提供调度原语以描述不同类型的循环。在 Ascend 后端，调度原语将协助新 DSA 多面体调度器生成代码。此类调度原语包括：`serial`， `vectorize`， `parallel`，和 `reduce`。

`serial` 会提示调度器该循环在调度生成时应保持前后顺序，不要做会改变顺序的调度变换，例如：

```python
@ms_kernel
def serial_test(a, b):
    row = a.shape[0]
    col = a.shape[1]
    for i in serial(row):
        for j in serial(i):
            b[i] = b[i] - a[i, j] * b[j]
    return b
```

这里 `serial` 提示 `i` 和 `j` 的计算有依赖关系，调度时应保持 `i` 和 `j` 从小的大的顺序。

`vectorize` 一般用于最内层循环，会提示调度器该循环有生成向量化指令的机会，例如：

```python
@ms_kernel
def vector_test(a, b):
    out = output_tensor(a.shape, a.dtype)
    row = a.shape[0]
    col = a.shape[1]
    for i in range(row):
        for j in vectorize(col):
            out[i, j] = a[i, j] + b[0, i]
    return out
```

这里 `vectorize` 提示最内层 `j` 轴循环包含同质化计算，调度时可以生成向量化指令加速内层循环。

`parallel` 一般用于最外层循环，会提示调度器该循环有并行执行机会，例如：

```python
@ms_kernel
def parallel_test(a, b):
    out = output_tensor(a.shape, a.dtype)
    row = a.shape[0]
    col = a.shape[1]
    for i in parallel(row):
        for j in range(col):
            out[i, j] = a[i, j] + b[0, j]
    return out
```

这里 `parallel` 提示最外层 `i` 轴循环无依赖关系，调度时可以并行加速。

`reduce` 会提示调度器该循环为运算中的一个 Reduction 轴，例如：

```python
def reduce_test(a):
    out = output_tensor((a.shape[0], ), a.dtype)
    row = a.shape[0]
    col = a.shape[1]
    for i in range(row):
        out[i] = 0.0
        for k in reduce(col):
            out[i] = out[i] + a[i, k]
    return out
```

这里 `reduce` 对应的 `k` 轴为累加轴。

用户在使用调度原语的时候需要注意：

- 上述调度原语只会在 Ascend 后端影响调度。在CPU和GPU后端，上述调度原语将被处理成普通的 `for` 循环关键词。
- 调度原语对于调度器只是提示作用，当调度原语的提示和调度器自身的分析验证相矛盾时，调度器将把上述调度原语将被处理成普通的 `for` 循环关键词。

### 属性

当前只支持对Tensor对象属性shape和dtype，例如 `a.shape`，`c.dtype`。

一个Tensor的shape属性会表达为一个 `tuple`，我们可以对它进行**固定**下标的取分量操作，例如 `a.shape[0]`。

同时，在 `grid`关键词中我们接受某个Tensor对象的 `shape`属性，那么循环的维度由Tensor的维度决定。例如：

```python
@ms_kernel
def kernel_func(a, b):
    c = output_tensor(a.shape, "float16")

    for arg in grid(a.shape):
        out[arg] = a[arg] + b[arg[0]]
    return  c
```

如果a是一个二维Tensor，那么循环内的表达式等价于 `out[i, j] = a[i, j] + b[i]`。而如果a是一个三维Tensor，那么循环内的表达式等价于 `out[i, j, k] = a[i, j, k] + b[i]`。

### 关键词

当前支持的关键词包括

- 全平台支持数学函数：`log`、`exp`、`sqrt`、`tanh`、`power`、`floor`
- 内存分配：`allocate`、 `output_tensor`
- 数据类型转化：`int32`、 `float16`、 `float32`、 `float64`
- 循环表达：`for`、 `range`、 `grid`
- 调度源语：`serial`、 `vec`、 `parallel`、 `reduce`
- 在当前版本中，我们对CPU/GPU后端提供部分进阶关键词：
    - 数学函数：`rsqrt`、 `erf`、 `isnan`、 `sin`、 `cos`、 `isinf`、 `isfinite`、 `atan`、 `atan2`(仅GPU)、 `expm1`(仅GPU)、 `floor`、 `ceil`、 `trunc`、 `round`、 `ceil_div`
    - 数据类型转换：`int8`，`int16`，`int64`

## 常见报错信息及错误归因

为了帮助用户高效地开发和定位bug，MindSpore Hybrid DSL 提供如下报错信息，包括

- TypeError: 当使用了`while`, `break` 和 `continue` 等 MindSpore Hybrid DSL 不支持的 Python 关键词。
- ValueError:
    - 使用了不属于上面的内置函数名；
    - 对张量取非 `shape` 或者 `dtype` 的属性。
- 其他常见报错：
    - “SyntaxError”: 写的 DSL 不符合基本 Python 语法（非上面的进阶用法中定义的MindSpore Hybrid DSL语法），由 Python 解释器本身报错；
    - “ValueError: Compile error”及“The pointer\[kernel_mod\] is null”: Python DSL符合语法但是编译失败，由 AKG 报错，具体错误原因检查 AKG 相关报错信息；
    - “Launch graph failed”: Python DSL符合语法，编译成功但是运行失败。具体原因参考硬件的报错信息。例如在昇腾芯片上遇到运行失败时，MindSpore 端会显示 “Ascend error occurred” 及对应硬件报错信息。
