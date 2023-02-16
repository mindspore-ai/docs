# MindSpore Hybrid Syntax Specification

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/operation/ms_kernel.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

The syntax of MindSpore Hybrid DSL is similar to Python syntax, such as function definitions, indentation, and annotations. Functions written in MindSpore Hybrid DSL can be used as ordinary `numpy` functions after adding `kernel` decorators, or they can be customized operators used for Custom.

```python
import numpy as np
import mindspore as ms
from mindspore.ops import kernel

@kernel
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

## Syntax Specification

### Variables

There are two kinds of variables in MindSpore Hybrid DSL: tensor variables and scalar variables.

Tensor variables, besides those in the inputs of the function, must be declared with `shape`和 `dtype` before use.

- declare a output tensor by `output_tensor`, such as `output_tensor(shape, dtype)`.
- declare an intermediate tensor by `allocate`, such as `allocate(shape, dtype)`.

Example of Tensor allocation:

```python
@kernel
def kernel_func(a, b):
    # We can use a and b directly as input tensors

    # d is a tensor with dtype fp16 and shape (2,), and will be used as an intermediate variable in the following code
    d = allocate((2,), "float16")
    # c is a tensor with same dtype and shape as a, and will be used as a output function in the following code
    c = output_tensor(a.shape, b.dtype)

    # assign value to c by d as the intermediate variable
    d[0] = b[0, 0]
    for i in range(4):
        for j in range(4):
            c[i, j] = d[0]

    # c as output
    return c
```

Scalar variables will regard its first assignment as the declaration. The assignment can be either a number or an expression. The place of the first assignment of a scalar variable defines its scope, such as within a certain level of for loop. Using the variable outside its scope will lead to error.

Example of using Scalar variable:

```python
@kernel
def kernel_func(a, b):
    c = output_tensor(a.shape, a.dtype)

    for i in range(10): # i loop
        for j in range(5): # j loop
            # assign a number to Scalar d
            d = 2.0
            # assign an expression to Scalar e
            e = a[i, j]
            # use scalars
            c[i, j] = d + e

    # Wrong: c[0, 0] = d
    # Can't use Scalar d outside its scope (j loop)
    return c
```

Unlike native Python language, once a variable is defined, we can't change its `shape`和 `dtype`.

### Expressions

MindSpore Hybrid DSL supports basic math operators, including `+, -, *, /`, as well as self-assign operators, including `=, +=, -=, *=, /=`.
Users can write codes like writing Python expressions.

**All the expressions must be based on scalars. Computation for the tensors must include all indices, such as `C[i, j] = A[i, j] + B[i, j]`. Currently, tensorized codes such as `C = A + B` are not supported.**

When writing assignment expressions, users must take care of the dtype of the expression and make them consistent on both sides of the equality. Otherwise, the error might be thrown on the stage of **operator compilation**. Any integer numbers in the expression will be treated as int32, while float numbers will be treated as float32. There is no implicit dtype casting in MindSpore Hybrid DSL, and all dtype casting must be written with dtype names as casting functions, including:

- int32
- float16
- float32
- (only on gpu backend) int8, int16, int64, float64

Example of dtype casting:

```python
@kernel
def kernel_func(a):
    c = output_tensor((2,), "float16")

    # Wrong: c[0] = 0.1 c's dtype is fp16, while 0.1's dtype is fp32
    c[0] = float16(0.1) # float16(0.1) cast the number 0.1 to dtype fp16
    c[1] = float16(a[0, 0]) # float16(a[0, 0]) cast the number 0.1 to dtype fp16
    return c
```

### Loop

Currently, only the `for` loop is supported. `while`, `break`, and `continue` are illegal in MindSpore Hybrid DSL.

Loops are the same as those in Python. `range` and `grid` are supported to express extents of loops. `range` is for one-dimensional loops and accepts a number as the upper bound of the loop, such as:

```python
@kernel
def kernel_func(a, b):
    c = output_tensor((3, 4, 5), "float16")

    for i in range(3):
        for j in range(4):
            for k in range(5):
                out[i, j, k] = a[i, j, k] + b[i, j, k]
    return  c
```

The iteration space of the above loops is `0 <= i < 3, 0 <= j < 4, 0 <= k < 5`.

`grid` is for multi-dimensional loops and accepts `tuple` as its input. For example, the above code can be also written as follows in `grid`:

```python
@kernel
def kernel_func(a, b):
    c = output_tensor((3, 4, 5), "float16")

    for arg in grid((4,5,6)):
        out[arg] = a[arg] + b[arg]
    return  c
```

Right now `arg` is equivalent to a three dimensional index `(i,j,k)`, with upper bound 4, 5, 6 respectively. We also have access to each element in `arg`, such as:

```python
@kernel
def kernel_func(a, b):
    c = output_tensor(a.shape, "float16")

    for arg in grid(a.shape):
        out[arg] = a[arg] + b[arg[0]]
    return  c
```

Then the expression inside loops is equivalent to `out[i, j, k] = a[i, j, k] + b[i]`.

### Scheduling keywords

From version 1.8, MindSpore Hybrid DSL provides scheduling keywords to describe the type of loops. On the Ascend backend, scheduling keywords will help the new DSA polyhedron scheduler generate codes. The scheduling keywords include `serial`, `vectorize`, `parallel`, and `reduce`.

`serial` indicates that the scheduler should keep the order of the loop and not apply loop transformations on such loops. For example,

```python
@kernel
def serial_test(a, b):
    row = a.shape[0]
    col = a.shape[1]
    for i in serial(row):
        for j in serial(i):
            b[i] = b[i] - a[i, j] * b[j]
    return b
```

Here `serial` indicates that there are dependence relations on `i` and `j`. `i` and `j` should be in ascending order during the scheduling.

`vectorize` is usually used in the innermost loop, indicating the chance of generation vector instructions. For example,

```python
@kernel
def vector_test(a, b):
    out = output_tensor(a.shape, a.dtype)
    row = a.shape[0]
    col = a.shape[1]
    for i in range(row):
        for j in vectorize(col):
            out[i, j] = a[i, j] + b[0, i]
    return out
```

Here `vectorize` indicates that the innermost `j` loop conducts the same computation at each iteration and that the computation can be accelerated via vector instructions.

`parallel` is usually used in the outermost loop, prompting the scheduler that the loop has the chance of parallel execution. For example,

```python
@kernel
def parallel_test(a, b):
    out = output_tensor(a.shape, a.dtype)
    row = a.shape[0]
    col = a.shape[1]
    for i in parallel(row):
        for j in range(col):
            out[i, j] = a[i, j] + b[0, j]
    return out
```

Here `parallel` indicates that there is no dependency between each iteration of the `i` loop and that the computation can be accelerated via parallelization.

`reduce` indicates that the loop is a reduction axis. For example,

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

Here `reduce` indicates that `k` is a reduction axis.

Notice that：

- The scheduling keywords will only influence the scheduling on the Ascend backend. On the CPU or GPU backend, the above scheduling keywords will be treated as the usual `for` keyword.
- The scheduling keywords only provide hints to the scheduler. When the hints from the scheduling keywords contradict the analysis and validation result from the scheduler, the above scheduling keywords will be treated as the usual `for` keyword.

### Attribute

Currently we support only tensor's `shape` and `dtype` attributes, such as `a.shape`, and `c.dtype`.

The shape attribute of a Tensor is a `tuple`. We have access to its element with a **fixed** index, such as `a.shape[0]`.

Once `grid` accepts one Tensor's `shape` attribute as its input, the dimension of the loops is the same as the dimension of the Tensor. For example:

```python
@kernel
def kernel_func(a, b):
    c = output_tensor(a.shape, "float16")

    for arg in grid(a.shape):
        out[arg] = a[arg] + b[arg[0]]
    return  c
```

If a is a two dimensional tensor, then the expression inside loops is equivalent to `out[i, j] = a[i, j] + b[i]`, while if a is a three dimensional tensor, then the expression inside loops is equivalent to `out[i, j, k] = a[i, j, k] + b[i]`.

### Keywords

Currently, we support keywords including:

- Math keywords (all platform): `log`, `exp`, `sqrt`, `tanh`, `power`, `floor`
- Memory allocation: `allocate`, `output_tensor`
- Datatype keywords: `int32`, `float16`, `float32`, `float64`
- For keywords: `for`, `range`, `grid`
- Scheduling keywords: `serial`, `vec`, `parallel`, `reduce`
- In current version, advanced keywords are provided for the CPU/GPU backend:
    - Math keywords: `rsqrt`, `erf`, `isnan`, `sin`, `cos`, `isinf`, `isfinite`, `atan`, `atan2` (only on GPU), `expm1` (only on GPU), `floor`, `ceil`, `trunc`, `round`, `ceil_div`
    - Datatype keywords: `int8`, `int16`, `int64`

## Frequent Error Messages and Error Attributions

To help users effectively develop and locate bugs, MindSpore Hybrid DSL provides the following error messages, including:

- TypeError: There are Python keywords such as `while`, `break` and `continue` which are not supported by MindSpore Hybrid DSL.
- ValueError:
    - There are built-in function names which are not in the above support list;
    - Take properties that are not `shape` or `dtype` for tensors.
- Other frequent error messages:
    - "SyntaxError": DSL does not conform to the Python syntax (not the syntax defined by MindSpore Hybrid DSL), and is reported by the Python interpreter itself
    - "ValueError: Compile error" and "The pointer\[kernel_mod\] is null": the kernel compiler fails in compiling the DSL. Check error messages from AKG for further information;
    - "Launch graph failed": The compiled kernel fails in running. Check the error message from the hardware. For example, when the kernel fails in Ascend, there will be an "Ascend error occurred" message and corresponding hareware error messages.
