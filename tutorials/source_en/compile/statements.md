# Graph Mode Syntax - Python Statements

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/compile/statements.md)

## Simple Statements

### raise Statements

Support the use of `raise` to trigger an exception. `raise` syntax format: `raise[Exception [, args]]`. The `Exception` in the statement is the type of the exception, and the `args` is the user-supplied argument to the exception, usually a string or other object. The following types of errors are supported: NoExceptionType, UnknownError, ArgumentError, NotSupportError, NotExistsError, DeviceProcessError, AbortedError, IndexError, ValueError, TypeError, KeyError, AttributeError, NameError, AssertionError, BaseException, KeyboardInterrupt, Exception, StopIteration, OverflowError, ZeroDivisionError, EnvironmentError, IOError, OSError, ImportError, MemoryError, UnboundLocalError, RuntimeError, NotImplementedError, IndentationError, RuntimeWarning.

The raise syntax in graph mode does not support variables of type `Dict`.

For example:

```python
import mindspore
from mindspore import nn

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

    @mindspore.jit
    def construct(self, x, y):
        if x <= y:
            raise ValueError("x should be greater than y.")
        else:
            x += 1
        return x

net = Net()
net(mindspore.tensor(-2), mindspore.tensor(-1))
```

The output result:

```text
ValueError: x should be greater than y.
```

### assert Statements

Supports the use of assert for exception checking, `assert` syntax format: `assert[Expression [, args]]`, where `Expression` is the judgment condition. If the condition is true, nothing will be done, while if the condition is false, an exception message of type `AssertError` will be thrown. The `args` are user-supplied exception arguments, which can usually be strings or other objects.

```python
import mindspore
from mindspore import nn

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

    @mindspore.jit
    def construct(self, x):
        assert x in [2, 3, 4]
        return x

net = Net()
net(mindspore.tensor(-1))
```

Appears normally in the output:

```text
AssertionError.
```

### pass Statements

The `pass` statement doesn't do anything and is usually used as a placeholder to maintain structural integrity. For example:

```python
import mindspore
from mindspore import nn

class Net(nn.Cell):
  @mindspore.jit
  def construct(self, x):
    i = 0
    while i < 5:
      if i > 3:
        pass
      else:
        x = x * 1.5
      i += 1
    return x

net = Net()
ret = net(10)
print("ret:", ret)
```

The result is as follows:

```text
ret: 50.625
```

### return Statements

The `return` statement usually returns the result to the place where it was called, and statements after the `return` statement are not executed. If the return statement does not have any expression or the function does not have a `return` statement, a `None` object is returned by default. There can be more than one `return` statement within a function, depending on the situation. For example:

```python
import mindspore
from mindspore import nn

class Net(nn.Cell):
  @mindspore.jit
  def construct(self, x):
      if x > 0:
        return x
      else:
        return 0

net = Net()
ret = net(10)
print("ret:", ret)
```

The result is as follows:

```text
ret: 10
```

As above, there can be multiple `return` statements in a control flow scenario statement. If there is no `return` statement in a function, the None object is returned by default, as in the following use case:

```python
import mindspore

@mindspore.jit
def foo():
  x = 3
  print("x:", x)

res = foo()
assert res is None
```

### break Statements

The `break` statement is used to terminate a loop statement, i.e., it stops execution of the loop statement even if the loop condition does not have a `False` condition or if the sequence is not fully recursive, usually used in `while` and `for` loops. In nested loops, the `break` statement stops execution of the innermost loop.

```python
import mindspore
from mindspore import nn

class Net(nn.Cell):
  @mindspore.jit
  def construct(self, x):
    for i in range(8):
      if i > 5:
        x *= 3
        break
      x = x * 2
    return x

net = Net()
ret = net(10)
print("ret:", ret)
```

The result is as follows:

```text
ret: 1920
```

### continue Statements

The `continue` statement is used to jump out of the current loop statement and into the next round of the loop. This is different from the `break` statement, which is used to terminate the entire loop statement. `continue` is also used in `while` and `for` loops. For example:

```python
import mindspore
from mindspore import nn

class Net(nn.Cell):
  @mindspore.jit
  def construct(self, x):
    for i in range(4):
      if i > 2:
        x *= 3
        continue
    return x

net = Net()
ret = net(3)
print("ret:", ret)
```

The result is as follows:

```text
ret: 9
```

## Compound Statements

### Conditional Control Statements

#### if Statements

Usage:

- `if (cond): statements...`

- `x = y if (cond) else z`

Parameter: `cond` - Variables of `bool` type and constants of `bool`, `List`, `Tuple`, `Dict` and `String` types are supported.

Restrictions:

- If `cond` is not a constant, the variable or constant assigned to a same sign in different branches should have same data type. If the data type of assigned variables or constants is `Tensor`, the variables and constants should have same shape and element type.

Example 1:

```python
import mindspore

x = mindspore.tensor([1, 4], mindspore.int32)
y = mindspore.tensor([0, 3], mindspore.int32)
m = 1
n = 2

@mindspore.jit()
def test_cond(x, y):
    if (x > y).any():
        return m
    else:
        return n

ret = test_cond(x, y)
print('ret:{}'.format(ret))
```

The data type of `m` returned by the `if` branch and `n` returned by the `else` branch must be same.

The result is as follows:

```text
ret:1
```

Example 2:

```python
import mindspore

x = mindspore.tensor([1, 4], mindspore.int32)
y = mindspore.tensor([0, 3], mindspore.int32)
m = 1
n = 2

@mindspore.jit()
def test_cond(x, y):
    out = 3
    if (x > y).any():
        out = m
    else:
        out = n
    return out

ret = test_cond(x, y)
print('ret:{}'.format(ret))
```

The variable or constant `m` assigned to `out` in `if` branch and the variable or constant `n` assigned to out in `false` branch must have same data type.

The result is as follows:

```text
ret:1
```

Example 3:

```python
import mindspore

x = mindspore.tensor([1, 4], mindspore.int32)
y = mindspore.tensor([0, 3], mindspore.int32)
m = 1

@mindspore.jit()
def test_cond(x, y):
    out = 2
    if (x > y).any():
        out = m
    return out

ret = test_cond(x, y)
print('ret:{}'.format(ret))
```

The variable or constant `m` assigned to `out` in `if` branch and the variable or constant `init` initially assigned to `out` must have same data type.

The result is as follows:

```text
ret:1
```

### Loop Statements

#### for Statements

Usage:

- `for i in sequence  statements...`

- `for i in sequence  statements... if (cond) break`

- `for i in sequence  statements... if (cond) continue`

Parameter: `sequence` - Iterative sequences (`Tuple`, `List`, `range` and so on).

Restrictions:

- The total number of graph operations is a multiple of number of iterations of the `for` loop. Excessive number of iterations of the `for` loop may cause the graph to occupy more memory than usage limit.

- The `for...else...` statement is not supported.

Example:

```python
import numpy as np
import mindspore

z = mindspore.tensor(np.ones((2, 3)))

@mindspore.jit()
def test_cond():
    x = (1, 2, 3)
    for i in x:
        z += i
    return z

ret = test_cond()
print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:[[7. 7. 7.]
 [7. 7. 7.]]
```

#### while Statements

Usage:

- `while (cond)  statements...`

- `while (cond)  statements... if (cond1) break`

- `while (cond)  statements... if (cond1) continue`

Parameter: `cond` - Variables of `bool` type and constants of `bool`, `list`, `tuple`, `dict` and `string` types are supported.

Restrictions:

- If `cond` is not a constant, the variable or constant assigned to a same sign inside body of `while` and outside body of `while` should have same data type.If the data type of assigned variables or constants is `Tensor`, the variables and constants should have same shape and element type.

- The `while...else...` statement is not supported.

Example 1:

```python
import mindspore

m = 1
n = 2

@mindspore.jit()
def test_cond(x, y):
    while x < y:
        x += 1
        return m
    return n

ret = test_cond(1, 5)
print('ret:{}'.format(ret))
```

The data type of `m` returned inside `while` and data type of `n` returned outside `while` must have same data type.

The result is as follows:

```text
ret:1
```

Example 2:

```python
import mindspore

m = 1
n = 2

def ops1(a, b):
    return a + b

@mindspore.jit()
def test_cond(x, y):
    out = m
    while x < y:
        x += 1
        out = ops1(out, x)
    return out

ret = test_cond(1, 5)
print('ret:{}'.format(ret))
```

The variable `op1` assigned to `out` inside `while` and the variable or constant `init` initially assigned to `out` must have same data type.

The result is as follows:

```text
ret:15
```

### Function Definition Statements

#### def Keyword

`def` is used to define a function, followed by the function identifier name and the original parentheses `()`, which may contain the function parameters.
Usage: `def function_name(args): statements...`.

For example:

```python
import mindspore

def number_add(x, y):
    return x + y

@mindspore.jit()
def test(x, y):
    return number_add(x, y)

ret = test(1, 5)
print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:6
```

Instructions:

- Functions can support no return value, and no return value means that the default function return value is None.
- `Construct` function of the outermost network and the inner network function is support kwargs, like:`def construct(**kwargs):`.
- Mixed use of variable argument and non-variable argument is supported, like:`def function(x, y, *args)` and `def function(x = 1, y = 1, **kwargs)`.

#### lambda Expression

A `lambda` expression is used to generate an anonymous function. Unlike normal functions, it computes and returns only one expression. Usage: `lambda x, y: x + y`.

For example:

```python
import mindspore

@mindspore.jit()
def test(x, y):
    number_add = lambda x, y: x + y
    return number_add(x, y)

ret = test(1, 5)
print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:6
```

#### Partial function partial

Function: partial function, fixed function input parameter. Usage: `partial(func, arg, ...)`.

Input parameter:

- `func` - function.

- `arg` - One or more parameters to be fixed, support positional parameters and key-value pair parameters.

Return Value: Returns some functions with fixed input value.

The example is as follows:

```python
import mindspore
from mindspore import ops

def add(x, y):
    return x + y

@mindspore.jit()
def test():
    add_ = ops.partial(add, x=2)
    m = add_(y=3)
    n = add_(y=5)
    return m, n

m, n = test()
print('m:{}'.format(m))
print('n:{}'.format(n))
```

The result is as follows:

```text
m:5
n:7
```

#### Function Parameters

- Default parameter value: The default value set to `Tensor` type data is currently not supported, and `int`, `float`, `bool`, `None`, `str`, `tuple`, `list`, `dict` type data is supported.
- Variable parameters: Inference and training of networks with variable parameters are supported.
- Key-value pair parameter: Functions with key-value pair parameters cannot be used for backward propagation.
- Variable key-value pair parameter: Functions with variable key-value pairs cannot be used for backward propagation.

### List Comprehension and Generator Expression

Support for List Comprehension and Generator Expression. Support for constructing a new sequence.

#### List Comprehension

List comprehension are used to generate lists. Usage: `[arg for loop if statements]`.

The example is as follows:

```python
import mindspore

@mindspore.jit()
def test():
    l = [x * x for x in range(1, 11) if x % 2 == 0]
    return l

ret = test()
print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:[4, 16, 36, 64, 100]
```

Restrictions:

The use of multiple levels of nested iterators is not supported in graph mode.

The example usage of the restriction is as follows (two levels of iterators are used):

```python
l = [y for x in ((1, 2), (3, 4), (5, 6)) for y in x]
```

An error will be prompted:

```text
TypeError:  The `generators` supports one `comprehension` in ListComp/GeneratorExp, but got 2 comprehensions.
```

#### Dict Comprehension

Dict comprehension is used to generate lists. Usage: `{key, value for loop if statements}`.

The example is as follows:

```python
import mindspore

@mindspore.jit()
def test():
    x = [('a', 1), ('b', 2), ('c', 3)]
    res = {k: v for (k, v) in x if v > 1}
    return res

ret = test()
print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:{'b': 2, 'c': 3}
```

Restrictions:

The use of multi-layer nested iterators is not supported in graph mode.

The example usage of the restriction is as follows (two levels of iterators are used):

```python
x = ({'a': 1, 'b': 2}, {'d': 1, 'e': 2}, {'g': 1, 'h': 2})
res = {k: v for y in x for (k, v) in y.items()}
```

An error will be prompted:

```text
TypeError:  The `generators` supports one `comprehension` in DictComp/GeneratorExp, but got 2 comprehensions.
```

#### Generator Expression

Generator expressions are used to generate lists. Usage: `(arg for loop if statements)`.

For example:

```python
import mindspore

@mindspore.jit()
def test():
    l = (x * x for x in range(1, 11) if x % 2 == 0)
    return l

ret = test()
print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:[4, 16, 36, 64, 100]
```

Usage restrictions are the same as list comprehension, i.e., the use of multiple levels of nested iterators is not supported in graph mode.

### With Statement

In graph mode, the `with` statement is supported with limitations. The `with` statement requires that the object must have two magic methods: `__enter__()` and `__exit__()`.

It is worth noting that the class used in the with statement needs to be decorated with a decorator@ms.jit_class or inherited from nn. Cell, and more on this can be found in [Calling the Custom Class](https://www.mindspore.cn/tutorials/en/master/compile/static_graph_expert_programming.html#using-jit-class).

For example:

```python
import mindspore
from mindspore import nn

@mindspore.jit_class
class Sample:
    def __init__(self):
        super(Sample, self).__init__()
        self.num = mindspore.tensor([2])

    def __enter__(self):
        return self.num * 2

    def __exit__(self, exc_type, exc_value, traceback):
        return self.num * 4

class TestNet(nn.Cell):
    @mindspore.jit
    def construct(self):
        res = 1
        obj = Sample()
        with obj as sample:
            res += sample
        return res, obj.num

test_net = TestNet()
out1, out2 = test_net()
print("out1:", out1)
print("out2:", out2)
```

The result is as follows:

```text
out1: [5]
out2: [2]
```
