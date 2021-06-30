# Static Graph Syntax Support

`Linux` `Ascend` `GPU` `CPU` `Model Development` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Static Graph Syntax Support](#static-graph-syntax-support)
    - [Overview](#overview)
    - [Data Types](#data-types)
        - [Built-in Python Data Types](#built-in-python-data-types)
            - [Number](#number)
            - [String](#string)
            - [List](#list)
            - [Tuple](#tuple)
            - [Dictionary](#dictionary)
        - [MindSpore User-defined Data Types](#mindspore-user-defined-data-types)
            - [Tensor](#tensor)
            - [Primitive](#primitive)
            - [Cell](#cell)
    - [Operators](#operators)
        - [Arithmetic Operators](#arithmetic-operators)
        - [Assignment Operators](#assignment-operators)
        - [Logical Operators](#logical-operators)
        - [Member Operators](#member-operators)
        - [Identity Operators](#identity-operators)
    - [Expressions](#expressions)
        - [Conditional Control Statements](#conditional-control-statements)
            - [single if](#single-if)
            - [side-by-side if](#side-by-side-if)
            - [if in if](#if-in-if)
        - [Loop Statements](#loop-statements)
            - [for](#for)
            - [while](#while)
            - [side-by-side while](#side-by-side-while)
            - [while in while](#while-in-while)
        - [Conditional Control Statements in Loop Statements](#conditional-control-statements-in-loop-statements)
            - [if in for](#if-in-for)
            - [if in while](#if-in-while)
        - [Function Definition Statements](#function-definition-statements)
            - [def Keyword](#def-keyword)
            - [lambda Expression](#lambda-expression)
    - [Functions](#functions)
        - [Python Built-in Functions](#python-built-in-functions)
            - [len](#len)
            - [isinstance](#isinstance)
            - [partial](#partial)
            - [map](#map)
            - [zip](#zip)
            - [range](#range)
            - [enumerate](#enumerate)
            - [super](#super)
            - [pow](#pow)
            - [print](#print)
        - [Function Parameters](#function-parameters)
    - [Network Definition](#network-definition)
        - [Instance Types on the Entire Network](#instance-types-on-the-entire-network)
        - [Network Construction Components](#network-construction-components)
        - [Network Constraints](#network-constraints)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/note/source_en/static_graph_syntax_support.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

## Overview

In graph mode, Python code is not executed by the Python interpreter. Instead, the code is compiled into a static computation graph, and then the static computation graph is executed.

Currently, only the function, Cell, and subclass instances modified by the `@ms_function` decorator can be built.
For a function, build the function definition. For the network, build the `construct` method and other methods or functions called by the `construct` method.

For details about how to use `ms_function`, click <https://www.mindspore.cn/docs/api/en/r1.3/api_python/mindspore.html#mindspore.ms_function>.

For details about the definition of `Cell`, click <https://www.mindspore.cn/docs/programming_guide/en/r1.3/cell.html>.

Due to syntax parsing restrictions, the supported data types, syntax, and related operations during graph building are not completely consistent with the Python syntax. As a result, some usage is restricted.

The following describes the data types, syntax, and related operations supported during static graph building. These rules apply only to graph mode.

> All the following examples run on the network in graph mode. For brevity, the network definition is not described.
>
> The `Tensor` cannot be directly constructed in static graphs. It can be transferred to the network through parameters or constructed in the `__init__` method as a network attribute and then used in the `construct` method of the network.

## Data Types

### Built-in Python Data Types

Currently, the following built-in `Python` data types are supported: `Number`, `String`, `List`, `Tuple`, and `Dictionary`.

#### Number

Supports `int`, `float`, and `bool`, but does not support complex numbers.

`Number` can be defined on the network. That is, the syntax `y = 1`, `y = 1.2`, and `y = True` are supported.

Forcible conversion to `Number` is not supported on the network. That is, the syntax `y = int(x)`, `y = float(x)`, and `y = bool(x)` are not supported.

#### String

`String` can be constructed on the network. That is, the syntax `y = "abcd"` is supported.

Forcible conversion to `String` is not supported on the network. That is, the syntax `y = str(x)` is not supported.

#### List

`List` can be constructed on the network, that is, the syntax `y = [1, 2, 3]` is supported.

Forcible conversion to `List` is not supported on the network. That is, the syntax `y = list(x)` is not supported.

`List` to be output in the computation graph will be converted into `Tuple`.

- Supported APIs

  `append`: adds an element to `list`.

  For example:

  ```python
  x = [1, 2, 3]
  x.append(4)
  ```

  The result is as follows:

  ```text
  x: (1, 2, 3, 4)
  ```

- Supported index values and value assignment

  Single-level and multi-level index values and value assignment are supported.

  The index value supports only `int`.

  The assigned value can be `Number`, `String`, `Tuple`, `List`, or `Tensor`.

  For example:

  ```python
  x = [[1, 2], 2, 3, 4]

  m = x[0][1]
  x[1] = Tensor(np.array([1, 2, 3]))
  x[2] = "ok"
  x[3] = (1, 2, 3)
  x[0][1] = 88
  n = x[-3]
  ```

  The result is as follows:

  ```text
  m: 2
  x: ([1, 88], Tensor(shape=[3], dtype=Int64, value=[1, 2, 3]), 'ok', (1, 2, 3))
  n: Tensor(shape=[3], dtype=Int64, value=[1, 2, 3])
  ```

#### Tuple

`Tuple` can be constructed on the network, that is, the syntax `y = (1, 2, 3)` is supported.

Forcible conversion to `Tuple` is not supported on the network. That is, the syntax `y = tuple(x)` is not supported.

- Supported index values

  The index value can be `int`, `slice`, `Tensor`, and multi-level index value. That is, the syntax `data = tuple_x[index0][index1]...` is supported.

  Restrictions on the index value `Tensor` are as follows:

    - `Tuple` stores `Cell`. Each `Cell` must be defined before a tuple is defined. The number of input parameters, input parameter type, and input parameter `shape` of each `Cell` must be the same. The number of outputs of each `Cell` must be the same. The output type must be the same as the output shape.

    - The index `Tensor` is a scalar `Tensor` whose `dtype` is `int32`. The value range is `[-tuple_len, tuple_len)`, negative index is not supported in `Ascend` backend.

    - This syntax does not support the running branches whose control flow conditions `if`, `while`, and `for` are variables. The control flow conditions can be constants only.

    - `GPU` and `Ascend` backend is supported.

  An example of the `int` and `slice` indexes is as follows:

  ```python
  x = (1, (2, 3, 4), 3, 4, Tensor(np.array([1, 2, 3])))
  y = x[1][1]
  z = x[4]
  m = x[1:4]
  n = x[-4]
  ```

  The result is as follows:

  ```text
  y: 3
  z: Tensor(shape=[3], dtype=Int64, value=[1, 2, 3])
  m: ((2, 3, 4), 3, 4)
  n: (2, 3, 4)
  ```

  An example of the `Tensor` index is as follows:

  ```python
  class Net(nn.Cell):
      def __init__(self):
          super(Net, self).__init__()
          self.relu = nn.ReLU()
          self.softmax = nn.Softmax()
          self.layers = (self.relu, self.softmax)

      def construct(self, x, index):
          ret = self.layers[index](x)
          return ret
  ```

#### Dictionary

`Dictionary` can be constructed on the network. That is, the syntax `y = {"a": 1, "b": 2}` is supported. Currently, only `String` can be used as the `key` value.

`Dictionary` to be output in the computational graph will extract all `value` values to form the `Tuple` output.

- Supported APIs

  `keys`: extracts all `key` values from `dict` to form `Tuple` and return it.

  `values`: extracts all `value` values from `dict` to form `Tuple` and return it.

  For example:

  ```python
  x = {"a": Tensor(np.array([1, 2, 3])), "b": Tensor(np.array([4, 5, 6])), "c": Tensor(np.array([7, 8, 9]))}
  y = x.keys()
  z = x.values()
  ```

  The result is as follows:

  ```text
  y: ("a", "b", "c")
  z: (Tensor(shape=[3], dtype=Int64, value=[1, 2, 3]), Tensor(shape=[3], dtype=Int64, value=[4, 5, 6]), Tensor(shape=[3], dtype=Int64, value=[7, 8, 9]))
  ```

- Supported index values and value assignment

  The index value supports only `String`. The assigned value can be `Number`, `Tuple`, or `Tensor`.

  For example:

  ```python
  x = {"a": Tensor(np.array([1, 2, 3])), "b": Tensor(np.array([4, 5, 6])), "c": Tensor(np.array([7, 8, 9]))}
  y = x["b"]
  x["a"] = (2, 3, 4)
  ```

  The result is as follows:

  ```text
  y: Tensor(shape=[3], dtype=Int64, value=[4, 5, 6])
  x: {"a": (2, 3, 4), Tensor(shape=[3], dtype=Int64, value=[4, 5, 6]), Tensor(shape=[3], dtype=Int64, value=[7, 8, 9])}
  ```

### MindSpore User-defined Data Types

Currently, MindSpore supports the following user-defined data types: `Tensor`, `Primitive`, and `Cell`.

#### Tensor

Currently, tensors cannot be constructed on the network. That is, the syntax `x = Tensor(args...)` is not supported.

You can use the `@constexpr` decorator to modify the function and generate the `Tensor` in the function.

For details about how to use `@constexpr`, click <https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.constexpr.html>.

The constant `Tensor` used on the network can be used as a network attribute and defined in `init`, that is, `self.x = Tensor(args...)`. Then the constant can be used in `construct`.

In the following example, `Tensor` of `shape = (3, 4), dtype = int64` is generated by `@constexpr`.

```python
@constexpr
def generate_tensor():
    return Tensor(np.ones((3, 4)))
```

The following describes the attributes, APIs supported by the `Tensor`.

- Supported attributes

  `shape`: obtains the shape of `Tensor` and returns a `Tuple`.

  `dtype`: obtains the data type of `Tensor` and returns a data type defined by `MindSpore`.

- Supported APIs

  `all`: reduces `Tensor` through the `all` operation. Only `Tensor` of the `Bool` type is supported.

  `any`: reduces `Tensor` through the `any` operation. Only `Tensor` of the `Bool` type is supported.

`view`: reshapes `Tensor` into input `shape`.

  `expand_as`: expands `Tensor` to the same `shape` as another `Tensor` based on the broadcast rule.

  For example:

  ```python
  x = Tensor(np.array([[True, False, True], [False, True, False]]))
  x_shape = x.shape
  x_dtype = x.dtype
  x_all = x.all()
  x_any = x.any()
  x_view = x.view((1, 6))

  y = Tensor(np.ones((2, 3), np.float32))
  z = Tensor(np.ones((2, 2, 3)))
  y_as_z = y.expand_as(z)
  ```

  The result is as follows:

  ```text
  x_shape: (2, 3)
  x_dtype: Bool
  x_all: Tensor(shape=[], dtype=Bool, value=False)
  x_any: Tensor(shape=[], dtype=Bool, value=True)
  x_view: Tensor(shape=[1, 6], dtype=Bool, value=[[True, False, True, False, True, False]])

  y_as_z: Tensor(shape=[2, 2, 3], dtype=Float32, value=[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])
  ```

#### Primitive

Currently, `Primitive` and its subclass instances can be constructed on the network. That is, the `reduce_sum = ReduceSum(True)` syntax is supported.

However, during construction, the parameter can be specified only in position parameter mode, and cannot be specified in the key-value pair mode. That is, the syntax `reduce_sum = ReduceSum(keep_dims=True)` is not supported.

Currently, the attributes and APIs related to `Primitive` and its subclasses cannot be called on the network.

For details about the definition of `Primitive`, click <https://www.mindspore.cn/docs/programming_guide/en/r1.3/operators.html>.

For details about the defined `Primitive`, click <https://www.mindspore.cn/docs/api/en/r1.3/api_python/mindspore.ops.html>.

#### Cell

Currently, `Cell` and its subclass instances can be constructed on the network. That is, the syntax `cell = Cell(args...)` is supported.

However, during construction, the parameter can be specified only in position parameter mode, and cannot be specified in the key-value pair mode. That is, the syntax `cell = Cell(arg_name=value)` is not supported.

Currently, the attributes and APIs related to `Cell` and its subclasses cannot be called on the network unless they are called through `self` in `contrcut` of `Cell`.

For details about the definition of `Cell`, click <https://www.mindspore.cn/docs/programming_guide/en/r1.3/cell.html>.

For details about the defined `Cell`, click <https://www.mindspore.cn/docs/api/en/r1.3/api_python/mindspore.nn.html>.

## Operators

Arithmetic operators and assignment operators support the `Number` and `Tensor` operations, as well as the `Tensor` operations of different `dtype`.

This is because these operators are converted to operators with the same name for computation, and they support implicit type conversion.

For details about the rules, click <https://www.mindspore.cn/docs/note/en/r1.3/operator_list_implicit.html>.

### Arithmetic Operators

| Arithmetic Operator | Supported Type
| :----------- |:--------
| `+` |`Number` + `Number`, `Tensor` + `Tensor`, `Tensor` + `Number`, `Tuple` + `Tuple`, `String` + `String`, and `List` + `List`
| `-` |`Number` - `Number`, `Tensor` - `Tensor`, and `Tensor` - `Number`
| `*` |`Number` \* `Number`, `Tensor` \* `Tensor`, and `Tensor` \* `Number`
| `/` |`Number` / `Number`, `Tensor` / `Tensor`, and `Tensor` / `Number`
| `%` |`Number` % `Number`, `Tensor` % `Tensor`, and `Tensor`% `Number`
| `**` |`Number` \*\* `Number`, `Tensor` \*\* `Tensor`, and `Tensor` \*\* `Number`
| `//` |`Number` // `Number`, `Tensor` // `Tensor`, and `Tensor` // `Number`
| `~`  | `~Tensor[Bool]`

### Assignment Operators

| Assignment Operator | Supported Type
| :----------- |:--------
| `=`          |Scalar and `Tensor`
| `+=` |`Number` += `Number`, `Tensor` += `Tensor`, `Tensor` += `Number`, `Tuple` += `Tuple`, and `String` += `String`
| `-=` |`Number` -= `Number`, `Tensor` -= `Tensor`, and `Tensor` -= `Number`
| `*=` |`Number` \*= `Number`, `Tensor` \*= `Tensor`, and `Tensor` \*= `Number`
| `/=` |`Number` /= `Number`, `Tensor` /= `Tensor`, and `Tensor` /= `Number`
| `%=` |`Number` %= `Number`, `Tensor` %= `Tensor`, and `Tensor` %= `Number`
| `**=` |`Number` \*\*= `Number`, `Tensor` \*\*= `Tensor`, and `Tensor` \*\*= `Number`
| `//=` |`Number` //= `Number`, `Tensor` //= `Tensor`, and `Tensor` //= `Number`

### Logical Operators

| Logical Operator | Supported Type
| :----------- |:--------
| `and` |`Number` and `Number`, `Tensor`, and `Tensor`
| `or` |`Number` or `Number`, and `Tensor` or `Tensor`
| `not` |not `Number`, not `Tensor`, and not `tuple`

### Member Operators

| Member Operator | Supported Type
| :----------- |:--------
| `in` |`Number` in `tuple`, `String` in `tuple`, `Tensor` in `Tuple`, `Number` in `List`, `String` in `List`, `Tensor` in `List`, and `String` in `Dictionary`
| `not in` | Same as `in`

### Identity Operators

| Identity Operator | Supported Type
| :----------- |:--------
| `is` | The value can only be `None`, `True`, or `False`.
| `is not` | The value can only be `None`, `True`, or `False`.

## Expressions

### Conditional Control Statements

#### single if

Usage:

- `if (cond): statements...`

- `x = y if (cond) else z`

Parameter: `cond` -- The supported types are `Number`, `Tuple`, `List`, `String`, `None`, `Tensor` and `Function`. It can also be an expression whose computation result type is one of them.

Restrictions:

- During graph building, if `if` is not eliminated, the data type and shape of `return` inside the `if` branch must be the same as those outside the `if` branch.

- When only `if` is available, the data type and shape of the `if` branch variable after the update must be the same as those before the update.

- When both `if` and `else` are available, the updated data type and shape of the `if` branch variable must be the same as those of the `else` branch.

- Does not support higher-order differential.

- Does not support `elif` statements.

Example 1:

```python
if x > y:
  return m
else:
  return n
```

The data types of `m` returned by the `if` branch and `n` returned by the `else` branch must be the same as those of shape.

Example 2:

```python
if x > y:
  out = m
else:
  out = n
return out
```

The data types of `out` after the `if` branch is updated and `else` after the `out` branch is updated must be the same as those of shape.

#### side-by-side if

Usage:

- `if (cond1):statements else:statements...if (cond2):statements...`

Parameters: `cond1` and `cond2` -- Consistent with `single if`.

Restrictions:

- Inherit all restrictions of `single if`.

- The total number of `if` in calculating graph can not exceed 50.

- Too many `if` will cause the compilation time to be too long. Reducing the number of `if` will help improve compilation efficiency.

Example:

```python
if x > y:
  out = x
else:
  out = y
if z > x:
  out = out + 1
return out
```

#### if in if

Usage:

- `if (cond1):if (cond2):statements...`

Parameters: `cond1` and `cond2` -- Consistent with `single if`.

Restrictions:

- Inherit all restrictions of `single if`.

- The total number of `if` in calculating graph can not exceed 50.

- Too many `if` will cause the compilation time to be too long. Reducing the number of `if` will help improve compilation efficiency.

Example:

```python
if x > y:
  z = z + 1
  if z > x:
    return m
else:
  return n
```

### Loop Statements

#### for

Usage:

- `for i in sequence`

Parameter: `sequence` --Iterative sequences (`Tuple` and `List`).

Restrictions:

- The total number of graph operations is a multiple of number of iterations of the `for` loop. Excessive number of iterations of the `for` loop may cause the graph to occupy more memory than usage limit.

Example:

```python
z = Tensor(np.ones((2, 3)))
x = (1, 2, 3)
for i in x:
  z += i
return z
```

The result is as follows:

```text
z: Tensor(shape=[2, 3], dtype=Int64, value=[[7, 7], [7, 7], [7, 7]])
```

#### single while

Usage:

- `while (cond)`

Parameter: `cond` -- Consistent with `single if`.

Restrictions:

- During graph building, if `while` is not eliminated, the data type and `shape` of `return` inside `while` must be the same as those outside `while`.

- The data type and shape of the updated variables in `while` must be the same as those before the update.

- Does not support training scenarios.

Example 1:

```python
while x < y:
  x += 1
  return m
return n
```

The `m` data type returned inside `while` inside and `n` data type returned outside `while` must be the same as those of shape.

Example 2:

```python
out = m
while x < y:
  x += 1
  out = out + 1
return out
```

In `while`, the data types of `out` before and after update must be the same as those of shape.

#### side-by-side while

Usage:

- `while (cond1):statements while (cond2):statemetns...`

Parameters: `cond1` and `cond2` -- Consistent with `single if`.

Restrictions:

- Inherit all restrictions of `single while`.

- The total number of `while` in calculating graph can not exceed 50.

- Too many `while` will cause the compilation time to be too long. Reducing the number of `while` will help improve compilation efficiency.

Example:

```python
out = m
while x < y:
  x += 1
  out = out + 1
while out > 10:
  out -= 10
return out
```

#### while in while

Usage:

-`while (cond1):while (cond2):statements...`

Parameters: `cond1` and `cond2` -- Consistent with `single if`.

Restrictions:

- Inherit all restrictions of `single while`.

- The total number of `while` in calculating graph can not exceed 50.

- Too many `while` will cause the compilation time to be too long. Reducing the number of `while` will help improve compilation efficiency.

Example:

```python
out = m
while x < y:
  while z < y:
    z += 1
    out = out + 1
  x += 1
return out
```

### Conditional Control Statements in Loop Statements

#### if in for

Usage:

- for i in sequence:if (cond)`

Parameters:

- `cond` -- Consistent with `single if`.

- `sequence` -- Iterative sequence(`Tuple`、`List`)

Restrictions:

- Inherit all restrictions of `single if`.

- Inherit all restrictions of `for`.

- If `cond` is variable, it is forbidden to use `if (cond):return`,`if (cond):continue`,`if (cond):break` statements.

- The total number of `if` is a multiple of number of iterations of the `for` loop. Excessive number of iterations of the `for` loop may cause the compilation time to be too long.

Example:

```python
z = Tensor(np.ones((2, 3)))
x = (1, 2, 3)
for i in x:
  if i < 3:
    z += i
return z
```

The result is as follows:

```text
z: Tensor(shape=[2, 3], dtype=Int64, value=[[4, 4], [4, 4], [4, 4]])
```

#### if in while

Usage:

- `while (cond1):if (cond2)`

Parameters: `cond1` and `cond2` -- Consistent with `single if`.

Restrictions:

- Inherit all restrictions of `single if` and `single while`.

- If `cond2` is variable, it is forbidden to use `if (cond2):return`,`if (cond2):continue`,`if (cond2):break` statements.

Example:

```python
out = m
while x < y:
  if z > 2*x:
    out = out + 1
  x += 1
return out
```

### Function Definition Statements

#### def Keyword

Defines functions.

Usage:

`def function_name(args): statements...`

For example:

```python
def number_add(x, y):
  return x + y
ret = number_add(1, 2)
```

The result is as follows:

```text
ret: 3
```

#### lambda Expression

Generates functions.

Usage: `lambda x, y: x + y`

For example:

```python
number_add = lambda x, y: x + y
ret = number_add(2, 3)
```

The result is as follows:

```text
ret: 5
```

## Functions

### Python Built-in Functions

Currently, the following built-in Python functions are supported: `len`, `isinstance`, `partial`, `map`, `range`, `enumerate`, `super`, and `pow`.

#### len

Returns the length of a sequence.

Calling: `len(sequence)`

Input parameter: `sequence` -- `Tuple`, `List`, `Dictionary`, or `Tensor`.

Return value: length of the sequence, which is of the `int` type. If the input parameter is `Tensor`, the length of dimension 0 is returned.

For example:

```python
x = (2, 3, 4)
y = [2, 3, 4]
d = {"a": 2, "b": 3}
z = Tensor(np.ones((6, 4, 5)))
x_len = len(x)
y_len = len(y)
d_len = len(d)
z_len = len(z)
```

The result is as follows:

```text
x_len: 3
y_len: 3
d_len: 2
z_len: 6
  ```

#### isinstance

Determines whether an object is an instance of a class. Different from operator `Isinstance`, the second input parameter of `Isinstance` is the type defined in the `dtype` module of MindSpore.

Calling: `isinstance(obj, type)`

Input parameters:

- `obj` -- Any instance of any supported type.

- `type` -- A type in the `MindSpore dtype` module.

Return value: If `obj` is an instance of `type`, return `True`. Otherwise, return `False`.

For example:

```python
x = (2, 3, 4)
y = [2, 3, 4]
z = Tensor(np.ones((6, 4, 5)))
x_is_tuple = isinstance(x, mstype.tuple_)
y_is_list= isinstance(y, mstype.list_)
z_is_tensor = isinstance(z, mstype.tensor)
```

The result is as follows:

```text
x_is_tuple: True
y_is_list: True
z_is_tensor: True
  ```

#### partial

A partial function used to fix the input parameter of the function.

Calling: `partial(func, arg, ...)`

Input parameters:

- `func` --Function.

- `arg` -- One or more parameters to be fixed. Position parameters and key-value pairs can be specified.

Return value: functions with certain input parameter values fixed

For example:

```python
def add(x, y):
  return x + y

add_ = partial(add, x=2)
m = add_(y=3)
n = add_(y=5)
```

The result is as follows:

```text
m: 5
n: 7
```

#### map

Maps one or more sequences based on the provided functions and generates a new sequence based on the mapping result.
If the number of elements in multiple sequences is inconsistent, the length of the new sequence is the same as that of the shortest sequence.

Calling: `map(func, sequence, ...)`

Input parameters:

- `func` --Function.

- `sequence` -- One or more sequences (`Tuple` or `List`).

Return value: A `Tuple`

For example:

```python
def add(x, y):
  return x + y

elements_a = (1, 2, 3)
elements_b = (4, 5, 6)
ret = map(add, elements_a, elements_b)
```

The result is as follows:

```text
ret: (5, 7, 9)
```

#### zip

Packs elements in the corresponding positions in multiple sequences into tuples, and then uses these tuples to form a new sequence.
If the number of elements in each sequence is inconsistent, the length of the new sequence is the same as that of the shortest sequence.

Calling: `zip(sequence, ...)`

Input parameter: `sequence` -- One or more sequences (`Tuple` or `List`)`.

Return value: A `Tuple`

For example:

```python
elements_a = (1, 2, 3)
elements_b = (4, 5, 6)
ret = zip(elements_a, elements_b)
```

The result is as follows:

```text
ret: ((1, 4), (2, 5), (3, 6))
```

#### range

Creates a `Tuple` based on the start value, end value, and step.

Calling:

- `range(start, stop, step)`

- `range(start, stop)`

- `range(stop)`

Input parameters:

- `start` -- start value of the count. The type is `int`. The default value is 0.

- `stop` -- end value of the count (exclusive). The type is `int`.

- `step` -- Step. The type is `int`. The default value is 1.

Return value: A `Tuple`

For example:

```python
x = range(0, 6, 2)
y = range(0, 5)
z = range(3)
```

The result is as follows:

```text
x: (0, 2, 4)
y: (0, 1, 2, 3, 4)
z: (0, 1, 2)
```

#### enumerate

Generates an index sequence of a sequence. The index sequence contains data and the corresponding subscript.

Calling:

- `enumerate(sequence, start)`

- `enumerate(sequence)`

Input parameters:

- `sequence` -- A sequence (`Tuple`, `List`, or `Tensor`).

- `start` -- Start position of the subscript. The type is `int`. The default value is 0.

Return value: A `Tuple`

For example:

```python
x = (100, 200, 300, 400)
y = Tensor(np.array([[1, 2], [3, 4], [5 ,6]]))
m = enumerate(x, 3)
n = enumerate(y)
```

The result is as follows:

```text
m: ((3, 100), (4, 200), (5, 300), (5, 400))
n: ((0, Tensor(shape=[2], dtype=Int64, value=[1, 2])), (1, Tensor(shape=[2], dtype=Int64, value=[3, 4])), (2, Tensor(shape=[2], dtype=Int64, value=[5, 6])))
```

#### super

Calls a method of the parent class (super class). Generally, the method of the parent class is called after `super`.

Calling:

- `super().xxx()`

- `super(type, self).xxx()`

Input parameters:

- `type` --Class.

- `self` --Object.

Return value: method of the parent class.

For example:

```python
class FatherNet(nn.Cell):
  def __init__(self, x):
      super(FatherNet, self).__init__(x)
      self.x = x

  def construct(self, x, y):
      return self.x * x

  def test_father(self, x):
      return self.x + x

class SingleSubNet(FatherNet):
def __init__(self, x, z):
    super(SingleSubNet, self).__init__(x)
    self.z = z

def construct(self, x, y):
    ret_father_construct = super().construct(x, y)
    ret_father_test = super(SingleSubNet, self).test_father(x)
    return ret_father_construct, ret_father_test
```

#### pow

Returns the power.

Calling: `pow(x, y)`

Input parameters:

- `x` -- Base number, `Number`, or `Tensor`.

- `y` -- Power exponent, `Number`, or `Tensor`.

Return value: `y` power of `x`, `Number`, or `Tensor`

For example:

```python
x = Tensor(np.array([1, 2, 3]))
y = Tensor(np.array([1, 2, 3]))
ret = pow(x, y)
```

The result is as follows:

```text
ret: Tensor(shape=[3], dtype=Int64, value=[1, 4, 27]))
```

#### print

Prints logs.

Calling: `print(arg, ...)`

Input parameter: `arg` -- Information to be printed (`int`, `float`, `bool`, `String` or `Tensor`).
When the `arg` is `int`, `float`, or `bool`, it will be printed out as a `0-D` tensor.

Return value: none

For example:

```python
x = Tensor(np.array([1, 2, 3]))
y = 3
print("x: ", x)
print("y: ", y)
```

The result is as follows:

```text
x: Tensor(shape=[3], dtype=Int64, value=[1, 2, 3]))
y: Tensor(shape=[], dtype=Int64, value=3))
```

### Function Parameters

- Default parameter value: The data types `int`, `float`, `bool`, `None`, `str`, `tuple`, `list`, and `dict` are supported, whereas `Tensor` is not supported.

- Variable parameters: Inference and training of networks with variable parameters are supported.

- Key-value pair parameter: Functions with key-value pair parameters cannot be used for backward propagation on computational graphs.

- Variable key-value pair parameter: Functions with variable key-value pairs cannot be used for backward propagation on computational graphs.

## Network Definition

### Instance Types on the Entire Network

- Common Python function with the [@ms_function](https://www.mindspore.cn/docs/api/en/r1.3/api_python/mindspore.html#mindspore.ms_function) decorator.

- Cell subclass inherited from [nn.Cell](https://www.mindspore.cn/docs/api/en/r1.3/api_python/nn/mindspore.nn.Cell.html).

### Network Construction Components

| Category                 | Content
| :-----------             |:--------
| `Cell` instance |[mindspore/nn/*](https://www.mindspore.cn/docs/api/en/r1.3/api_python/mindspore.nn.html) and user-defined [Cell](https://www.mindspore.cn/docs/api/en/r1.3/api_python/nn/mindspore.nn.Cell.html).
| Member function of a `Cell` instance | Member functions of other classes in the construct function of Cell can be called.
| `dataclass` instance | Class decorated with @dataclass.
| `Primitive` operator |[mindspore/ops/operations/*](https://www.mindspore.cn/docs/api/en/r1.3/api_python/mindspore.ops.html)
| `Composite` operator |[mindspore/ops/composite/*](https://www.mindspore.cn/docs/api/en/r1.3/api_python/mindspore.ops.html)
| `constexpr` generation operator | Value computation operator generated by [@constexpr](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.constexpr.html).
| Function                 | User-defined Python functions and system functions listed in the preceding content.

### Network Constraints

1. By default, the input parameters of the entire network (that is, the outermost network input parameters) support only `Tensor`. To support non-`Tensor`, you can set the `support_non_tensor_inputs` attribute of the network to `True`.

   During network initialization, `self.support_non_tensor_inputs = True` is set. Currently, this configuration supports only the forward network and does not support the backward network. That is, the backward operation cannot be performed on the network whose input parameters are not `Tensor`.

   The following is an example of supporting the outermost layer to transfer scalars:

   ```python
   class ExpandDimsNet(nn.Cell):
       def __init__(self):
           super(ExpandDimsNet, self).__init__()
           self.support_non_tensor_inputs = True
           self.expandDims = ops.ExpandDims()

       def construct(self, input_x, input_axis):
           return self.expandDims(input_x, input_axis)
   expand_dim_net = ExpandDimsNet()
   input_x = Tensor(np.random.randn(2,2,2,2).astype(np.float32))
   expand_dim_net(input_x, 0)
   ```

2. You are not allowed to modify non-`Parameter` data members of the network.

   For example:

   ```python
   class Net(Cell):
       def __init__(self):
           super(Net, self).__init__()
           self.num = 2
           self.par = Parameter(Tensor(np.ones((2, 3, 4))), name="par")

       def construct(self, x, y):
           return x + y
   ```

   In the preceding defined network, `self.num` is not a `Parameter` and cannot be modified. `self.par` is a `Parameter` and can be modified.

3. When an undefined class member is used in the `construct` function, `AttributeError` is not thrown like the Python interpreter. Instead, it is processed as `None`.

   For example:

   ```python
   class Net(Cell):
       def __init__(self):
           super(Net, self).__init__()

       def construct(self, x):
           return x + self.y
    ```

   In the preceding defined network, `construct` uses the undefined class member `self.y`. In this case, `self.y` is processed as `None`.
