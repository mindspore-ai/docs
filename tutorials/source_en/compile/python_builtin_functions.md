# Graph Mode Syntax - Python Built-in Functions

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/compile/python_builtin_functions.md)

Python built-in functions supported by the current static graph mode include: `int`, `float`, `bool`, `str`, `tuple`, `list`, `dict`, `getattr`, `hasattr`, `len`, `isinstance`, `all`, `any`, `round`, `max`, `min`, `sum`, `abs`, `map`, `zip` , `range`, `enumerate`, `super`, `pow`, `print`, `filter`, `type`. The use of built-in functions in graph mode is similar to the corresponding Python built-in functions.

## int

Function: Return the integer value based on the input number or string.

Call: `int(x=0, base=10)`, converted to decimal by default.

Input parameter:

- `x` - the object need to be converted to integer, the valid type of x includes `int`, `float`, `bool`, `str`, `Tensor` and third-party object (such as `numpy.ndarray`).

- `base` - the base to convert. `base` is only allowed when `x` is constant `str`.

Return value: the converted integer.

For example:

```python
import mindspore

@mindspore.jit
def func(x):
    a = int(3)
    b = int(3.6)
    c = int('12', 16)
    d = int('0xa', 16)
    e = int('10', 8)
    f = int(x)
    return a, b, c, d, e, f

x = mindspore.tensor([-1.0], mindspore.float32)
a, b, c, d, e, f = func(x)
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
```

The result is as follows:

```text
a:  3
b:  3
c:  18
d:  10
e:  8
f:  -1
```

## float

Function: Return the floating-point number based on the input number or string.

Calling: `float(x=0)`.

Input parameter: `x` - the object need to be converted to floating number, the valid type of x includes `int`, `float`, `bool`, `str`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: the converted floating-point number.

For example:

```python
import mindspore

@mindspore.jit
def func(x):
    a = float(1)
    b = float(112)
    c = float(-123.6)
    d = float('123')
    e = float(x.asnumpy())
    return a, b, c, d, e

x = mindspore.tensor([-1], mindspore.int32)
a, b, c, d, e = func(x)
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
```

The result is as follows:

```text
a:  1.0
b:  112.0
c:  -123.5999984741211
d:  123.0
e:  -1.0
```

## bool

Function: Return the boolean value based on the input.

Calling: `bool(x=false)`

Input parameter: `x` - the object need to be converted to boolean value, the valid type of x includes `int`, `float`, `bool`, `str`, `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: the converted boolean scalar.

For example:

```python
import mindspore

@mindspore.jit
def func():
    a = bool()
    b = bool(0)
    c = bool("abc")
    d = bool([1, 2, 3, 4])
    e = bool(mindspore.tensor([10]).asnumpy())
    return a, b, c, d, e

a, b, c, d, e = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
```

The result is as follows:

```text
a:  False
b:  False
c:  True
d:  True
e:  True
```

## str

Function: Return the string value based on the input.

Calling: `str(x='')`

Input parameter: `x` - the object need to be converted to string value, the valid type of x includes `int`, `float`, `bool`, `str`, `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: string converted from `x`.

For example, a is an empty string:

```python
import numpy as np
import mindspore

@mindspore.jit
def func():
    a = str()
    b = str(0)
    c = str([1, 2, 3, 4])
    d = str(mindspore.tensor([10]))
    e = str(np.array([1, 2, 3, 4]))
    return a, b, c, d, e

a, b, c, d, e = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
```

The result is as follows:

```text
a:                                             # a is empty string
b:  0
c:  [1, 2, 3, 4]
d:  Tensor(shape=[1], dtype=Int64, value=[10])
e:  [1 2 3 4]
```

## tuple

Function: Return a tuple based on the input object.

Calling: `tuple(x=())`.

Input parameter: `x` - the object that need to be converted to tuple, the valid type of x includes `list`, `tuple`, `dict`, `Tensor` or third-party object (such as `numpy.ndarray`).

Return value: tuple with elements of `x`, `x` is cut based on zero dimension.

For example:

```python
import numpy as np
import mindspore

@mindspore.jit
def func():
    a = tuple((1, 2, 3))
    b = tuple(np.array([1, 2, 3]))
    c = tuple({'a': 1, 'b': 2, 'c': 3})
    d = tuple(mindspore.tensor([1, 2, 3]))
    return a, b, c, d

a, b, c, d = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
```

The result is as follows:

```text
a:  (1, 2, 3)
b:  (1, 2, 3)
c:  ('a', 'b', 'c')
d:  (Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 2), Tensor(shape=[], dtype=Int64, value= 3))
```

## list

Function: Return a list based on the input object.

Calling: `list(x=())`.

Input parameter: `x` - the object that need to be converted to list, the valid type of x includes `list`, `tuple`, `dict`, `Tensor` or third-party object (such as `numpy.ndarray`).

Return value: list with elements of `x`, `x` is cut based on zero dimension.

For example:

```python
import numpy as np
import mindspore

@mindspore.jit
def func():
    a = list((1, 2, 3))
    b = list(np.array([1, 2, 3]))
    c = list({'a':1, 'b':2, 'c':3})
    d = list(mindspore.tensor([1, 2, 3]))
    return a, b, c, d
a_t, b_t, c_t, d_t = func()
print("a_t: ", a_t)
print("b_t: ", b_t)
print("c_t: ", c_t)
print("d_t: ", d_t)
```

The result is as follows:

```text
a_t:  [1, 2, 3]
b_t:  [1, 2, 3]
c_t:  ['a', 'b', 'c']
d_t:  [Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 2), Tensor(shape=[], dtype=Int64, value= 3)]
```

## dict

Function: Used to create a dictionary.

Examples of code usage are as follows:

```python
import mindspore

@mindspore.jit
def func():
    a = dict()                                          # Create an empty dictionary
    b = dict(a='a', b='b', t='t')                       # Pass in keywords
    c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))   # Mapping function approach to constructing dictionaries
    d = dict([('one', 1), ('two', 2), ('three', 3)])    # Iterable object approach to constructing dictionaries
    return a, b, c, d

a, b, c, d = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
```

```text
a:  {}
b:  {'a': 'a', 'b': 'b', 't': 't'}
c:  {'one': 1, 'two': 2, 'three': 3}
d:  {'one': 1, 'two': 2, 'three': 3}
```

## getattr

Function: Get the attribute of python object.

Calling: `getattr(x, attr, default)`.

Input parameter:

- `x` - The object to get attribute, `x` can be all types that graph mode supports. Third-party library types are also supported when the JIT syntax support level option is 'Lax'.

- `attr` - The name of the attribute, the type of `attr` should be `str`.

- `default` - Optional input. If `x` do not have `attr`, `default` will be returned. `default` can be all types that graph mode supports. Third-party library types are also supported when the JIT syntax support level option is 'Lax'. If `default` is not set and `x` does not have attribute `attr`, AttributeError will be raised.

Return value: Target attribute or `default`.

For example:

```python
import numpy as np
import mindspore

@mindspore.jit_class
class MSClass1:
    def __init__(self):
        self.num0 = 0

ms_obj = MSClass1()

@mindspore.jit
def func(x):
    a = getattr(ms_obj, 'num0')
    b = getattr(ms_obj, 'num1', 2)
    c = getattr(x.asnumpy(), "shape", np.array([0, 1, 2, 3, 4]))
    return a, b, c

x = mindspore.tensor([-1.0], mindspore.float32)
a, b, c = func(x)
print("a: ", a)
print("b: ", b)
print("c: ", c)
```

The result is as follows:

```text
a:  0
b:  2
c:  (1,)
```

The attribute of object in graph mode may be different from that in pynative mode. It is suggested to use `default` input or call `hasattr` before using `getattr` to avoid AttributeError.

'getattr(x.asnumpy(), "shape", np.array([0, 1, 2, 3, 4]))' is a high-level usage, and more introduction can be found in the [AST Extended Syntaxes (LAX level)](https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html#ast-extended-syntaxes-lax-level) chapter.

## hasattr

Function: Judge whether an object has an attribute.

Calling: `hasattr(x, attr)`.

Input parameter:

- `x` - The object to get attribute, `x` can be all types that graph mode supports. Third-party library types are also supported when the JIT syntax support level option is 'Lax'.

- `attr` - The name of the attribute, the type of `attr` should be `str`.

Return value: boolean value indicates whether `x` has `attr`.

For example:

```python
import numpy as np
import mindspore

@mindspore.jit_class
class MSClass1:
    def __init__(self):
        self.num0 = 0

ms_obj = MSClass1()

@mindspore.jit
def func():
    a = hasattr(ms_obj, 'num0')
    b = hasattr(ms_obj, 'num1')
    c = hasattr(mindspore.tensor(np.array([1, 2, 3, 4])).asnumpy(), "__len__")
    return a, b, c

a, b, c = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
```

The result is as follows:

```text
a:  True
b:  False
c:  True
```

'hasattr(Tensor(np.array([1, 2, 3, 4])).asnumpy(), "__len__")' is a high-level usage, and more introduction can be found in the [AST Extended Syntaxes (LAX level)](https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html#ast-extended-syntaxes-lax-level) chapter.

## len

Function: Return the length of an object (string or other iterable object).

Calling: `len(sequence)`.

Input parameter: `sequence` - `Tuple`, `List`, `Dictionary`, `Tensor` or third-party object (such as numpy.ndarray).

Return value: length of the sequence, which is of the `int` type. If the input parameter is `Tensor`, the length of dimension 0 is returned.

For example:

```python
import numpy as np
import mindspore

z = mindspore.tensor(np.ones((6, 4, 5)))

@mindspore.jit()
def test(w):
    x = (2, 3, 4)
    y = [2, 3, 4]
    d = {"a": 2, "b": 3}
    n = np.array([1, 2, 3, 4])
    x_len = len(x)
    y_len = len(y)
    d_len = len(d)
    z_len = len(z)
    n_len = len(n)
    w_len = len(w.asnumpy())
    return x_len, y_len, d_len, z_len, n_len, w_len

input_x = mindspore.tensor([1, 2, 3, 4])
x_len, y_len, d_len, z_len, n_len, w_len = test(input_x)
print('x_len:{}'.format(x_len))
print('y_len:{}'.format(y_len))
print('d_len:{}'.format(d_len))
print('z_len:{}'.format(z_len))
print('n_len:{}'.format(n_len))
print('w_len:{}'.format(w_len))
```

The result is as follows:

```text
x_len:3
y_len:3
d_len:2
z_len:6
n_len:4
w_len:4
```

'len(w.asnumpy())' is a high-level usage, and more introduction can be found in the [AST Extended Syntaxes (LAX level)](https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html#ast-extended-syntaxes-lax-level) chapter.

## isinstance

Function: Determines whether an object is an instance of a class.

Calling: `isinstance(obj, type)`.

Input parameters:

- `obj` - Any instance of any supported type.

- `type` - `bool`, `int`, `float`, `str`, `list`, `tuple`, `dict`, `Tensor`, `Parameter`, or the types of third-party libraries (e.g. numpy.ndarray) or a `tuple` containing only those types.

Return value: If `obj` is an instance of `type`, return `True`. Otherwise, return `False`.

For example:

```python
import mindspore
import numpy as np

z = mindspore.tensor(np.ones((6, 4, 5)))

@mindspore.jit()
def test(w):
    x = (2, 3, 4)
    y = [2, 3, 4]
    x_is_tuple = isinstance(x, tuple)
    y_is_list = isinstance(y, list)
    z_is_tensor = isinstance(z, mindspore.Tensor)
    w_is_ndarray = isinstance(w.asnumpy(), np.ndarray)
    return x_is_tuple, y_is_list, z_is_tensor, w_is_ndarray

w = mindspore.tensor(np.array([-1, 2, 4]))
x_is_tuple, y_is_list, z_is_tensor, w_is_ndarray = test(w)
print('x_is_tuple:{}'.format(x_is_tuple))
print('y_is_list:{}'.format(y_is_list))
print('z_is_tensor:{}'.format(z_is_tensor))
print('w_is_ndarray:{}'.format(w_is_ndarray))
```

The result is as follows:

```text
x_is_tuple:True
y_is_list:True
z_is_tensor:True
w_is_ndarray:True
```

'isinstance(w.asnumpy(), np.ndarray)' is a high-level usage, and more introduction can be found in the [AST Extended Syntaxes (LAX level)](https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html#ast-extended-syntaxes-lax-level) chapter.

## all

Function: Judge whether all of the elements in the input is true.

Calling: `all(x)`.

Input parameter: - `x` - Iterable object, the valid types include `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: boolean, return `True` if all elements are `True`, otherwise `False`.

For example:

```python
import numpy as np
import mindspore

@mindspore.jit
def func():
    a = all(['a', 'b', 'c', 'd'])
    b = all(['a', 'b', '', 'd'])
    c = all([0, 1, 2, 3])
    d = all(('a', 'b', 'c', 'd'))
    e = all(('a', 'b', '', 'd'))
    f = all((0, 1, 2, 3))
    g = all([])
    h = all(())
    x = mindspore.tensor(np.array([0, 1, 2, 3]))
    i = all(x.asnumpy())
    return a, b, c, d, e, f, g, h, i

a, b, c, d, e, f, g, h, i = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
print("g: ", g)
print("h: ", h)
print("i: ", i)
```

The result is as follows:

```text
a:  True
b:  False
c:  False
d:  True
e:  False
f:  False
g:  True
h:  True
i:  False
```

'all(x.asnumpy())' is a high-level usage, and more introduction can be found in the [AST Extended Syntaxes (LAX level)](https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html#ast-extended-syntaxes-lax-level) chapter.

## any

Function: Judge whether any of the elements in the input is true.

Calling: `any(x)`.

Input parameter: - `x` - Iterable object, the valid types include `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: boolean, return `False` if all elements are `False`, otherwise `True`. Elements count as `True` except for 0, null, and `False`.

For example:

```python
import numpy as np
import mindspore

@mindspore.jit
def func():
    a = any(['a', 'b', 'c', 'd'])
    b = any(['a', 'b', '', 'd'])
    c = any([0, '', False])
    d = any(('a', 'b', 'c', 'd'))
    e = any(('a', 'b', '', 'd'))
    f = any((0, '', False))
    g = any([])
    h = any(())
    x = mindspore.tensor(np.array([0, 1, 2, 3]))
    i = any(x.asnumpy())
    return a, b, c, d, e, f, g, h, i

a, b, c, d, e, f, g, h, i = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
print("g: ", g)
print("h: ", h)
print("i: ", i)
```

The result is as follows:

```text
a:  True
b:  True
c:  False
d:  True
e:  True
f:  False
g:  False
h:  False
i:  True
```

## round

Function: Return the rounding value of input.

Calling: `round(x, digit=0)`

Input parameter:

- `x` - the object to rounded, the valid types include `int`, `float`, `bool`, `Tensor` and third-party object that defines magic function `__round__()`.

- `digit` - the number of decimal places to round, the default value is 0. `digit` can be `int` object or `None`. If `x` is `Tensor`, then `round()` does not support input `digit`.

Return value: the value after rounding.

For example:

```python
import mindspore

@mindspore.jit
def func():
    a = round(10)
    b = round(10.123)
    c = round(10.567)
    d = round(10, 0)
    e = round(10.72, -1)
    f = round(17.12, -1)
    g = round(10.17, 1)
    h = round(10.12, 1)
    return a, b, c, d, e, f, g, h

a, b, c, d, e, f, g, h = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: {:.2f}".format(e))
print("f: {:.2f}".format(f))
print("g: {:.2f}".format(g))
print("h: {:.2f}".format(h))
```

The result is as follows:

```text
a:  10
b:  10
c:  11
d:  10
e: 10.00
f: 20.00
g: 10.20
h: 10.10
```

## max

Function: Return the maximum of inputs.

Calling: `max(*data)`.

Input parameter: - `*data` - If `*data` is single input, `max` will compare all elements within `data` and `data` must be iterable object. If there are multiple inputs, then `max()` will compare each of them. The valid types of `data` include `int`, `float`, `bool`, `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: boolean, the maximum of the inputs.

For example:

```python
import numpy as np
import mindspore

@mindspore.jit
def func():
    a = max([0, 1, 2, 3])
    b = max((0, 1, 2, 3))
    c = max({1: 10, 2: 20, 3: 3})
    d = max(np.array([1, 2, 3, 4]))
    e = max(('a', 'b', 'c'))
    f = max((1, 2, 3), (1, 4))
    g = max(mindspore.tensor([1, 2, 3]))
    return a, b, c, mindspore.tensor(d), e, f, g

a, b, c, d, e, f, g = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
print("g: ", g)
```

The result is as follows:

```text
a:  3
b:  3
c:  3
d:  4
e:  c
f:  (1, 4)
g:  3
```

## min

Function: Return the minimum of inputs.

Calling: `min(*data)`.

Input parameter: - `*data` - If `*data` is single input, then `min()` will compare all elements within `data` and `data` must be iterable object. If there are multiple inputs, then `min()` will compare each of them. The valid types of `data` include `int`, `float`, `bool`, `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: boolean, the minimum of the inputs.

For example:

```python
import numpy as np
import mindspore

@mindspore.jit
def func():
    a = min([0, 1, 2, 3])
    b = min((0, 1, 2, 3))
    c = min({1: 10, 2: 20, 3: 3})
    d = min(np.array([1, 2, 3, 4]))
    e = min(('a', 'b', 'c'))
    f = min((1, 2, 3), (1, 4))
    g = min(mindspore.tensor([1, 2, 3]))
    return a, b, c, mindspore.tensor(d), e, f, g

a, b, c, d, e, f, g = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
print("g: ", g)
```

The result is as follows:

```text
a:  0
b:  0
c:  1
d:  1
e:  a
f:  (1, 2, 3)
g:  1
```

## sum

Function: Return the sum of input sequence.

Calling: `sum(x, n=0)`.

Input parameter:

- `x` - iterable with numbers, the valid types include `list`, `tuple`, `Tensor` and third-party object (such as `numpy.ndarray`).

- `n` - the number that will be added to the sum of `x`, which is assumed to be 0 if not given.

Return value: the value obtained by summing `x` and adding it to `n`.

For example:

```python
import numpy as np
import mindspore

@mindspore.jit
def func():
    a = sum([0, 1, 2])
    b = sum((0, 1, 2), 10)
    c = sum(np.array([1, 2, 3]))
    d = sum(mindspore.tensor([1, 2, 3]), 10)
    e = sum(mindspore.tensor([[1, 2], [3, 4]]))
    f = sum([1, mindspore.tensor([[1, 2], [3, 4]]), mindspore.tensor([[1, 2], [3, 4]])], mindspore.tensor([[1, 1], [1, 1]]))
    return a, b, mindspore.tensor(c), d, e, f

a, b, c, d, e, f = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
```

The result is as follows:

```text
a:  3
b:  13
c:  6
d:  16
e:  [4 6]
f:  [[ 4  6]
 [ 8 10]]
```

## abs

Function: Return the absolute value of the input.

Calling: `abs(x)`.

Input parameter: - `x` - The valid types of `x` include `int`, `float`, `bool`, `complex`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: the absolute value of the input.

For example:

```python
import mindspore

@mindspore.jit
def func():
    a = abs(-45)
    b = abs(100.12)
    c = abs(mindspore.tensor([-1, 2]).asnumpy())
    return a, b, c

a, b, c = func()
print("a: ", a)
print("b: {:.2f}".format(b))
print("c: ", c)
```

The result is as follows:

```text
a:  45
b: 100.12
c:  [1 2]
```

'abs(Tensor([-1, 2]).asnumpy())' is a high-level usage, and more introduction can be found in the [AST Extended Syntaxes (LAX level)](https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html#ast-extended-syntaxes-lax-level) chapter.

## map

Function: Maps one or more sequences based on the provided functions and generates a new sequence based on the mapping result. The current requirement is that the number of elements in multiple sequences be the same.

Calling: `map(func, sequence, ...)`.

Input parameters:

- `func` - Function.

- `sequence` - One or more sequences (`Tuple` or `List`).

Return value: Return a new sequence.

For example:

```python
import mindspore

def add(x, y):
    return x + y

@mindspore.jit()
def test():
    elements_a = (1, 2, 3)
    elements_b = (4, 5, 6)
    ret1 = map(add, elements_a, elements_b)
    elements_c = [0, 1, 2]
    elements_d = [6, 7, 8]
    ret2 = map(add, elements_c, elements_d)
    return ret1, ret2

ret1, ret2 = test()
print('ret1:{}'.format(ret1))
print('ret2:{}'.format(ret2))
```

The result is as follows:

```text
ret1: (5, 7, 9)
ret2: [6, 8, 10]
```

## zip

Function: Packs elements in the corresponding positions in multiple sequences into tuples, and then uses these tuples to form a new sequence. If the number of elements in each sequence is inconsistent, the length of the new sequence is the same as that of the shortest sequence.

Calling: `zip(sequence, ...)`.

Input parameter: `sequence` - One or more sequences (`Tuple` or `List`)`.

Return value: Return a new sequence.

For example:

```python
import mindspore

@mindspore.jit()
def test():
    elements_a = (1, 2, 3)
    elements_b = (4, 5, 6)
    ret = zip(elements_a, elements_b)
    return ret

ret = test()
print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:((1, 4), (2, 5), (3, 6))
```

## range

Function: Creates a `Tuple` based on the start value, end value, and step.

Calling:

- `range(start, stop, step)`

- `range(start, stop)`

- `range(stop)`

Input parameters:

- `start` - start value of the count. The type is `int`. The default value is 0.

- `stop` - end value of the count (exclusive). The type is `int`.

- `step` - Step. The type is `int`. The default value is 1.

Return value: Return a `Tuple`.

For example:

```python
import mindspore

@mindspore.jit()
def test():
    x = range(0, 6, 2)
    y = range(0, 5)
    z = range(3)
    return x, y, z

x, y, z = test()
print('x:{}'.format(x))
print('y:{}'.format(y))
print('z:{}'.format(z))
```

The result is as follows:

```text
x:(0, 2, 4)
y:(0, 1, 2, 3, 4)
z:(0, 1, 2)
```

## enumerate

Function: Generates an index sequence of a sequence. The index sequence contains data and the corresponding subscript.

Calling:

- `enumerate(sequence, start)`

- `enumerate(sequence)`

Input parameters:

- `sequence` - A sequence (`Tuple`, `List`, or `Tensor`).

- `start` - Start position of the subscript. The type is `int`. The default value is 0.

Return value: A `Tuple`.

For example:

```python
import mindspore
import numpy as np

y = mindspore.tensor(np.array([[1, 2], [3, 4], [5, 6]]))

@mindspore.jit()
def test():
    x = (100, 200, 300, 400)
    m = enumerate(x, 3)
    n = enumerate(y)
    return m, n

m, n = test()
print('m:{}'.format(m))
print('n:{}'.format(n))
```

The result is as follows:

```text
m:((3, 100), (4, 200), (5, 300), (6, 400))
n:((0, Tensor(shape=[2], dtype=Int64, value= [1, 2])), (1, Tensor(shape=[2], dtype=Int64, value= [3, 4])), (2, Tensor(shape=[2], dtype=Int64, value= [5, 6])))
```

## super

Function: Calls a method of the parent class (super class). Generally, the method of the parent class is called after `super`.

Calling:

- `super().xxx()`

- `super(type, self).xxx()`

Input parameters:

- `type` - Class.

- `self` - Object.

Return value: method of the parent class.

For example:

```python
import mindspore
from mindspore import nn

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

    @mindspore.jit
    def construct(self, x, y):
        ret_father_construct = super().construct(x, y)
        ret_father_test = super(SingleSubNet, self).test_father(x)
        return ret_father_construct, ret_father_test

x = 3
y = 6
z = 9
f_net = FatherNet(x)
net = SingleSubNet(x, z)
out = net(x, y)
print("out:", out)
```

The result is as follows:

```text
out: (9, 6)
```

## pow

Function: Return the power.

Calling: `pow(x, y)`

Input parameters:

- `x` - Base number, `Number`, or `Tensor`.

- `y` - Power exponent, `Number`, or `Tensor`.

Return value: `y` power of `x`, `Number`, or `Tensor`

For example:

```python
import mindspore
import numpy as np

x = mindspore.tensor(np.array([1, 2, 3]))
y = mindspore.tensor(np.array([1, 2, 3]))

@mindspore.jit()
def test(x, y):
    return pow(x, y)

ret = test(x, y)

print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:[ 1  4 27]
```

## print

Function: Prints logs.

Calling: `print(arg, ...)`.

Input parameter: `arg` - Information to be printed (`int`, `float`, `bool`, `String` or `Tensor`, or third-party library data types).

Return value: none

For example:

```python
import mindspore
import numpy as np

x = mindspore.tensor(np.array([1, 2, 3]), mindspore.int32)
y = mindspore.tensor(3, mindspore.int32)

@mindspore.jit()
def test(x, y):
    print(x)
    print(y)
    return x, y

ret = test(x, y)
```

The result is as follows:

```text
Tensor(shape=[3], dtype=Int32, value= [1 2 3])
Tensor(shape=[], dtype=Int32, value=3)
```

## filter

Function: According to the provided function to judge the elements of a sequence. Each element is passed into the function as a parameter in turn, and the elements whose return result is not 0 or False form a new sequence.

Calling: `filter(func, sequence)`

Input parameters:

- `func` - Function.

- `sequence` - A sequence (`Tuple` or `List`).

Return value: Return a new sequence.

For example:

```python
import mindspore

def is_odd(x):
    if x % 2:
        return True
    return False

@mindspore.jit()
def test():
    elements1 = (1, 2, 3, 4, 5)
    ret1 = filter(is_odd, elements1)
    elements2 = [6, 7, 8, 9, 10]
    ret2 = filter(is_odd, elements2)
    return ret1, ret2

ret1, ret2 = test()
print('ret1:{}'.format(ret1))
print('ret2:{}'.format(ret2))
```

The result is as follows:

```text
ret1:[1, 3, 5]
ret2:[7, 9]
```

## type

Function: Output the type of the input parameter.

Valid inputs: number, list, tuples, dict, numpy.ndarray, constant Tensor.

Examples of code usage are as follows:

```python
import numpy as np
import mindspore

@mindspore.jit
def func():
    a = type(1)
    b = type(1.0)
    c = type([1, 2, 3])
    d = type((1, 2, 3))
    e = type({'a': 1, 'b': 2})
    f = type(np.array([1, 2, 3]))
    g = type(mindspore.tensor([1, 2, 3]))
    return a, b, c, d, e, f, g

a, b, c, d, e, f, g = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
print("g: ", g)
```

```text
a:  <class 'int'>
b:  <class 'float'>
c:  <class 'list'>
d:  <class 'tuple'>
e:  <class 'dict'>
f:  <class 'numpy.ndarray'>
g:  <class 'mindspore.common.tensor.Tensor'>
```

> There is another way to use type as a native Python function, i.e. type(name, bases, dict) returns a class object of type name, which is not supported currently because of the low usage scenario.
