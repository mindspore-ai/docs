# Calling the Custom Class

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.1/tutorials/experts/source_en/network/jit_class.md)

## Overview

In static graph mode, using `jit_class` to decorate a custom class, users can create and call the instance of this custom class, and obtain attributes and methods for that custom class.

`jit_class` is applied to static graph mode, expanding the scope of support for improving static graph compilation syntax. In dynamic graph mode, that is, PyNative mode, the use of `jit_class` does not affect the execution logic of the PyNative mode.

This document describes how to use `jit_class` decorator so that you can use `jit_class` decorator more effectively.

## jit_class Decorates Custom Class

After decorating a custom class with `@jit_class`, you can create and call the instance of the custom class and obtain the attributes and methods.

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class InnerNet:
    value = ms.Tensor(np.array([1, 2, 3]))

class Net(nn.Cell):
    def construct(self):
        return InnerNet().value

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
out = net()
print(out)
```

```text
[1 2 3]
```

`jit_class` supports nesting use of the custom class, nesting uses scenarios of custom classes and nn. Cell. It should be noted that when a class inherits, if the parent class is decorated with `jit_class`, the subclass will also have the ability to `jit_class`.

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class Inner:
    def __init__(self):
        self.value = ms.Tensor(np.array([1, 2, 3]))

@ms.jit_class
class InnerNet:
    def __init__(self):
        self.inner = Inner()

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.inner_net = InnerNet()

    def construct(self):
        out = self.inner_net.inner.value
        return out

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
out = net()
print(out)
```

```text
[1 2 3]
```

`jit_class` only support decorating custom classes, not nn. Cell and nonclass types. If you execute the following use case, an error will appear.

```python
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class Net(nn.Cell):
    def construct(self, x):
        return x

ms.set_context(mode=ms.GRAPH_MODE)
x = ms.Tensor(1)
net = Net()
net(x)
```

The error information is as follows:

```text
TypeError: Decorator jit_class is used for user-defined classes and cannot be used for nn.Cell: Net<>.
```

```python
import mindspore as ms

@ms.jit_class
def func(x, y):
    return x + y

func(1, 2)
```

The error information is as follows:

```text
TypeError: Decorator jit_class can only be used for class type, but got <function func at 0x7fee33c005f0>.
```

## Obtaining the Attributes and Methods of the Custom Class

Support calling the attributes and methods of a class by its class name or its instance.

```python
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class InnerNet:
    def __init__(self, val):
        self.number = val

    def act(self, x, y):
        return self.number * (x + y)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.inner_net = InnerNet(2)

    def construct(self, x, y):
        return self.inner_net.number + self.inner_net.act(x, y)

ms.set_context(mode=ms.GRAPH_MODE)
x = ms.Tensor(2, dtype=ms.int32)
y = ms.Tensor(3, dtype=ms.int32)
net = Net()
out = net(x, y)
print(out)
```

```text
12
```

## Creating Instance of the Custom Class

In the static graph mode, when you create the instance of the custom class, the parameter requirement is a constant.

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class InnerNet:
    def __init__(self, val):
        self.number = val + 3

class Net(nn.Cell):
    def construct(self):
        net = InnerNet(2)
        return net.number

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
out = net()
print(out)
```

```text
5
```

For other scenarios, when creating an instance of a custom class, there is a restriction that no parameters must be constants. For example, the following use case:

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class InnerNet:
    def __init__(self, val):
        self.number = val + 3

class Net(nn.Cell):
    def __init__(self, val):
        super(Net, self).__init__()
        self.inner = InnerNet(val)

    def construct(self):
        return self.inner.number

ms.set_context(mode=ms.GRAPH_MODE)
x = ms.Tensor(2, dtype=ms.int32)
net = Net(x)
out = net()
print(out)
```

```text
5
```

## Calling the Instance of the Custom Class

When you call an instance of a custom class, the `__call__` function method of that class is called.

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class InnerNet:
    def __init__(self, number):
        self.number = number

    def __call__(self, x, y):
        return self.number * (x + y)

class Net(nn.Cell):
    def construct(self, x, y):
        net = InnerNet(2)
        out = net(x, y)
        return out

ms.set_context(mode=ms.GRAPH_MODE)
x = ms.Tensor(2, dtype=ms.int32)
y = ms.Tensor(3, dtype=ms.int32)
net = Net()
out = net(x, y)
print(out)
```

```text
10
```

If the class does not define the `__call__` function, an error message will be reported. If you execute the following use case, an error will appear.

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class InnerNet:
    def __init__(self, number):
        self.number = number

class Net(nn.Cell):
    def construct(self, x, y):
        net = InnerNet(2)
        out = net(x, y)
        return out

ms.set_context(mode=ms.GRAPH_MODE)
x = ms.Tensor(2, dtype=ms.int32)
y = ms.Tensor(3, dtype=ms.int32)
net = Net()
out = net(x, y)
print(out)
```

The error information is as follows:

```text
RumtimeError: MsClassObject: 'InnerNet' has no `__call__` function, please check the code.
```