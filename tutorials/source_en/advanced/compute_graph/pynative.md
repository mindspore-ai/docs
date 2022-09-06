# Dynamic Graph Mode Application

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced/compute_graph/pynative.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

In dynamic graph mode, MindSpore supports single-operator execution, common function execution, network execution, and independent gradient computation. The following uses sample code to describe how to use these operations and precautions.

## Operations

First, import related dependencies and set the running mode to dynamic graph mode.

```python
import numpy as np
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE)
```

### Executing a Single Operator

The following is the sample code for executing the addition operator [mindspore.ops.Add](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Add.html#mindspore.ops.Add):

```python
add = ops.Add()
x = ms.Tensor(np.array([1, 2]).astype(np.float32))
y = ms.Tensor(np.array([3, 5]).astype(np.float32))
z = add(x, y)
print("x:", x.asnumpy(), "\ny:", y.asnumpy(), "\nz:", z.asnumpy())
```

```text
    x: [1. 2.]
    y: [3. 5.]
    z: [4. 7.]
```

### Executing a Function

Execute the customized function `add_func`. The sample code is as follows:

```python
add = ops.Add()

def add_func(x, y):
    z = add(x, y)
    z = add(z, x)
    return z

x = ms.Tensor(np.array([1, 2]).astype(np.float32))
y = ms.Tensor(np.array([3, 5]).astype(np.float32))
z = add_func(x, y)
print("x:", x.asnumpy(), "\ny:", y.asnumpy(), "\nz:", z.asnumpy())
```

```text
    x: [1. 2.]
    y: [3. 5.]
    z: [5. 9.]
```

### Executing a Network

Execute the customized network `Net` and define the network structure in construct. The sample code is as follows:

```python
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()

    def construct(self, x, y):
        return self.mul(x, y)

net = Net()
x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
y = ms.Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))
z = net(x, y)

print("x:", x.asnumpy(), "\ny:", y.asnumpy(), "\nz:", z.asnumpy())
```

```text
    x: [1. 2. 3.]
    y: [4. 5. 6.]
    z: [ 4. 10. 18.]
```

## Synchronous Execution

In dynamic graph mode, operators are executed asynchronously on the device to improve performance. Therefore, when an operator execution error occurs, the error information may be displayed at the end of the program execution. To solve this problem, the pynative_synchronize setting is added to MindSpore to determine whether to use asynchronous execution on the operator device.

In dynamic graph mode, operators are executed asynchronously by default. You can set context to determine whether to execute operators asynchronously. When an operator fails to be executed, you can easily view the location of the error code through the call stack. The sample code is as follows:

```python
import mindspore as ms

# Set pynative_synchronize to synchronize operators.
ms.set_context(mode=ms.PYNATIVE_MODE, pynative_synchronize=True)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.get_next = ops.GetNext([ms.float32], [(1, 1)], 1, "test")

    def construct(self, x1,):
        x = self.get_next()
        x = x + x1
        return x

ms.set_context()
x1 = np.random.randn(1, 1).astype(np.float32)
net = Net()
output = net(ms.Tensor(x1))
print(output.asnumpy())
```

Output: In this case, the operator is executed synchronously. When an error occurs during operator execution, you can view the complete call stack and find the code line where the error occurs.

```text
Traceback (most recent call last):
  File "test.py", line 24, in <module>
    output = net(ms.Tensor(x1))
  File ".../mindspore/nn/cell.py", line 602, in __call__
    raise err
  File ".../mindspore/nn/cell.py", line 599, in __call__
    output = self._run_construct(cast_inputs, kwargs)
  File ".../mindspore/nn/cell.py", line 429, in _run_construct
    output = self.construct(*cast_inputs, **kwargs)
  File "test.py", line 17, in construct
    x = self.get_next()
  File ".../mindspore/ops/primitive.py", line 294, in __call__
    return _run_op(self, self.name, args)
  File ".../mindspore/common/api.py", line 90, in wrapper
    results = fn(*arg, **kwargs)
  File ".../mindspore/ops/primitive.py", line 754, in _run_op
    output = real_run_op(obj, op_name, args)
RuntimeError: mindspore/ccsrc/plugin/device/gpu/kernel/data/dataset_iterator_kernel.cc:139 Launch] For 'GetNext', gpu Queue(test) Open Failed: 2
```
