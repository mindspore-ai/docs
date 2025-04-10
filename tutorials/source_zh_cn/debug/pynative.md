# 动态图调试

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_zh_cn/debug/pynative.md)

## 概述

在MindSpore中，动态图模式又被称为PyNative模式，该模式下支持按照Python的语法去执行。在网络开发中，更容易进行调试，支持单算子执行，逐行语句执行，查看中间变量等等调试手段。本教程主要介绍使用MindSpore动态图模式调试的基本方法。

## 基本方法

在动态图下执行出错时，往往会有Python调用栈，但是由于MindSpore存在算子下发的多级流水以及host和device的异步执行，有时该调用栈并不准确，当Python捕获到异常时，往往已经跑过了真正出错的地方。此外，在正常开发调试脚本时，往往需要查看传递的变量是否准确，函数的调用是否准确。因此在动态图下调试时，我们需要进行一些常用的调试步骤，例如打断点、查看日志、查看变量值、单步执行等等。

## 同步模式定位问题

由于MindSpore动态图下框架存在多线程异步行为，因此当Python调用栈存在不准确的场景时，我们需要先开启同步模式，来找到精准的报错调用栈。

### 设置同步模式

当出现错误时，首先设置context，将动态图设置成同步执行模式：

```python
import mindspore
mindspore.runtime.launch_blocking()
```

设置完成后，重新执行脚本。此时脚本出错的时候就会出错在正确的调用栈位置了，可以根据调用栈信息区分不同的错误类型。

1. 调用栈出错的位置为与MindSpore框架API无关的地方，需要检查一下是否为Python语法问题。
2. 调用栈出错在MindSpore框架API，常见的错误位置如下：

   - 出错在前向API上，调用栈最后在如下两种Python函数上：

     ```python
     self._executor.run_op_async(*args)
     ```

     或者

     ```python
     pyboost_xxx
     ```

     其中针对`pyboost_xxx`情况，`xxx`为具体的算子接口名称。

     针对该类报错，需要检查输入数据的shape和dtype类型，类型是否符合该API的要求。

   - 出现在反向传播

     ```python
     self._executor.new_graph(obj, *args, *(kwargs.values()))
     self._executor.end_graph(obj, output, *args, *(kwargs.values()))
     self._executor.grad(grad, obj, weights, grad_position, *args)
     ```

     此时可以根据报错检查一下是否是输入类型，输出类型以及使用的求导接口输入有错。

## 单步调试定位问题

### 设置断点

在实际开发过程中，可以通过PyCharm，VsCode等IDE在图形化界面中进行断点设置，也可以通过直接在Python脚本中插入pdb，添加断点。例如：

```python
def some_function():
    x = 10
    import pdb; pdb.set_trace()  # 在这里设置断点
    y = x + 1
    return y
```

### 单步执行

在IDE中，可以使用调试工具栏中的“Step Over”、“Step Into”、“Step Out”按钮来单步执行代码，在Python命令行中可以使用交互命令 `n(next), s(step)` 命令来进行逐行执行。此外，可以通过在断点处观察和打印变量值，来判断脚本的执行结果是否准确。

### 日志记录

在调试过程中，往往需要通过查看日志来定位问题，在MindSpore中，可以通过GLOG_v来进行日志级别的控制，默认值：2，具体级别如下：

- 0-DEBUG
- 1-INFO
- 2-WARNING
- 3-ERROR，表示程序执行出现报错，输出错误日志，程序可能不会终止
- 4-CRITICAL，表示程序执行出现异常，将会终止执行程序

详细的日志控制方法见[环境变量](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/env_var_list.html#日志)。

### 常见PDB调试命令

- c (continue): 继续执行直到下一个断点。
- n (next): 执行下一行代码。
- s (step): 执行下一行代码，如果下一行代码是函数调用，则进入函数内部。
- l (list): 显示当前执行的代码行周围的源代码。
- p (print): 打印表达式的值。
- q (quit): 退出pdb调试器。
- h (help): 显示帮助信息。

## 反向问题定位

在动态图下需要查看反向精度是否准确，往往可以利用动态图的反向hook功能，来查看在反向传播中的梯度是否符合预期。

- 查看Parameter的梯度是否符合预期，可以通过对Parameter注册hook

  通过

  ```python
  register_hook(hook_fn)
  ```

  来注册hook，例如：

  ```python
  import mindspore
  from mindspore import Tensor
  def hook_fn(grad):
      return grad * 2

  def hook_test(x, y):
      z = x * y
      z.register_hook(hook_fn)
      z = z * y
      return z

  ms_grad = mindspore.value_and_grad(hook_test, grad_position=(0,1))
  output = ms_grad(Tensor(1, mindspore.float32), Tensor(2, mindspore.float32))
  print(output)
  ```

  详细API使用说明可以[参考](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore/Tensor/mindspore.Tensor.register_hook.html#mindspore.Tensor.register_hook)。

- 可以通过`mindspore.ops.HookBackward`查看执行过程中的梯度，例如：

  ```python
  import mindspore
  from mindspore import ops
  from mindspore import Tensor
  def hook_fn(grad):
      print(grad)

  hook = ops.HookBackward(hook_fn)
  def hook_test(x, y):
      z = x * y
      z = hook(z)
      z = z * y
      return z

  def backward(x, y):
      return mindspore.value_and_grad(hook_test, grad_position=(0,1))(x, y)

  output = backward(Tensor(1, mindspore.float32), Tensor(2, mindspore.float32))

  print(output)
  ```

  详细API使用说明可以[参考](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.HookBackward.html)。

- 可以通过`mindspore.nn.Cell.register_backward_hook`查看某个Cell的梯度，例如：

  ```python
  import numpy as np
  import mindspore
  from mindspore import Tensor, nn, ops
  def backward_hook_fn(cell_id, grad_input, grad_output):
      print("backward input: ", grad_input)
      print("backward output: ", grad_output)

  class Net(nn.Cell):
      def __init__(self):
          super(Net, self).__init__()
          self.relu = nn.ReLU()
          self.handle = self.relu.register_backward_hook(backward_hook_fn)

      def construct(self, x):
          x = x + x
          x = self.relu(x)
          return x

  net = Net()
  output = mindspore.value_and_grad(net, grad_position=(0,1))(Tensor(np.ones([1]).astype(np.float32)))

  print(output)
  ```

  详细API使用说明可以[参考](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_backward_hook)。

## 更多实际案例

参考[调试案例](https://www.hiascend.com/forum/forum-0106101385921175002-1.html?filterCondition=1&topicClassId=0631105934233557004)。