# Dynamic Graph Debugging

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/debug/pynative.md)

## Overview

In MindSpore, the dynamic graph mode, also known as PyNative mode, supports execution according to Python syntax. In network development, it is easier to debug, supporting single-operator execution, line-by-line statement execution, viewing intermediate variables, and other debugging tools. This tutorial introduces the basic method of debugging using MindSpore dynamic graph mode.

## Basic Method

When there is an execution error under a dynamic graph, there is often a Python call stack, but due to the multi-level flow of arithmetic downstream and asynchronous execution of host and device in MindSpore, sometimes that call stack is not accurate, and by the time Python catches an exception, it has often already run through the place where the real error occurred. In addition, in normal development of debugging scripts, it is often necessary to see if the passed variables are accurate and if the function calls are accurate. Therefore, when debugging under dynamic graphs, we need to perform some common debugging steps, such as breakpoints, checking logs, checking variable values, single-step execution.

## Synchronized Mode Positioning Problems

Because of the multi-threaded asynchronous behavior of the framework under the MindSpore dynamic graph, when there is an inaccurate scenario of the Python call stack, we need to turn on the synchronous mode first to find the accurate error reporting call stack.

### Setting Synchronous Mode

When an error occurs, set the context first, then set the dynamic graph into synchronized execution mode:

```python
import mindspore
mindspore.runtime.launch_blocking()
```

After the setup is complete, re-execute the script. At this point, the script error will be accurately error in the correct call stack, you can call the stack information to distinguish between different types of errors.

1. Call stack errors are in places that are not related to the MindSpore Framework API and need to be checked for Python syntax issues.
2. Call stack error on MindSpore Framework API, common error locations are as follows:

   - Error on the forward API, the call stack ends up on the following two Python functions:

     ```python
     self._executor.run_op_async(*args)
     ```

     or

     ```python
     pyboost_xxx
     ```

     where for the `pyboost_xxx` case, `xxx` is the name of the specific operator interface.

     For this type of error, you need to check the shape and dtype type of the input data, and whether the type meets the requirements of this API.

   - Appears in backpropagation

     ```python
     self._executor.new_graph(obj, *args, *(kwargs.values()))
     self._executor.end_graph(obj, output, *args, *(kwargs.values()))
     self._executor.grad(grad, obj, weights, grad_position, *args)
     ```

     At this point, you can check to see if there is an error in the input type, the output type, and the input of the derivation interface used.

## Single-step Debugging Positioning Problems

### Setting Breakpoints

In the actual development process, you can use PyCharm, VsCode and other IDEs to set breakpoints in the graphical interface, or you can add breakpoints by inserting a pdb directly into the Python script. For example:

```python
def some_function():
    x = 10
    import pdb; pdb.set_trace()  # Set a breakpoint here
    y = x + 1
    return y
```

### Single-step Execution

In the IDE, you can use the `Step Over`, `Step Into`, and `Step Out` buttons in the debugging toolbar to execute code in a single step. On the Python command line, you can use the interactive commands `n(next), s(step)` to perform line-by-line execution. In addition, you can determine the accuracy of a script's execution results by observing and printing the values of variables at breakpoints.

### Log Records

In the debugging process, it is often necessary to view the log to locate the problem, in MindSpore, you can use GLOG_v to log level control, the default value: 2, the specific level is as follows:

- 0-DEBUG
- 1-INFO
- 2-WARNING
- 3-ERROR, indicates that there is an error in the execution of the program, the error log is output, and the program may not be terminated.
- 4-CRITICAL, indicates that the program execution is abnormal and will be terminated.

See [environment variables](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/env_var_list.html#log) for detailed logging controls.

### Common PDB Debugging Commands

- c (continue): Continues execution until the next breakpoint.
- n (next): Execute the next line of code.
- s (step): Execute the next line of code, or go inside a function if the next line of code is a function call.
- l (list): Show the source code around the currently executed line.
- p (print): Prints the value of an expression.
- q (quit): Quit the pdb debugger.
- h (help): Display help information.

## Reverse Problem Localization

When you need to see if the backpropagation accuracy is accurate under a dynamic graph, you can often utilize the backhook function of the dynamic graph to see if the gradient in the backpropagation is as expected.

- To see if the gradient of a Parameter is as expected, you can register a hook with the Parameter:

  Through

  ```python
  register_hook(hook_fn)
  ```

  to register hook, such as

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

  Detailed API usage instructions can be [referenced](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/mindspore/Tensor/mindspore.Tensor.register_hook.html#mindspore.Tensor.register_hook).

- Viewing the gradient during execution can be done with `mindspore.ops.HookBackward`, for example:

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

  Detailed API usage instructions can be [referenced](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/ops/mindspore.ops.HookBackward.html).

- Viewing the gradient of a particular Cell can be done with `mindspore.nn.Cell.register_backward_hook`, for example:

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

  Detailed API usage instructions can be [referenced](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_backward_hook).

## More Practical Examples

Refer to [debugging case](https://www.hiascend.com/forum/forum-0106101385921175002-1.html?filterCondition=1&topicClassId=0631105934233557004).