# Network Construction and Training Error Analysis

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/tutorials/experts/source_en/debug/mindrt_debug.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>&nbsp;&nbsp;

The following lists the common network construction and training errors in static graph mode.

## Incorrect Context Configuration

When performing network training, you must run the following command to specify the backend device: `set_context(device_target=device)`. MindSpore supports CPU, GPU, and Ascend. If a GPU backend device is incorrectly specified as Ascend by running `set_context(device_target="Ascend")`, the following error message will be displayed:

```python
ValueError: For 'set_context', package type mindspore-gpu support 'device_target' type gpu or cpu, but got Ascend.
```

The running backend specified by the script must match the actual hardware device.

For details, visit the following website:

[MindSpore Configuration Error - 'set_context' Configuration Error](https://bbs.huaweicloud.com/forum/thread-183514-1-1.html)

For details about the context configuration, see ['set_context'](https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore/mindspore.set_context.html).

## Syntax Errors

- Incorrect construct parameter

    In MindSpore, the basic unit of the neural network is `nn.Cell`. All the models or neural network layers must inherit this base class. As a member function of this base class, `construct` defines the calculation logic to be executed and must be rewritten in all inherited classes. The function prototype of `construct` is:

    ```python
    def construct(self, *inputs, **kwargs):
    ```

    When the function is rewritten, the following error message may be displayed:

    ```python
    TypeError: The function construct needs 0 positional argument and 0 default argument, but provided 1
    ```

    This is because the function parameter list is incorrect when the user-defined `construct` function is implemented, for example, `"def construct(*inputs, **kwargs):"`, where `self` is missing. In this case, an error is reported when MindSpore parses the syntax.

    For details, visit the following website:

    [MindSpore Syntax Error - 'construct' Definition Error](https://bbs.huaweicloud.com/forum/thread-178902-1-1.html)

- Incorrect control flow syntax

    In static graph mode, Python code is not executed by the Python interpreter. Instead, the code is built into a static computational graph for execution. The control flow syntax supported by MindSpore includes if, for, and while statements. The attributes of the objects returned by different branches of the if statement may be inconsistent. As a result, an error is reported. The error message is displayed as follows:

    ```c++
    TypeError: Cannot join the return values of different branches, perhaps you need to make them equal.
    Type Join Failed: dtype1 = Float32, dtype2 = Float16.
    ```

    According to the error message, the return values of different branches of the if statement are of different types. One is float32, and the other is float16. As a result, a build error is reported.

    ```c++
    ValueError: Cannot join the return values of different branches, perhaps you need to make them equal.
    Shape Join Failed: shape1 = (2, 3, 4, 5), shape2 = ().
    ```

    According to the error message, the dimension shapes of the return values of different branches of the if statement are different. One is a 4-bit tensor of `2*3*4*5`, and the other is a scalar. As a result, a build error is reported.

    For details, visit the following website:

    [MindSpore Syntax Error - Type (Shape) Join Failed](https://www.mindspore.cn/docs/en/r2.0.0-alpha/faq/network_compilation.html)

    The number of loops of the for and while statements may exceed the permitted range. As a result, the function call stack exceeds the threshold. The error message is displayed as follows:

    ```c++
    RuntimeError: Exceed function call depth limit 1000, (function call depth: 1001, simulate call depth: 997).
    ```

    One solution to the problem that the function call stack exceeds the threshold is to simplify the network structure and reduce the number of loops. Another method is to use `set_context(max_call_depth=value)` to increase the threshold of the function call stack.

    For details, visit the following website:

    [MindSpore Syntax Error - Exceed function call depth limit](https://bbs.huaweicloud.com/forum/thread-182165-1-1.html)

## Operator Build Errors

Operator build errors are mainly caused by input parameters that do not meet requirements or operator functions that are not supported.

For example, when the ReduceSum operator is used, the following error message is displayed if the input data exceeds eight dimensions:

```c++
RuntimeError: ({'errCode': 'E80012', 'op_name': 'reduce_sum_d', 'param_name': 'x', 'min_value': 0, 'max_value': 8, 'real_value': 10}, 'In op, the num of dimensions of input/output[x] should be in the range of [0, 8], but actually is [10].')
```

For details, visit the following website:

[MindSpore Operator Build Error - ReduceSum Operator Does Not Support Input of More Than Eight Dimensions](https://bbs.huaweicloud.com/forum/thread-182168-1-1.html)

For example, the Parameter parameter does not support automatic type conversion. When the Parameter operator is used, an error is reported during data type conversion. The error message is as follows:

```c++
RuntimeError: Data type conversion of 'Parameter' is not supported, so data type int32 cannot be converted to data type float32 automatically.
```

For details, visit the following website:

[MindSpore Operator Build Error - Error Reported Due to Inconsistent ScatterNdUpdate Operator Parameter Types](https://bbs.huaweicloud.com/forum/thread-182175-1-1.html)

## Operator Execution Errors

Operator execution errors are mainly caused by improper input data , operator implementation, or operator initialization. Generally, the analogy method can be used to analyze operator execution errors.

For details, see the following example:

[MindSpore Operator Execution Error - nn.GroupNorm Operator Output Exception](https://bbs.huaweicloud.com/forum/thread-182191-1-1.html)

## Insufficient Resources

During network debugging, `Out Of Memory` errors often occur. MindSpore divides the memory into four layers for management on the Ascend device, including runtime, context, dual cursors, and memory overcommitment.

For details about memory management and FAQs of MindSpore on the Ascend device, see [MindSpore Ascend Memory Management](https://bbs.huaweicloud.com/forum/thread-171161-1-1.html).
