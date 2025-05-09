# 网络构建与训练常见错误分析

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_zh_cn/debug/error_analysis/mindrt_debug.md)&nbsp;&nbsp;

静态图模式下，网络构建与训练过程的常见的报错类型如下所示：

## context配置问题

执行网络训练时，需要指定后端设备，使用方式是：`set_context(device_target=device)`。MindSpore支持CPU，GPU和昇腾后端Ascend。如果在GPU设备上，错误指定后端设备为Ascend，即`set_context(device_target="Ascend")`，会得到如下报错信息：

```python
ValueError: For 'set_context', package type mindspore-gpu support 'device_target' type gpu or cpu, but got Ascend.
```

脚本设置的运行后端要求与实际的硬件设备相匹配。

参考实例链接：

[MindSpore 配置问题 - 'set_context'配置报错](https://www.hiascend.com/developer/blog/details/0229106885219029083)。

关于context配置的详细使用说明请参考['set_context'](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore/mindspore.set_context.html)。

## 语法问题

### construct函数参数错误

MindSpore中神经网络的基本构成单元为`nn.Cell`。模型或神经网络层应当继承该基类。基类的成员函数`construct`是定义要执行的计算逻辑，所有继承类都必须重写此方法。`construct`函数的定义原型为：

```python
def construct(self, *inputs, **kwargs):
```

在重写该函数时，有时会出现下面的错误信息：

```python
TypeError: The function construct needs 0 positional argument and 0 default argument, but provided 1
```

这是因为，用户自定义实现`construct`函数时，函数参数列表错误，缺少`self`，例如`"def construct(*inputs, **kwargs):"`。此时，MindSpore在进行语法解析时发生报错。

参考实例链接：

[MindSpore 语法问题 - 'construct' 函数定义报错](https://www.hiascend.com/developer/blog/details/0230106556970619074)。

### 控制流语法错误

静态图模式下，Python代码并不是由Python解释器去执行，而是将代码编译成静态计算图，然后执行静态计算图。MindSpore支持的控制流语法涉及if语句、for语句以及while语句。if语句可能存在不同分支返回对象的属性不一致，导致报错。报错信息如下所示：

```c++
TypeError: Cannot join the return values of different branches, perhaps you need to make them equal.
Type Join Failed: dtype1 = Float32, dtype2 = Float16.
```

此时由报错信息可知，报错原因是if语句不同分支返回值的类型不一致：一个是float32，另一个是float16，导致编译报错。

```c++
ValueError: Cannot join the return values of different branches, perhaps you need to make them equal.
Shape Join Failed: shape1 = (2, 3, 4, 5), shape2 = ().
```

由报错信息可知，报错原因是if语句不同分支返回值的维度shape不一致：一个是`2*3*4*5`的四位Tensor，另一个是标量，导致编译报错。

参考实例链接：

[MindSpore 语法问题 - Type(Shape) Join Failed](https://www.mindspore.cn/docs/zh-CN/r2.6.0/faq/network_compilation.html?highlight=type%20join%20failed)

for语句以及while语句可能存在循环次数过大，导致函数调用栈超限的问题。报错信息如下所示：

```c++
RuntimeError: Exceed function call depth limit 1000, (function call depth: 1001, simulate call depth: 997).
```

超出函数调用栈限制问题，一种解决方式是简化网络的结构，减少循环次数。另一种方式是使用`mindspore.set_recursion_limit(recursion_limit=value)`调大函数调用栈的阈值。

参考实例链接：

[MindSpore 语法问题 - Exceed function call depth limit](https://www.hiascend.com/developer/blog/details/0223111589074862027)。

## 算子编译错误

算子编译错误主要是由于输入参数不符合要求，算子功能不支持等问题。

例如，使用ReduceSum算子时，输入数据超过八维时报错信息如下：

```c++
RuntimeError: ({'errCode': 'E80012', 'op_name': 'reduce_sum_d', 'param_name': 'x', 'min_value': 0, 'max_value': 8, 'real_value': 10}, 'In op, the num of dimensions of input/output[x] should be in the range of [0, 8], but actually is [10].')
```

参考实例链接：

[MindSpore 算子编译问题 - ReduceSum算子不支持八维以上输入](https://www.hiascend.com/developer/blog/details/0229108037306667164)

例如，Parameter参数不支持类型自动转换，使用Parameter算子时，进行数据类型转换时报错，报错信息如下：

```c++
RuntimeError: Data type conversion of 'Parameter' is not supported, so data type int32 cannot be converted to data type float32 automatically.
```

参考实例链接：

[MindSpore 算子编译问题 - ScatterNdUpdate算子参数类型不一致报错](https://www.hiascend.com/developer/blog/details/0232107351416081120)

另外，有时候在算子编译过程中会出现`Response is empty`、`Try to send request before Open()`、`Try to get response before Open()`这一类的报错，如下所示：

```c++
>       result = self._graph_executor.compile(obj, args_list, phase, self._use_vm_mode())
E       RuntimeError: Response is empty
E
E       ----------------------------------------------------
E       - C++ Call Stack: (For framework developers)
E       ----------------------------------------------------
E       mindspore/ccsrc/backend/common/session/kernel_build_client.h:100 Response
```

这个问题的直接原因一般是算子编译的子进程挂了或者调用阻塞卡住导致的超时，可以从以下几个方面进行排查：

1. 检查日志，在这个错误前是否有其他错误日志，如果有请先解决前面的错误，一些算子相关的问题（比如昇腾上TBE包没装好，GPU上没有nvcc）会导致后续的此类报错；

2. 如果有使用图算融合特性，有可能是图算的AKG算子编译卡死超时导致，可以尝试关闭图算特性；

3. 在昇腾上可以尝试减少算子并行编译的进程数；

4. 检查主机的内存和cpu占用情况，有可能是主机的内存和cpu占用过高，导致算子编译进程无法启动，出现了编译失败，可以尝试找出占用内存和cpu过高的进程，对其进行优化；

5. 如果是在云上的训练环境遇到这个问题，可以尝试重启内核。

## 算子执行错误

算子执行问题，发生的原因主要包括输入数据问题、算子实现问题以及算子初始化问题等场景。算子执行错误的分析方法一般可采用类比法。

具体分析可参考实例：

[MindSpore 算子执行错误 - nn.GroupNorm算子输出异常](https://www.hiascend.com/developer/blog/details/0229107351277363132)。

## 资源不足

在调试网络的时候，经常会遇到`Out Of Memory`报错，MindSpore在Ascend设备上对内存分成4层进行管理。包括Runtime、Context、双游标和内存复用。

关于MindSpore在昇腾后端（Ascend）上的内存管理及常见问题的具体内容，请参考[MindSpore Ascend 内存管理](https://www.hiascend.com/developer/blog/details/0229107352026042135)。
