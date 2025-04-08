# mindspore.jit 实践

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_zh_cn/compile/jit_compilation.md)

在本节中，我们将进一步探讨MindSpore的工作原理，以及如何使其高效运行。`mindspore.jit()` 转换会执行对MindSpore Python函数的即时编译（just-in-time compilation），以便在后续过程中高效执行。它发生在函数第一次执行的时候，这个过程会花费一些时间。

## 对函数进行JIT编译

### 函数定义

```python
from mindspore import Tensor

def f(a: Tensor, b: Tensor, c: Tensor):
    return a * b + c
```

### 使用 `mindspore.jit` 包装

```python
import mindspore

jitted_f = mindspore.jit(f)
```

### 运行

```python
import numpy as np
import mindspore
from mindspore import Tensor

# 构造数据
f_input = [Tensor(np.random.randn(2, 3), mindspore.float32) for _ in range(3)]

# 运行原始函数
out = f(*f_input)
print(f"{out=}")

# 运行jit转换后的函数
out = jitted_f(*f_input)
print(f"{out=}")
```

> mindspore.jit不能在终端中使用临时源代码进行编译，必须作为`.py`文件运行。

## 更多的用法

### 常用配置介绍

`mindspore.jit`接口详情见[API 文档](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore/mindspore.jit.html)，常用配置如下：

- capture_mode: 用于指定创建`图`的方式（如：`ast`通过解析Python构建， `bytecode`通过解析Python字节码构建， `trace`通过追踪Python代码的执行进行构建。）
- jit_level: 用于控制编译优化的级别。（如： 默认O0， 使用更多的优化可选择O1）
- fullgraph: 是否将整个函数编译为`图`，默认为False，jit会尽可能兼容函数中的Python语法，设置为True一般可以获得更好的性能，但对语法要求更高。
- backend: 用于指定编译的后端。

### 使用方法

下面分别给出了`ast`、`bytecode`和`trace`方式下的用法。

```python
import mindspore

# 使用ast方式构建图
jitted_by_ast_and_levelO0_f = mindspore.jit(f, capture_mode="ast", jit_level="O0") # 这个是默认配置，跟上面的jitted_f是一样的
jitted_by_ast_and_levelO1_f = mindspore.jit(f, capture_mode="ast", jit_level="O1")
jitted_by_ast_and_ge_f = mindspore.jit(f, capture_mode="ast", backend="GE")

# 使用bytecode方式构建图
jitted_by_bytecode_and_levelO0_f = mindspore.jit(f, capture_mode="bytecode", jit_level="O0")
jitted_by_bytecode_and_levelO1_f = mindspore.jit(f, capture_mode="bytecode", jit_level="O1")
jitted_by_bytecode_and_ge_f = mindspore.jit(f, capture_mode="bytecode", backend="GE")


# 使用trace方式构建图，不支持直接通过mindspore.jit(f, capture_mode="trace", ...)的方式转换
@mindspore.jit(capture_mode="trace", jit_level="O0")
def jitted_by_trace_and_levelO0_f(a, b, c):
    return a * b + c

@mindspore.jit(capture_mode="trace", jit_level="O1")
def jitted_by_trace_and_levelO1_f(a, b, c):
    return a * b + c

@mindspore.jit(capture_mode="trace", backend="GE")
def jitted_by_trace_and_ge_f(a, b, c):
    return a * b + c

# 使用fullgraph (这里以ast为例子)
jitted_by_ast_and_levelO0_fullgraph_f = mindspore.jit(f, capture_mode="ast", jit_level="O0", fullgraph=True)
jitted_by_ast_and_levelO1_fullgraph_f = mindspore.jit(f, capture_mode="ast", jit_level="O1", fullgraph=True)
jitted_by_ast_and_ge_fullgraph_f = mindspore.jit(f, capture_mode="ast", backend="GE", fullgraph=True)


# 用字典记录，方便后续调用
function_dict = {
    "function ": f,

    "function jitted by ast and levelO0": jitted_by_ast_and_levelO0_f,
    "function jitted by ast and levelO1": jitted_by_ast_and_levelO1_f,
    "function jitted by ast and ge": jitted_by_ast_and_ge_f,

    "function jitted by bytecode and levelO0": jitted_by_bytecode_and_levelO0_f,
    "function jitted by bytecode and levelO1": jitted_by_bytecode_and_levelO1_f,
    "function jitted by bytecode and ge": jitted_by_bytecode_and_ge_f,

    "function jitted by trace and levelO0": jitted_by_trace_and_levelO0_f,
    "function jitted by trace and levelO1": jitted_by_trace_and_levelO1_f,
    "function jitted by trace and ge": jitted_by_trace_and_ge_f,

    "function jitted by ast and levelO0 fullgraph": jitted_by_ast_and_levelO0_fullgraph_f,
    "function jitted by ast and levelO1 fullgraph": jitted_by_ast_and_levelO1_fullgraph_f,
    "function jitted by ast and ge fullgraph": jitted_by_ast_and_ge_fullgraph_f
}
```

> 当构建图的方式选择为trace的时候不支持直接通过`mindspore.jit(f, capture_mode="trace", ...)`的方式转换，需要通过装饰器`@mindspore.jit(capture_mode="trace", ...)`用法对函数进行包装。

### 运行

```python
# 构造数据
dataset = [[Tensor(np.random.randn(2, 3), mindspore.float32) for _ in range(3)] for i in range(1000)]

for s, f in function_dict.items():
    s_time = time.time()

    out = f(*dataset[0])

    time_to_prepare = time.time() - s_time
    s_time = time.time()

    # 每个函数都运行1000次
    for _ in range(1000):
        out = f(*dataset[i])

    time_to_run_thousand_times = time.time() - s_time

    print(f"{s}, out shape: {out.shape}, time to prepare: {time_to_prepare:.2f}s, time to run thousand times: {time_to_run_thousand_times:.2f}s")
```

## 我们做的一些实验

下面展示了我们在Atlas A2训练系列产品上运行的一些实验，不同的软硬件条件下，可能会有很大的差异，以下结果仅供参考。

结果说明：

- *准备时间(time to prepare)：潜在的jitted后的对象重用和设备内存拷贝等，可能会导致比较结果不准确。

- *运行一千次的时间(time to run thousand times)：潜在的异步执行操作等，可能会导致测试时间不准确。

### 测试一个简单的函数

定义一个函数 `funtion(a,b,c)=a*b+c`，并使用 `mindspore.jit` 进行转换, 可以通过以下命令运行[simple_funtion.py](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_zh_cn/compile/code/simple_function.py)脚本：

```shell
export GLOG_v=3  # 可选，设置更高的MindSpore日志级别，以减少一些系统打印，让结果看起来更美观
python code/simple_funtion.py
```

结果如下：

| enable jit | jit level | capture mode | backend | fullgraph | *time to prepare | *time to run thousand times |
| --- | --- | --- | --- | --- | --- | --- |
| false | -     | -          | -             | -     | ~4.16s | **~0.09s**    |
|||||||||
| true  | O0    | ast        | ms_backend    | false | ~0.21s | **~0.53s**   |
| true  | O1    | ast        | ms_backend    | false | ~0.03s | ~0.54s   |
| true  | -     | ast        | ge            | false | ~1.01s | ~1.03s   |
|||||||||
| true  | O0    | bytecode   | ms_backend    | false | ~0.13s | **~0.69s**   |
| true  | O1    | bytecode   | ms_backend    | false | ~0.00s | ~0.71s   |
| true  | -     | bytecode   | ge            | false | ~0.00s | ~0.70s   |
|||||||||
| true  | O0    | trace      | ms_backend    | false | ~0.17s | ~3.46s   |
| true  | O1    | trace      | ms_backend    | false | ~0.15s | ~3.45s   |
| true  | -     | trace      | ge            | false | ~0.17s | **~3.42s**   |
|||||||||
| true  | O0    | ast        | ms_backend    | true  | ~0.02s | ~0.54s   |
| true  | O1    | ast        | ms_backend    | true  | ~0.03s | **~0.53s**   |
| true  | -     | ast        | ge            | true  | ~0.14s | ~0.99s   |

### 测试一个简单的卷积模块 (Conv Module)

定义一个在经典网络`resnet`中使用到的核心模块`BasicBlock`, 并使用 `mindspore.jit` 进行转换, 可以通过以下命令运行[simple_conv.py](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_zh_cn/compile/code/simple_conv.py)脚本：

```shell
python code/simple_conv.py
```

结果如下：

**forward**

| enable jit | jit level | capture mode | backend | fullgraph | *time to prepare | *time to run thousand times |
| --- | --- | --- | --- | --- | --- | --- |
| false | -     | -          | -             | -     | ~6.86s | ~1.80s    |
| true  | O0    | ast        | ms_backend    | false | ~0.88s | **~1.00s**    |
| true  | O1    | ast        | ms_backend    | false | ~0.68s | ~1.06s    |

**forward + backward**

| enable jit | jit level | capture mode | backend | fullgraph | *time to prepare | *time to run thousand times |
| --- | --- | --- | --- | --- | --- | --- |
| false | -     | -          | -             | -     | ~1.93s | ~5.69s    |
| true  | O0    | ast        | ms_backend    | false | ~0.84s | ~1.89s    |
| true  | O1    | ast        | ms_backend    | false | ~0.80s | **~1.87s**    |

### 测试一个简单的注意力模块 (Attention Module)

我们定义一个在经典网络`llama3`中使用到的核心模块`LlamaAttention`, 并使用 `mindspore.jit` 进行转换, 可以通过以下命令运行[simple_attention.py](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_zh_cn/compile/code/simple_attention.py)脚本：

```shell
python code/simple_attention.py
```

结果如下：

**forward**

| enable jit | jit level | capture mode | backend | fullgraph | *time to prepare | *time to run thousand times |
| --- | --- | --- | --- | --- | --- | --- |
| false | -     | -          | -             | -     | ~4.73s | ~4.28s    |
| true  | O0    | ast        | ms_backend    | false | ~1.69s | ~4.46s    |
| true  | O1    | ast        | ms_backend    | false | ~1.38s | **~2.15s**    |

**forward + backward**

| enable jit | jit level | capture mode | backend | fullgraph | *time to prepare | *time to run thousand times |
| --- | --- | --- | --- | --- | --- | --- |
| false | -     | -          | -             | -     | ~0.16s | ~12.15s    |
| true  | O0    | ast        | ms_backend    | false | ~1.78s | ~5.30s    |
| true  | O1    | ast        | ms_backend    | false | ~1.69s | **~3.12s**    |
