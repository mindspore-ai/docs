# Just-in-time Compilation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_en/compile/jit_compilation.md)

In this section, we will further explore the working principles of MindSpore and how to run it efficiently. The `mindspore.jit()` transformation performs JIT(just-in-time) compilation on MindSpore Python functions to enable efficient execution in subsequent processes. This compilation occurs during the function’s first execution and may take some time.

## How to use JIT

### Define a function

```python
from mindspore import Tensor

def f(a: Tensor, b: Tensor, c: Tensor):
    return a * b + c
```

### Wrapping functions using `mindspore.jit`

```python
import mindspore

jitted_f = mindspore.jit(f)
```

### Running

```python
import numpy as np
import mindspore
from mindspore import Tensor

f_input = [Tensor(np.random.randn(2, 3), mindspore.float32) for _ in range(3)]

# Run the original function
out = f(*f_input)
print(f"{out=}")

# run the JIT-compiled function
out = jitted_f(*f_input)
print(f"{out=}")
```

> `mindspore.jit` cannot compile temporary source code entered directly in the terminal, it must be executed as a `.py` file.

## Advanced Usages

### Common Configurations

For details about the mindspore.jit interface, refer to the [API documentation](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mindspore/mindspore.jit.html). Common configurations include:

- capture_mode: Specifies the method used to create the computational `graph` (e.g., `ast` for building by parsing Python code, `bytecode` for building from Python bytecode, and `trace` for constructing by tracing Python code execution).
- jit_level: Controls the level of compilation optimization (e.g., default is `O0`; for additional optimization, choose `O1`).
- fullgraph: Determines whether to compile the entire function into a computational `graph`. Defaults to False, allowing jit to maximize compatibility with Python syntax. Setting this to True usually yields better performance but requires stricter syntax adherence.
- backend: Specifies the backend used for compilation.

### How to Use

The following provides the usage for `ast`, `bytecode`, and `trace` modes respectively.

```python
import mindspore

# constructing graph with ast mode
jitted_by_ast_and_levelO0_f = mindspore.jit(f, capture_mode="ast", jit_level="O0")
jitted_by_ast_and_levelO1_f = mindspore.jit(f, capture_mode="ast", jit_level="O1")
jitted_by_ast_and_ge_f = mindspore.jit(f, capture_mode="ast", backend="GE")

# constructing graph with bytecode mode
jitted_by_bytecode_and_levelO0_f = mindspore.jit(f, capture_mode="bytecode", jit_level="O0")
jitted_by_bytecode_and_levelO1_f = mindspore.jit(f, capture_mode="bytecode", jit_level="O1")
jitted_by_bytecode_and_ge_f = mindspore.jit(f, capture_mode="bytecode", backend="GE")


# constructing graph with trace mode
# direct conversion via mindspore.jit(f, capture_mode="trace", ...) is not supported. instead, functions must be wrapped using the decorator @mindspore.jit(capture_mode="trace", ...)
@mindspore.jit(capture_mode="trace", jit_level="O0")
def jitted_by_trace_and_levelO0_f(a, b, c):
    return a * b + c

@mindspore.jit(capture_mode="trace", jit_level="O1")
def jitted_by_trace_and_levelO1_f(a, b, c):
    return a * b + c

@mindspore.jit(capture_mode="trace", backend="GE")
def jitted_by_trace_and_ge_f(a, b, c):
    return a * b + c

# use fullgraph (example as ast mode)
jitted_by_ast_and_levelO0_fullgraph_f = mindspore.jit(f, capture_mode="ast", jit_level="O0", fullgraph=True)
jitted_by_ast_and_levelO1_fullgraph_f = mindspore.jit(f, capture_mode="ast", jit_level="O1", fullgraph=True)
jitted_by_ast_and_ge_fullgraph_f = mindspore.jit(f, capture_mode="ast", backend="GE", fullgraph=True)

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

> When using trace mode to build the graph, direct conversion using `mindspore.jit(f, capture_mode="trace", ...)` is not supported. Instead, functions must be wrapped using the decorator `@mindspore.jit(capture_mode="trace", ...)`.

### Running

```python
# make data
dataset = [[Tensor(np.random.randn(2, 3), mindspore.float32) for _ in range(3)] for i in range(1000)]

for s, f in function_dict.items():
    s_time = time.time()

    out = f(*dataset[0])

    time_to_prepare = time.time() - s_time
    s_time = time.time()

    # run each function 1000 times
    for _ in range(1000):
        out = f(*dataset[i])

    time_to_run_thousand_times = time.time() - s_time

    print(f"{s}, out shape: {out.shape}, time to prepare: {time_to_prepare:.2f}s, time to run a thousand times: {time_to_run_thousand_times:.2f}s")
```

## Experiments and Results

Below, we present several experiments conducted on the `Atlas A2` training product series. Note that results may vary significantly under different hardware and software conditions, and thus, the following results are for reference only.

Explanation of Results:

- time to prepare: potential jitted object reuse and device memory copy may lead to inaccurate comparison.

- time to run a thousand times: potential asynchronous execution operations may lead to inaccurate testing times.

### Test a simple function

Define a function `f(a, b, c)=a*b+c` and convert it using `mindspore.jit`. You can run the script [simple_function.py](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_zh_cn/compile/code/simple_function.py) using the following command:

```shell
export GLOG_v=3  # Optionally, set a higher MindSpore log level to reduce some system print outputs, making the results more intuitive.
python code/simple_funtion.py
```

Results:

| enable jit | jit level | capture mode | backend | fullgraph | *time to prepare | *time to run a thousand times |
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

### Test a simple conv module

Define the `BasicBlock` module, used in the `ResNet`, and convert it using `mindspore.jit`. You can run the script [simple_conv.py](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_zh_cn/compile/code/simple_conv.py) using the following command:

```shell
python code/simple_conv.py
```

Results:

**forward**

| enable jit | jit level | capture mode | backend | fullgraph | *time to prepare | *time to run a thousand times |
| --- | --- | --- | --- | --- | --- | --- |
| false | -     | -          | -             | -     | ~6.86s | ~1.80s    |
| true  | O0    | ast        | ms_backend    | false | ~0.88s | **~1.00s**    |
| true  | O1    | ast        | ms_backend    | false | ~0.68s | ~1.06s    |

**forward + backward**

| enable jit | jit level | capture mode | backend | fullgraph | *time to prepare | *time to run a thousand times |
| --- | --- | --- | --- | --- | --- | --- |
| false | -     | -          | -             | -     | ~1.93s | ~5.69s    |
| true  | O0    | ast        | ms_backend    | false | ~0.84s | ~1.89s    |
| true  | O1    | ast        | ms_backend    | false | ~0.80s | **~1.87s**    |

### Test a simple attention module

Define the `LlamaAttention` module, used in the `Llama3`, and convert it using `mindspore.jit`. You can run the script [simple_attention.py](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_zh_cn/compile/code/simple_attention.py) using the following command:

```shell
python code/simple_attention.py
```

Results:

**forward**

| enable jit | jit level | capture mode | backend | fullgraph | *time to prepare | *time to run a thousand times |
| --- | --- | --- | --- | --- | --- | --- |
| false | -     | -          | -             | -     | ~4.73s | ~4.28s    |
| true  | O0    | ast        | ms_backend    | false | ~1.69s | ~4.46s    |
| true  | O1    | ast        | ms_backend    | false | ~1.38s | **~2.15s**    |

**forward + backward**

| enable jit | jit level | capture mode | backend | fullgraph | *time to prepare | *time to run a thousand times |
| --- | --- | --- | --- | --- | --- | --- |
| false | -     | -          | -             | -     | ~0.16s | ~12.15s    |
| true  | O0    | ast        | ms_backend    | false | ~1.78s | ~5.30s    |
| true  | O1    | ast        | ms_backend    | false | ~1.69s | **~3.12s**    |
