# mindspore

<a href="https://gitee.com/mindspore/docs/blob/master/docs/api_cpp/source_zh_cn/mindspore.md" target="_blank"><img src="./_static/logo_source.png"></a>

\#include &lt;[ms_tensor.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/ms_tensor.h)&gt;

## KernelCallBack

```cpp
using KernelCallBack = std::function<bool(std::vector<tensor::MSTensor *> inputs, std::vector<tensor::MSTensor *> outputs, const CallBackParam &opInfo)>
```

一个函数包装器。KernelCallBack 定义了指向回调函数的指针。

## CallBackParam

一个结构体。CallBackParam定义了回调函数的输入参数。

### 公有属性

```cpp
node_name
```

**string** 类型变量。节点名参数。

```cpp
node_type
```

**string** 类型变量。节点类型参数。
