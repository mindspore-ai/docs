# mindspore

<a href="https://gitee.com/mindspore/docs/blob/r1.1/docs/api_cpp/source_en/mindspore.md" target="_blank"><img src="./_static/logo_source.png"></a>

\#include &lt;[ms_tensor.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/ms_tensor.h)&gt;

## KernelCallBack

```cpp
using KernelCallBack = std::function<bool(std::vector<tensor::MSTensor *> inputs, std::vector<tensor::MSTensor *> outputs, const CallBackParam &opInfo)>
```

A function wrapper. KernelCallBack defines the pointer for callback function.

## CallBackParam

A **struct**. CallBackParam defines input arguments for callback function.

### Public Attributes

#### node_name

```cpp
node_name
```

A **string** variable. Node name argument.

#### node_type

```cpp
node_type
```

A **string** variable. Node type argument.
