# Typedef mindspore::MSKernelCallBack

\#include &lt;[types.h](https://atomgit.com/mindspore/mindspore-lite/blob/r2.9.0/include/api/types.h)&gt;

```cpp
using MSKernelCallBack = std::function<bool(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs, const MSCallBackParam &opInfo)>
```

一个函数包装器。MSKernelCallBack 定义了指向回调函数的指针。
