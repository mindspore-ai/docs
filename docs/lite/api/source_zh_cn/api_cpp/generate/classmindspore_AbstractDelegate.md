# Class AbstractDelegate

\#include &lt;[delegate_api.h](https://atomgit.com/mindspore/mindspore-lite/blob/r2.9/include/api/delegate_api.h)&gt;

`AbstractDelegate`定义了MindSpore Lite 创建Delegate（抽象类）。

## 构造函数

```cpp
AbstractDelegate()
AbstractDelegate(const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs)
      : inputs_(inputs), outputs_(outputs)
```

## 析构函数

```cpp
virtual ~AbstractDelegate() = default
```

## 公有成员函数

### inputs

```cpp
const std::vector<mindspore::MSTensor> &inputs()
```

返回AbstractDelegate的inputTensor。

### outputs

```cpp
const std::vector<mindspore::MSTensor> &outputs()
```

返回AbstractDelegate的outputTensor。

## 保护成员变量

### inputs_

```cpp
std::vector<mindspore::MSTensor> inputs_
```

### outputs_

```cpp
std::vector<mindspore::MSTensor> outputs_
```
