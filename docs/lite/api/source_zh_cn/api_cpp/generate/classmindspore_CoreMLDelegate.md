# Class CoreMLDelegate

\#include &lt;[delegate.h](https://atomgit.com/mindspore/mindspore-lite/blob/master/include/api/delegate.h)&gt;

`CoreMLDelegate`继承自`Delegate`类，定义了CoreML框架接入MindSpore Lite的代理接口。

## 构造函数

```cpp
CoreMLDelegate()
```

## 公有成员函数

### Init

```cpp
Status Init() override
```

初始化CoreMLDelegate资源，仅在内部图编译阶段调用。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### Build

```cpp
Status Build(DelegateModel *model) override
```

CoreMLDelegate在线构图，仅在内部图编译阶段调用。

- 参数

    - `model`: 指向存储[DelegateModel](./classmindspore_DelegateModel.md)实例的指针。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。
