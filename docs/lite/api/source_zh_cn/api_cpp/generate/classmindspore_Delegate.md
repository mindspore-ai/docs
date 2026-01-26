# Class Delegate

\#include &lt;[delegate.h](https://atomgit.com/mindspore/mindspore-lite/blob/master/include/api/delegate.h)&gt;

`Delegate`定义了第三方AI框架接入MindSpore Lite的代理接口。

## 构造函数和析构函数

```cpp
Delegate() = default
virtual ~Delegate() = default
```

## 公有成员函数

### Init

```cpp
virtual Status Init() = 0
```

初始化Delegate资源。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### Build

```cpp
virtual Status Build(DelegateModel *model) = 0
```

Delegate在线构图。

- 参数

    - `model`: 指向存储[DelegateModel](#delegatemodel)实例的指针。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### CreateKernel

```cpp
std::shared_ptr<kernel::Kernel> CreateKernel(const std::shared_ptr<kernel::Kernel> &node) override
```

创建Kernel。

- 参数

    - `node`: 指向Kernel[Kernel]实例的共享指针。

- 返回值

  Kernel类共享指针。

### IsDelegateNode

```cpp
bool IsDelegateNode(const std::shared_ptr<kernel::Kernel> &node) override
```

是否是Delegate节点。

- 参数

    - `node`: 指向Kernel[Kernel]实例的共享指针。

- 返回值

  bool值。

### ReplaceNode

```cpp
void ReplaceNodes(const std::shared_ptr<LiteDelegateGraph> &graph) override
```

替换节点。

- 参数

    - `graph`: 要被替换的节点。
