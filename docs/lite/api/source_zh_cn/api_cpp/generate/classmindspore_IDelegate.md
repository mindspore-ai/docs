# Template Class IDelegate

```cpp
std::vector<mindspore::MSTensor> outputs_
```

\#include &lt;[delegate_api.h](https://atomgit.com/mindspore/mindspore-lite/blob/master/include/api/delegate_api.h)&gt;

`IDelegate`定义了MindSpore Lite 创建Delegate（模板类）。

## 构造函数

```cpp
IDelegate()
```

```cpp
IDelegate(const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs)
      : AbstractDelegate(inputs, outputs)
```

## 析构函数

```cpp
virtual ~IDelegate() = default
```

## 公有成员函数

### ReplaceNodes

```cpp
virtual void ReplaceNodes(const std::shared_ptr<Graph> &graph) = 0
```

替换Delegate节点。

### IsDelegateNode

```cpp
virtual bool IsDelegateNode(const std::shared_ptr<Node> &node) = 0
```

判断节点是否属于Delegate。

### CreateKernel

```cpp
virtual std::shared_ptr<Kernel> CreateKernel(const std::shared_ptr<Node> &node) = 0
```

创建Kernel。
