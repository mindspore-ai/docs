# Class Graph

\#include &lt;[graph.h](https://gitee.com/mindspore/mindspore-lite/blob/master/include/api/graph.h)&gt;

## 构造函数和析构函数

```cpp
  Graph()
  explicit Graph(const std::shared_ptr<GraphData> &graph_data)
  explicit Graph(std::shared_ptr<GraphData> &&graph_data)
  explicit Graph(std::nullptr_t)
  ~Graph()
```

- 参数

    - `graph_data`: 输出通道数。

## 公有成员函数

### ModelType

```cpp
  enum ModelType ModelType() const
```

获取模型类型。

- 返回值

  模型类型。

### operator==(std::nullptr_t)

```cpp
  bool operator==(std::nullptr_t) const
```

判断是否为空指针。

- 返回值

  是否为空指针。

### operator!=(std::nullptr_t)

```cpp
  bool operator!=(std::nullptr_t) const
```

判断是否为非空指针。

- 返回值

  是否为非空指针。
