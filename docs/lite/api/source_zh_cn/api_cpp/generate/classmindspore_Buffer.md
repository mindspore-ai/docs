# Class Buffer

\#include &lt;[types.h](https://atomgit.com/mindspore/mindspore-lite/blob/r2.9/include/api/types.h)&gt;

Buffer定义了MindSpore中Buffer数据的结构。

## 构造函数

```cpp
  Buffer()
```

```cpp
  Buffer(const void *data, size_t data_len)
```

## 析构函数

```cpp
  ~Buffer()
```

## 公有成员函数

| 函数                                | 云侧推理是否支持 | 端侧推理是否支持 |
|-----------------------------|--------|--------|
| [const void *Data() const](#data)     |    √    |    √    |
| [void *MutableData()](#mutabledata)     |    √    |    √    |
| [size_t DataSize() const](#datasize)     |    √    |    √    |
| [bool ResizeData(size_t data_len)](#resizedata)     |    √    |    √    |
| [bool SetData(const void *data, size_t data_len)](#setdata)     |    √    |    √    |
| [Buffer Clone() const](#clone)     |    √    |    √    |

### Data

```cpp
const void *Data() const
```

获取只读的数据地址。

- 返回值

  const void指针。

### MutableData

```cpp
void *MutableData()
```

获取可写的数据地址。

- 返回值

  void指针。

### DataSize

```cpp
size_t DataSize() const
```

获取data大小。

- 返回值

  当前data大小。

### ResizeData

```cpp
bool ResizeData(size_t data_len)
```

重置data大小。

- 参数

    - `data_len`: data大小

- 返回值

  是否配置成功。

### SetData

```cpp
bool SetData(const void *data, size_t data_len)
```

配置Data和大小。

- 参数

    - `data`: data地址
    - `data_len`: data大小

- 返回值

  是否配置成功。

### Clone

```cpp
Buffer Clone() const
```

拷贝一份自身的副本。

- 返回值

  指向副本的指针。
