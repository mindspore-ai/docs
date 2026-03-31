# Class Allocator

\#include &lt;[allocator.h](https://atomgit.com/mindspore/mindspore-lite/blob/r2.9.0/include/api/allocator.h)&gt;

内存管理基类。

## 析构函数

```cpp
virtual ~Allocator()
```

析构函数。

## 公有成员函数

### Malloc

```cpp
virtual void *Malloc(size_t size)
```

内存分配。

- 参数

    - `size`: 要分配的内存大小，单位为Byte。

```cpp
virtual void *Malloc(size_t weight, size_t height, DataType type)
```

Image格式内存分配。

- 参数

    - `weight`: 要分配的Image格式内存的宽度。
    - `height`: 要分配的Image格式内存的高度。
    - `type`: 要分配的Image格式内存的数据类型。

### Free

```cpp
virtual void *Free(void *ptr)
```

内存释放。

- 参数

    - `ptr`: 要释放的内存地址，该值由[Malloc](#malloc)分配。

### RefCount

```cpp
virtual int RefCount(void *ptr)
```

返回分配内存的引用计数。

- 参数

    - `ptr`: 要操作的内存地址，该值由[Malloc](#malloc)分配。

### SetRefCount

```cpp
virtual int SetRefCount(void *ptr, int ref_count)
```

设置分配内存的引用计数。

- 参数

    - `ptr`: 要操作的内存地址，该值由[Malloc](#malloc)分配。
    - `ref_count`: 引用计数值。

### DecRefCount

```cpp
virtual int DecRefCount(void *ptr, int ref_count)
```

分配的内存引用计数减一。

- 参数

    - `ptr`: 要操作的内存地址，该值由[Malloc](#malloc)分配。
    - `ref_count`: 引用计数值。

### IncRefCount

```cpp
virtual int IncRefCount(void *ptr, int ref_count)
```

分配的内存引用计数加一。

- 参数

    - `ptr`: 要操作的内存地址，该值由[Malloc](#malloc)分配。
    - `ref_count`: 引用计数值。

### Create

```cpp
static std::shared_ptr<Allocator> Create()
```

创建默认的内存分配器。

### Prepare

```cpp
virtual void *Prepare(void *ptr)
```

对分配的内存进行预处理。

- 参数

    - `ptr`: 要操作的内存地址，该值由[Malloc](#malloc)分配。

## 保护的数据成员

### aligned_size_

内存对齐的字节数。默认值: `32` 。
