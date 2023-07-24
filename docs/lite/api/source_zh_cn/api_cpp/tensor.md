# mindspore::tensor

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/lite/api/source_zh_cn/api_cpp/tensor.md)

## MSTensor

\#include &lt;[ms_tensor.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/ms_tensor.h)&gt;

MSTensor定义了MindSpore Lite中的张量。

### 构造函数和析构函数

#### MSTensor

```cpp
MSTensor()
```

MindSpore Lite MSTensor的构造函数。

- 返回值

    MindSpore Lite MSTensor的实例。

#### ~MSTensor

```cpp
virtual ~MSTensor()
```

MindSpore Lite Model的析构函数。

### 公有成员函数

#### data_type

```cpp
virtual TypeId data_type() const
```

获取MindSpore Lite MSTensor的数据类型。

> TypeId在[mindspore/mindspore/core/ir/dtype/type_id\.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/core/ir/dtype/type_id.h)中定义。只有TypeId枚举中的数字类型或kObjectTypeString可用于MSTensor。

- 返回值

    MindSpore Lite MSTensor类的MindSpore Lite TypeId。

#### shape

```cpp
virtual std::vector<int> shape() const
```

获取MindSpore Lite MSTensor的形状。

- 返回值

    一个包含MindSpore Lite MSTensor形状数值的整型向量。

#### ElementsNum

```cpp
virtual int ElementsNum() const
```

获取MSTensor中的元素个数。

- 返回值

    MSTensor中的元素个数

#### Size

```cpp
virtual size_t Size() const
```

获取MSTensor中的数据的字节数大小。

- 返回值

    MSTensor中的数据的字节数大小。

#### MutableData

```cpp
virtual void *MutableData() const
```

获取MSTensor中的数据的指针。

> 该数据指针可用于对MSTensor中的数据进行读取和写入。
>
> 如果是opencl的MSTensor，使用MutableData()接口，数据将会从Device映射到Host。

- 返回值

    指向MSTensor中的数据的指针。
