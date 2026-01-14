# Class MSTensor

\#include &lt;[types.h](https://gitee.com/mindspore/mindspore-lite/blob/master/include/api/types.h)&gt;

`MSTensor`定义了MindSpore中的张量。

## 构造函数和析构函数

```cpp
MSTensor()
explicit MSTensor(const std::shared_ptr<Impl> &impl)
MSTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, const void *data, size_t data_len)
explicit MSTensor(std::nullptr_t)
~MSTensor()
```

注意：MSTensor构造时，若data指针通过malloc生成，用户在构造完成MSTensor后，需自行释放free，否则存在内存泄露。

## 静态公有成员函数

| 函数                                                                                                                                                                                                                 | 云侧推理是否支持 | 端侧推理是否支持 |
|------------------------------------------------------------------------------------------------------------------|---------|---------|
| [MSTensor *CreateTensor(const std::string &name, DataType type, const std::vector\<int64_t\> &shape, const void *data, size_t data_len, const std::string &device = "", int device_id = -1) noexcept](#createtensor)     |    √    |    √    |
| [MSTensor *CreateTensor(const std::string &name, const MSTensor &tensor, const std::string &device = "", int device_id = -1) noexcept](#createtensor-1)     |    √    |    √    |
| [MSTensor *CreateRefTensor(const std::string &name, DataType type, const std::vector\<int64_t\> &shape, void *data, size_t data_len) noexcept](#createreftensor)     |    √    |    √    |
| [static inline MSTensor CreateDeviceTensor(const std::string &name, DataType type, const std::vector\<int64_t\> &shape, void *data, size_t data_len) noexcept](#createdevicetensor)     |    √    |    ✕    |
| [static inline MSTensor *CreateTensorFromFile(const std::string &file, DataType type = DataType::kNumberTypeUInt8, const std::vector\<int64_t\> &shape = {}) noexcept](#createtensorfromfile)     |    √    |    ✕    |
| [MSTensor *StringsToTensor(const std::string &name, const std::vector\<std::string\> &str)](#stringstotensor)     |    √    |    √    |
| [std::vector\<std::string\> TensorToStrings(const MSTensor &tensor)](#tensortostrings)     |    √    |    √    |
| [void DestroyTensorPtr(MSTensor *tensor) noexcept](#destroytensorptr)     |    √    |    √    |

### CreateTensor

```cpp
MSTensor *CreateTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                       const void *data, size_t data_len, const std::string &device = "",
                       int device_id = -1) noexcept
```

创建一个`MSTensor`对象，其数据需复制后才能由`Model`访问，必须与`DestroyTensorPtr`成对使用。

- 参数

    - `name`: 名称。
    - `type`：数据类型。
    - `shape`：张量的形状。
    - `data`：数据指针，指向一段已开辟的内存。
    - `data_len`：数据长度，以字节为单位。
    - `device`：设备类型，表明Tensor的内存存放的位置位于设备侧。
    - `device_id`：设备编号。

- 返回值

  `MStensor`指针。

### CreateTensor

```cpp
MSTensor *CreateTensor(const std::string &name, const MSTensor &tensor, const std::string &device = "",
                       int device_id = -1) noexcept
```

创建一个`MSTensor`对象，其数据需复制后才能由`Model`访问，必须与`DestroyTensorPtr`成对使用。

- 参数

    - `name`: 名称。
    - `tensor`：用于作为拷贝的源MSTensor。
    - `device`：设备类型，表明Tensor的内存存放的位置位于设备侧。
    - `device_id`：设备编号。

- 返回值

  `MStensor`指针。

### CreateRefTensor

```cpp
MSTensor *CreateRefTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, void *data,
                          size_t data_len) noexcept
```

创建一个`MSTensor`对象，其数据可以直接由`Model`访问，必须与`DestroyTensorPtr`成对使用。

- 参数

    - `name`: 名称。
    - `type`：数据类型。
    - `shape`：张量的形状。
    - `data`：数据指针，指向一段已开辟的内存。
    - `data_len`：数据长度，以字节为单位。

- 返回值

  `MStensor`指针。

### CreateDeviceTensor

```cpp
static inline MSTensor CreateDeviceTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                                          void *data, size_t data_len) noexcept
```

创建一个`MSTensor`对象，其device数据可以直接由`Model`访问，不需要与`DestroyTensorPtr`成对使用。

- 参数

    - `name`: 名称。
    - `type`：数据类型。
    - `shape`：张量的形状。
    - `data`：数据指针，指向一段已开辟的device内存。
    - `data_len`：数据长度，以字节为单位。

- 返回值

  `MStensor`对象。

### CreateTensorFromFile

```cpp
static inline MSTensor *CreateTensorFromFile(const std::string &file, DataType type = DataType::kNumberTypeUInt8,
                                             const std::vector<int64_t> &shape = {}) noexcept
```

创建一个`MSTensor`对象，其数据由文件路径`file`所指定，必须与`DestroyTensorPtr`成对使用。

- 参数

    - `file`: 文件路径，指向存放数据的二进制格式文件，可以是相对路径或者绝对路径。
    - `type`：`file`文件保存的数据类型，也是创建后`MSTensor`对象的数据类型。
    - `shape`：张量的形状，`shape`的乘积代表了`file`文件内数据的个数。

- 返回值

  `MStensor`指针。

### StringsToTensor

```cpp
MSTensor *StringsToTensor(const std::string &name, const std::vector<std::string> &str)
```

创建一个字符串类型的`MSTensor`对象，其数据需复制后才能由`Model`访问，必须与`DestroyTensorPtr`成对使用。

- 参数

    - `name`: 名称。
    - `str`：装有若干个字符串的`vector`容器。

- 返回值

  `MStensor`指针。

### TensorToStrings

```cpp
std::vector<std::string> TensorToStrings(const MSTensor &tensor)
```

将字符串类型的`MSTensor`对象解析为字符串。

- 参数

    - `tensor`: 张量对象。

- 返回值

  装有若干个字符串的`vector`容器。

### DestroyTensorPtr

```cpp
void DestroyTensorPtr(MSTensor *tensor) noexcept
```

销毁一个由`Clone`、`StringsToTensor`、`CreateRefTensor`或`CreateTensor`所创建的对象，请勿用于销毁其他来源的`MSTensor`。

- 参数

    - `tensor`: 由`Clone`、`StringsToTensor`、`CreateRefTensor`或`CreateTensor`返回的指针。

## 公有成员函数

| 函数                            | 云侧推理是否支持 | 端侧推理是否支持 |
|---------------------------------------------------------------------------------------------|---------|---------|
| [std::string Name() const](#name)     |    √    |    √    |
| [enum DataType DataType() const](#datatype)     |    √    |    √    |
| [const std::vector\<int64_t\> &Shape() const](#shape)     |    √    |    √    |
| [int64_t ElementNum() const](#elementnum)     |    √    |    √    |
| [std::shared_ptr\<const void\> Data() const](#data)     |    √    |    √    |
| [void *MutableData()](#mutabledata)     |    √    |    √    |
| [size_t DataSize() const](#datasize)     |    √    |    √    |
| [int GetDevice() const](#getdevice)     |    √    |    ✕    |
| [int GetDeviceId() const](#getdeviceid)     |    √    |    ✕    |
| [bool IsConst() const](#isconst)     |    √    |    √    |
| [bool IsDevice() const](#isdevice)     |    √    |    ✕    |
| [MSTensor *Clone() const](#clone)     |    √    |    √    |
| [bool operator==(std::nullptr_t) const](https://www.mindspore.cn/lite/api/zh-CN/master/generate/classmindspore_MSTensor.html#operatorstd-nullptr-t)     |    √    |    √    |
| [bool operator!=(std::nullptr_t) const](https://www.mindspore.cn/lite/api/zh-CN/master/generate/classmindspore_MSTensor.html#operatorstd-nullptr-t-1)     |    √    |    √    |
| [bool operator!=(const MSTensor &tensor) const](#operatorconst-mstensor-tensor)     |    √    |    √    |
| [bool operator==(const MSTensor &tensor) const](#operatorconst-mstensor-tensor-1)     |    √    |    √    |
| [void SetShape(const std::vector\<int64_t\> &shape)](#setshape)     |    √    |    √    |
| [void SetDataType(enum DataType data_type)](#setdatatype)     |    √    |    √    |
| [void SetTensorName(const std::string &name)](#settensorname)     |    √    |    √    |
| [void SetAllocator(std::shared_ptr\<Allocator\> allocator)](#setallocator)     |    √    |    √    |
| [std::shared_ptr\<Allocator\> allocator() const](#allocator)     |    √    |    √    |
| [void SetFormat(mindspore::Format format)](#setformat)     |    √    |    √    |
| [mindspore::Format format() const](#format)     |    √    |    √    |
| [void SetData(void *data, bool own_data = true)](#setdata)     |    √    |    √    |
| [void SetDeviceData(void *data)](#setdevicedata)     |    √    |    √    |
| [void *GetDeviceData()](#getdevicedata)     |    √    |    √    |
| [std::vector\<QuantParam\> QuantParams() const](#quantparams)     |    √    |    √    |
| [void SetQuantParams(std::vector\<QuantParam\> quant_params)](#setquantparams)     |    √    |    √    |
| [const std::shared_ptr\<Impl\> impl()](#impl)     |    √    |    √    |

### Name

```cpp
std::string Name() const
```

获取`MSTensor`的名字。

- 返回值

  `MSTensor`的名字。

### DataType

```cpp
enum DataType DataType() const
```

获取`MSTensor`的数据类型。

- 返回值

  `MSTensor`的数据类型。

### Shape

```cpp
const std::vector<int64_t> &Shape() const
```

获取`MSTensor`的Shape。

- 返回值

  `MSTensor`的Shape。

### ElementNum

```cpp
int64_t ElementNum() const
```

获取`MSTensor`的元素个数。

- 返回值

  `MSTensor`的元素个数。

### Data

```cpp
std::shared_ptr<const void> Data() const
```

获取指向`MSTensor`中的数据拷贝的智能指针。

- 返回值

  指向`MSTensor`中的数据拷贝的智能指针。

### MutableData

```cpp
void *MutableData()
```

获取`MSTensor`中的数据的指针。如果为空指针，为`MSTensor`的数据申请内存，并返回申请内存的地址，如果不为空，返回数据的指针。

- 返回值

  指向`MSTensor`中的数据的指针。

### DataSize

```cpp
size_t DataSize() const
```

获取`MSTensor`中的数据的以字节为单位的内存长度。

- 返回值

  `MSTensor`中的数据的以字节为单位的内存长度。

### GetDevice

```cpp
int GetDevice() const
```

获取`MSTensor`所处的设备类型。

- 返回值

  `MSTensor`所处的设备类型。

### GetDeviceId

```cpp
int GetDeviceId() const
```

获取`MSTensor`所处的设备编号。

- 返回值

  `MSTensor`所处的设备编号。

### IsConst

```cpp
bool IsConst() const
```

判断`MSTensor`中的数据是否是常量数据。

- 返回值

  `MSTensor`中的数据是否是常量数据。

### IsDevice

```cpp
bool IsDevice() const
```

判断`MSTensor`中是否在设备上。

- 返回值

  `MSTensor`中是否在设备上。

### Clone

```cpp
MSTensor *Clone() const
```

拷贝一份自身的副本。

- 返回值

  指向深拷贝副本的指针，必须与`DestroyTensorPtr`成对使用。

### operator==(std::nullptr_t)

```cpp
bool operator==(std::nullptr_t) const
```

判断`MSTensor`是否合法。

- 返回值

  `MSTensor`是否合法。

### operator!=(std::nullptr_t)

```cpp
bool operator!=(std::nullptr_t) const
```

判断`MSTensor`是否合法。

- 返回值

  `MSTensor`是否合法。

### operator==(const MSTensor &tensor)

```cpp
bool operator==(const MSTensor &tensor) const
```

判断`MSTensor`是否与另一个MSTensor相等。

- 返回值

  `MSTensor`是否与另一个MSTensor相等。

### operator!=(const MSTensor &tensor)

```cpp
bool operator!=(const MSTensor &tensor) const
```

判断`MSTensor`是否与另一个MSTensor不相等。

- 返回值

  `MSTensor`是否与另一个MSTensor不相等。

### SetShape

```cpp
void SetShape(const std::vector<int64_t> &shape)
```

设置`MSTensor`的Shape，目前在[Delegate](#delegate)机制使用。

### SetDataType

```cpp
void SetDataType(enum DataType data_type)
```

设置`MSTensor`的DataType，目前在[Delegate](#delegate)机制使用。

### SetTensorName

```cpp
void SetTensorName(const std::string &name)
```

设置`MSTensor`的名字，目前在[Delegate](#delegate)机制使用。

### SetAllocator

```cpp
void SetAllocator(std::shared_ptr<Allocator> allocator)
```

设置`MSTensor`数据所属的内存池。

- 参数

    - `model`: 指向Allocator的指针。

### allocator

```cpp
std::shared_ptr<Allocator> allocator() const
```

获取`MSTensor`数据所属的内存池。

- 返回值

  指向Allocator的指针。

### SetFormat

```cpp
void SetFormat(mindspore::Format format)
```

设置`MSTensor`数据的format，目前在[Delegate](#delegate)机制使用。

### format

```cpp
mindspore::Format format() const
```

获取`MSTensor`数据的format，目前在[Delegate](#delegate)机制使用。

### SetData

```cpp
void SetData(void *data, bool own_data = true)
```

设置指向`MSTensor`数据的指针。

- 参数

    - `data`: 新的数据的地址。
    - `own_data`: 是否在`MSTensor`析构时释放数据内存。如果为true，将在`MSTensor`析构时释放数据内存，如果重复调用`SetData`，将仅释放新的数据内存，老的数据内存需要用户释放；如果为false，需要用户释放数据内存。由于向前兼容，默认为true，建议用户设置为false。

### SetDeviceData

```cpp
void SetDeviceData(void *data)
```

设置数据的设备地址，由用户负责设备内存的申请和释放。仅适用于Ascend和GPU硬件后端。

### GetDeviceData

```cpp
void *GetDeviceData()
```

获取由`SetDeviceData`接口设置的`MSTensor`数据的设备地址。

### QuantParams

```cpp
std::vector<QuantParam> QuantParams() const
```

获取`MSTensor`的量化参数，目前在[Delegate](#delegate)机制使用。

### SetQuantParams

```cpp
void SetQuantParams(std::vector<QuantParam> quant_params)
```

设置`MSTensor`的量化参数，目前在[Delegate](#delegate)机制使用。

### impl

```cpp
const std::shared_ptr<Impl> impl()
```

获取实现类的指针。
