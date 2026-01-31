# Class DeviceInfoContext

\#include &lt;[context.h](https://atomgit.com/mindspore/mindspore-lite/blob/master/include/api/context.h)&gt;

DeviceInfoContext类定义不同硬件设备的环境信息。

## 构造函数

```cpp
DeviceInfoContext()
```

## 析构函数

```cpp
virtual ~DeviceInfoContext() = default
```

## 公有成员变量

```cpp
struct Data
```

数据。

## 公有成员函数

| 函数                                                                | 云侧推理是否支持 | 端侧推理是否支持 |
|-------------------------------------------------------------------|---------|---------|
| [enum DeviceType GetDeviceType() const](#getdevicetype)     |    √    |    √    |
| [std::shared_ptr\<T\> Cast()](#cast)     |    √    |    √    |
| [std::string GetProvider() const](#getprovider)     |    √    |    √    |
| [void SetProvider(const std::string &provider)](#setprovider)     |    √    |    √    |
| [std::string GetProviderDevice() const](#getproviderdevice)     |    √    |    √    |
| [void SetProviderDevice(const std::string &device)](#setproviderdevice)     |    √    |    √    |
| [void SetAllocator(const std::shared_ptr\<Allocator\> &allocator)](#setallocator)     |    ✕    |    ✕    |
| [std::shared_ptr\<Allocator\> GetAllocator() const](#getallocator)     |    ✕    |    ✕    |

### GetDeviceType

```cpp
virtual enum DeviceType GetDeviceType() const = 0
```

获取该DeviceInfoContext的类型。

- 返回值

  该DeviceInfoContext的类型。

  ```cpp
  enum DeviceType {
    kCPU = 0,
    kGPU,
    kKirinNPU,
    kAscend910,
    kAscend310,
    // add new type here
    kInvalidDeviceType = 100,
  };
  ```

### Cast

```cpp
template <class T>
std::shared_ptr<T> Cast()
```

在打开`-fno-rtti`编译选项的情况下提供类似RTTI的功能，将DeviceInfoContext转换为`T`类型的指针，若转换失败返回`nullptr`。

- 返回值

  转换后`T`类型的指针，若转换失败则为`nullptr`。

### GetProvider

```cpp
inline std::string GetProvider() const
```

获取设备的生产商名。

### SetProvider

```cpp
inline void SetProvider(const std::string &provider)
```

设置设备生产商名。

- 参数

    - `provider`: 生产商名。

### GetProviderDevice

```cpp
inline std::string GetProviderDevice() const
```

获取生产商设备名。

### SetProviderDevice

```cpp
inline void SetProviderDevice(const std::string &device)
```

设置设备生产商设备名。

- 参数

    - `device`: 设备名。

### SetAllocator

```cpp
void SetAllocator(const std::shared_ptr<Allocator> &allocator)
```

设置内存管理器。

- 参数

    - `allocator`: 内存管理器。

### GetAllocator

```cpp
std::shared_ptr<Allocator> GetAllocator() const
```

获取内存管理器。
