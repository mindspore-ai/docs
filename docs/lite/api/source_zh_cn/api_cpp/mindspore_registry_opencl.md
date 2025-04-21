# mindspore::registry::opencl

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_zh_cn/api_cpp/mindspore_registry_opencl.md)

## 接口汇总

| 类名 | 描述 |
| --- | --- |
| [OpenCLRuntimeWrapper](#openclruntimewrapper) | 端侧GPU操作的接口类|

## OpenCLRuntimeWrapper

\#include <[include/registry/opencl_runtime_wrapper.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/opencl_runtime_wrapper.h)>

OpenCLRuntimeWrapper类包装了内部OpenCL的相关接口，用于支持南向GPU算子的开发。

### OpenCLRuntimeWrapper

```c++
OpenCLRuntimeWrapper() = default;
```

构造函数。

### ~OpenCLRuntimeWrapper

```c++
~OpenCLRuntimeWrapper = default;
```

析构函数。

## 公有成员函数

### LoadSource

```c++
Status LoadSource(const std::string &program_name, const std::string &source);
```

加载OpenCL源代码并指定程序名。

- 参数

    - `program_name`: OpenCL源程序名称。

    - `source`: OpenCL源程序。

### BuildKernel

```c++
Status BuildKernel(cl::Kernel *kernel, const std::string &program_name, const std::string &kernel_name,
                     const std::vector<std::string> &build_options_ext = {});
```

构建OpenCL代码。

- 参数

    - `kernel`: 用于返回已编译的内核。

    - `program_name`: OpenCL源程序名称。

    - `kernel_name`: OpenCL内核名称。

    - `build_options_ext`: OpenCL内核构建选项。

### SetKernelArg

```c++
Status SetKernelArg(const cl::Kernel &kernel, uint32_t index, void *const value);
```

设置OpenCL内核运行时指针类参数的值。

- 参数

    - `kernel`: OpenCL内核。

    - `index`: OpenCL内核参数索引。

    - `value`: OpenCL内核参数值指针。

```c++
template <typename T>
  typename std::enable_if<!std::is_pointer<T>::value, Status>::type SetKernelArg(const cl::Kernel &kernel,
                                                                                 uint32_t index, const T value);
```

设置OpenCL内核运行时非指针类参数的值。

- 参数

    - `kernel`: OpenCL内核。

    - `index`: OpenCL内核参数索引。

    - `value`: OpenCL内核参数值。

### RunKernel

```c++
Status RunKernel(const cl::Kernel &kernel, const cl::NDRange &global, const cl::NDRange &local,
                   cl::CommandQueue *command_queue = nullptr, cl::Event *event = nullptr);
```

运行OpenCL内核。

- 参数

    - `kernel`: OpenCL内核。

    - `global`: 工作项的总数量。

    - `local`: 每个工作组中工作项的数量。

    - `command_queue`: 使用的指令队列，默认空，使用框架内默认队列。

    - `event`: 事件对象的指针，用来标识本次执行命令，默认空，无事件标识。

### SyncCommandQueue

```c++
Status SyncCommandQueue();
```

同步指令队列。

### GetAllocator

```c++
std::shared_ptr<Allocator> GetAllocator();
```

获取GPU内存分配器的智能指针。通过[Allocator接口](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html)，可申请GPU内存，用于OpenCL内核的运算。

### MapBuffer

```c++
void *MapBuffer(void *host_ptr, int flags, bool sync = true);
```

重新将GPU内存映射到主机内存地址，以便读写。

- 参数

    - `host_ptr`: 主机内存地址（为GPU内存所映射的CPU地址）。

    - `flags`: 内存映射的OpenCL功能符号，如CL_MAP_READ，CL_MAP_WRITE。

    - `sync`: 是否同步标志。

### UnmapBuffer

```c++
Status UnmapBuffer(void *host_ptr);
```

将改变后的内存数据，写入GPU。

- 参数

    - `host_ptr`: 主机内存地址（为GPU内存所映射的CPU地址）。

### ReadImage

```c++
Status ReadImage(void *buffer, void *dst_data);
```

读取解析Image形式的GPU内存到目标地址，写入的数据格式为NHWC4（C轴4数据对齐的NHWC格式数据）。

- 参数

    - `buffer`: Image格式的GPU内存所映射的主机内存地址。

    - `dst_data`: 目标地址。

### WriteImage

```c++
Status WriteImage(void *buffer, void *src_data);
```

从源地址`src_data`读取数据，写入到Image形式的GPU内存`buffer`。

- 参数

    - `buffer`: Image格式的GPU内存所映射的主机内存地址。

    - `src_data`: 源地址。

### DeviceMaxWorkGroupSize

```c++
uint64_t DeviceMaxWorkGroupSize();
```

获取支持的最大工作组数量。

### GetMaxImage2DWidth

```c++
uint64_t GetMaxImage2DWidth();
```

获取Image内存数据支持的最大宽度。

### GetMaxImage2DHeight

```c++
uint64_t GetMaxImage2DHeight();
```

获取Image内存数据支持的最大高度。

### GetImagePitchAlignment

```c++
uint64_t GetImagePitchAlignment();
```

获取Image内存数据的宽度对齐值。
