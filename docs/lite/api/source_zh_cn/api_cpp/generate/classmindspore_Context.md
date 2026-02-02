# Class Context

\#include &lt;[context.h](https://atomgit.com/mindspore/mindspore-lite/blob/master/include/api/context.h)&gt;

Context类用于保存执行中的环境变量。

## 构造函数

```cpp
Context()
```

## 析构函数

```cpp
~Context() = default
```

## 公有成员变量

### Data

```cpp
struct Data
```

Context的数据。

## 公有成员函数

| 函数                                                                            | 云侧推理是否支持 | 端侧推理是否支持 |
|-------------------------------------------------------------------------------|--------|--------|
| [void SetThreadNum(int32_t thread_num)](#setthreadnum)     |    √    |    √    |
| [int32_t GetThreadNum() const](#getthreadnum)     |    √    |    √    |
| [void SetGroupInfoFile(std::string group_info_file)](#setgroupinfofile)     |    √    |    √    |
| [void SetInterOpParallelNum(int32_t parallel_num)](#setinteropparallelnum)     |    √    |    √    |
| [int32_t GetInterOpParallelNum() const](#getinteropparallelnum)     |    √    |    √    |
| [void SetThreadAffinity(int mode)](#setthreadaffinity)     |    √    |    √    |
| [int GetThreadAffinityMode() const](#getthreadaffinitymode)     |    √    |    √    |
| [void SetThreadAffinity(const std::vector\<int\> &core_list)](#setthreadaffinity-1)     |    √    |    √    |
| [std::vector\<int32_t\> GetThreadAffinityCoreList() const](#getthreadaffinitycorelist)     |    √    |    √    |
| [void SetEnableParallel(bool is_parallel)](#setenableparallel)     |    ✕    |    √    |
| [bool GetEnableParallel() const](#getenableparallel)     |    ✕    |    √    |
| [void SetBuiltInDelegate(DelegateMode mode)](#setbuiltindelegate)     |    ✕    |    √    |
| [DelegateMode GetBuiltInDelegate() const](#getbuiltindelegate)     |    ✕    |    √    |
| [void set_delegate(const std::shared_ptr\<AbstractDelegate\> &delegate)](https://www.mindspore.cn/lite/api/zh-CN/master/generate/classmindspore_Context.html#set-delegate)         | ✕      | √      |
| [void SetDelegate(const std::shared_ptr\<Delegate\> &delegate)](#setdelegate)     |    ✕    |    √    |
| [std::shared_ptr\<AbstractDelegate\> get_delegate() const](https://www.mindspore.cn/lite/api/zh-CN/master/generate/classmindspore_Context.html#get-delegate)     |    ✕    |    √    |
| [std::shared_ptr\<Delegate\> GetDelegate() const](#getdelegate)     |    ✕    |    √    |
| [void SetMultiModalHW(bool float_mode)](#setmultimodalhw)     |    ✕    |    √    |
| [bool GetMultiModalHW() const](#getmultimodalhw)     |    ✕    |    √    |
| [std::vector\<std::shared_ptr\<DeviceInfoContext\>\> &MutableDeviceInfo()](#mutabledeviceinfo)     |    √    |    √    |

### SetThreadNum

```cpp
void SetThreadNum(int32_t thread_num)
```

设置运行时的线程数。

- 参数

    - `thread_num`: 运行时的线程数。

### GetThreadNum

```cpp
int32_t GetThreadNum() const
```

获取当前线程数设置。

- 返回值

  当前线程数设置。

### SetGroupInfoFile

```cpp
void SetGroupInfoFile(std::string group_info_file)
```

设置组信息文件。

- 参数

    - `group_info_file`: 组信息文件名字。

### SetInterOpParallelNum

```cpp
void SetInterOpParallelNum(int32_t parallel_num)
```

设置运行时的算子并行推理数目。

- 参数

    - `parallel_num`: 运行时的算子并行数。

### GetInterOpParallelNum

```cpp
int32_t GetInterOpParallelNum() const
```

获取当前算子并行数设置。

- 返回值

  当前算子并行数设置。

### SetThreadAffinity

```cpp
void SetThreadAffinity(int mode)
```

设置运行时的CPU绑核策略。

- 参数

    - `mode`: 绑核的模式，有效值为0-2，0为默认不绑核，1为绑大核，2为绑中核。

### GetThreadAffinityMode

```cpp
int GetThreadAffinityMode() const
```

获取当前CPU绑核策略。

- 返回值

  当前CPU绑核策略，有效值为0-2，0为默认不绑核，1为绑大核，2为绑中核。

### SetThreadAffinity

```cpp
void SetThreadAffinity(const std::vector<int> &core_list)
```

设置运行时的CPU绑核列表。如果SetThreadAffinity和SetThreadAffinity同时设置，core_list生效，mode不生效。

- 参数

    - `core_list`: CPU绑核的列表。

### GetThreadAffinityCoreList

```cpp
std::vector<int32_t> GetThreadAffinityCoreList() const
```

获取当前CPU绑核列表。

- 返回值

  当前CPU绑核列表。

### SetEnableParallel

```cpp
void SetEnableParallel(bool is_parallel)
```

设置运行时是否支持并行。

- 参数

    - `is_parallel`: bool量，为true则支持并行。

### GetEnableParallel

```cpp
bool GetEnableParallel() const
```

获取当前是否支持并行。

- 返回值

  返回值为true，代表支持并行。

### SetBuiltInDelegate

```cpp
void SetBuiltInDelegate(DelegateMode mode)
```

设置内置Delegate模式，以使用第三方AI框架辅助推理。

- 参数

    - `mode`: 内置Delegate模式，可选配置选项`kNoDelegate`、`kCoreML`、`kNNAPI`。`kNoDelegate`表示不使用第三方AI框架辅助推理，`kCoreML`表示使用CoreML进行推理（在iOS上可选），`kNNAPI`表示使用NNAPI进行推理（在Android上可选）。

### GetBuiltInDelegate

```cpp
DelegateMode GetBuiltInDelegate() const
```

获取当前内置Delegate模式。

- 返回值

  返回当前内置Delegate模式。

### set_delegate

```cpp
void set_delegate(const std::shared_ptr<AbstractDelegate> &delegate)
```

设置Delegate，Delegate定义了用于支持第三方AI框架接入的代理。

- 参数

    - `delegate`: Delegate指针。

### SetDelegate

```cpp
void SetDelegate(const std::shared_ptr<Delegate> &delegate)
```

设置Delegate，Delegate定义了用于支持第三方AI框架接入的代理。

- 参数

    - `delegate`: Delegate指针。

### get_delegate

```cpp
std::shared_ptr<AbstractDelegate> get_delegate() const
```

获取当前Delegate。

- 返回值

  当前Delegate的指针。

### GetDelegate

```cpp
std::shared_ptr<Delegate> GetDelegate() const
```

获取当前Delegate。

- 返回值

  当前Delegate的指针。

### SetMultiModalHW

```cpp
void SetMultiModalHW(bool float_mode)
```

在多设备中，配置量化模型是否以浮点模式运行。

- 参数

    - `float_mode`: 是否以浮点模式运行。

### GetMultiModalHW

```cpp
bool GetMultiModalHW() const
```

获取当前配置中，量化模型的运行模式。

- 返回值

  当前配置中，量化模型是否以浮点模式运行。

### MutableDeviceInfo

```cpp
std::vector<std::shared_ptr<DeviceInfoContext>> &MutableDeviceInfo()
```

修改该context下的[DeviceInfoContext](#deviceinfocontext)数组，仅端侧推理支持数组中有多个成员是异构场景。

- 返回值

  存储DeviceInfoContext的vector的引用。
