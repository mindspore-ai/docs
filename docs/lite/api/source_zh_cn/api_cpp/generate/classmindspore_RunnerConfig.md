# Class RunnerConfig

\#include &lt;[model_parallel_runner.h](https://atomgit.com/mindspore/mindspore-lite/blob/master/include/api/model_parallel_runner.h)&gt;

RunnerConfig定义了ModelParallelRunner中使用的配置选项参数。

## 构造函数和析构函数

```cpp
RunnerConfig()
~RunnerConfig()
```

## 公有成员函数

| 函数                   | 云侧推理是否支持 | 端侧推理是否支持 |
|-------------------------------------------------------------|---------|---------|
| [void SetWorkersNum(int32_t workers_num)](#setworkersnum)     |    √    |    ✕    |
| [int32_t GetWorkersNum() const](#getworkersnum)     |    √    |    ✕    |
| [void SetContext(const std::shared_ptr\<Context\> &context)](#setcontext)     |    √    |    ✕    |
| [std::shared_ptr\<Context\> GetContext() const](#getcontext)     |    √    |    ✕    |
| [inline void SetConfigInfo(const std::string &section, const std::map\<std::string, std::string\> &config)](#setconfiginfo)     |    √    |    ✕    |
| [inline std::map\<std::string, std::map\<std::string, std::string\>\> GetConfigInfo() const](#getconfiginfo)     |    √    |    ✕    |
| [inline void SetConfigPath(const std::string &config_path)](#setconfigpath)     |    √    |    ✕    |
| [inline std::string GetConfigPath() const](#getconfigpath)     |    √    |    ✕    |
| [void SetDeviceIds(const std::vector\<uint32_t\> &device_ids)](#setdeviceids)     |    √    |    ✕    |
| [std::vector\<uint32_t\> GetDeviceIds() const](#getdeviceids)     |    √    |    ✕    |

### SetWorkersNum

```cpp
void SetWorkersNum(int32_t workers_num)
```

设置RunnerConfig的worker的个数。

- 参数

    - `workers_num`: worker的数量。

### GetWorkersNum

```cpp
int32_t GetWorkersNum() const
```

获取RunnerConfig的worker的个数。

- 返回值

  RunnerConfig类中配置的worker数量。

### SetContext

```cpp
void SetContext(const std::shared_ptr<Context> &context)
```

设置RunnerConfig的context参数。

- 参数

    - `context`: worker上下文配置。

### GetContext

```cpp
std::shared_ptr<Context> GetContext() const
```

获取RunnerConfig配置的上下文参数。

- 返回值

  上下文配置类`Context`对象。

### SetConfigInfo

```cpp
void SetConfigInfo(const std::string &key, const std::map<std::string, std::string> &config)
```

设置RunnerConfig的配置参数。

- 参数

    - `key`: string类型关键字。
    - `config`: map类型的配置参数。

### GetConfigInfo

```cpp
std::map<std::string, std::map<std::string, std::string>> GetConfigInfo() const
```

获取RunnerConfig配置参数信息。

- 返回值

  `map`类型的配置信息。

### SetConfigPath

```cpp
void SetConfigPath(const std::string &config_path)
```

设置RunnerConfig中的配置文件路径。

- 参数

    - `config_path`: 配置文件路径。

### GetConfigPath

```cpp
std::string GetConfigPath() const
```

获取RunnerConfig中的配置文件的路径。

- 返回值

  RunnerConfig类中的配置文件路径。

### SetDeviceIds

```cpp
void SetDeviceIds(const std::vector<uint32_t> &device_ids)
```

设置RunnerConfig中的设备ID列表。

- 参数

    - `device_ids`: 设备ID列表。

### GetDeviceIds

```cpp
std::vector<uint32_t> GetDeviceIds() const
```

获取RunnerConfig中的设备ID列表。

- 返回值

  RunnerConfig类中的设备ID列表。
