# Class MultiModelRunner

\#include &lt;[multi_model_runner.h](https://atomgit.com/mindspore/mindspore-lite/blob/r2.9.0/include/api/multi_model_runner.h)&gt;

MultiModelRunner用于创建包含多个Model的mindir，并提供调度多个模型的方式。

## 构造函数

```c++
MultiModelRunner()
```

## 析构函数

```c++
~MultiModelRunner()
```

## 公有成员函数

| 函数                                                                                                                                                                                                                 | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|---------|
| [inline Status Build(const std::string &model_path, const ModelType &model_type, const std::shared_ptr\<Context\> &model_context = nullptr)](#build)     |    √    |    ✕   |
| [std::vector\<ModelExecutor\> GetModelExecutor() const](#getmodelexecutor)     |    √    |    ✕   |
| [inline Status LoadConfig(const std::string &config_path)](#loadconfig)     |    √    |    ✕   |
| [inline Status UpdateConfig(const std::string &section, const std::pair\<std::string, std::string\> &config)](#updateconfig)     |    √    |    ✕   |

### Build

```cpp
inline Status Build(const std::string &model_path, const ModelType &model_type,
             const std::shared_ptr<Context> &model_context = nullptr)
```

根据路径读取加载模型，并将模型编译至可在Device上运行的状态。与Model接口中同名接口的差别在于传入的model_path指定的mindir文件中包含多个Model。

- 参数

    - `model_path`: 模型文件路径。
    - `model_type`: `ModelType::kMindIR`，对应``mindir`模型（MindSpore导出或`converter_lite`工具导出）。
    - `model_context`: 模型[Context](#context)。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### GetModelExecutor

```cpp
std::vector<ModelExecutor> GetModelExecutor() const
```

获取MultiModelRunner中创建的多个ModelExecutor对象。

- 返回值

  包含MultiModelRunner中创建的所有ModelExecutor对象的列表。

### LoadConfig

```cpp
inline Status LoadConfig(const std::string &config_path)
```

根据路径读取配置文件。

- 参数

    - `config_path`: 配置文件路径。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### UpdateConfig

```cpp
inline Status UpdateConfig(const std::string &section, const std::pair<std::string, std::string> &config)
```

刷新配置，读文件相对比较费时，如果少部分配置发生变化可以通过该接口更新部分配置。

- 参数

    - `section`: 配置的章节名。
    - `config`: 要更新的配置对。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。
