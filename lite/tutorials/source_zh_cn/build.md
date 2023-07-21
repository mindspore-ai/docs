# 编译

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.7/lite/tutorials/source_zh_cn/build.md)

本章节介绍如何快速编译出MindSpore Lite，其包含的模块如下：

| 模块 | 支持平台 | 说明 |
| --- | ---- | ---- |
| converter | Linux | 模型转换工具 |
| runtime | Linux、Android | 模型推理框架 |
| benchmark | Linux、Android | 基准测试工具 |
| time_profiler | Linux、Android | 性能分析工具 |

## Linux环境编译

### 环境要求

- 系统环境：Linux x86_64，推荐使用Ubuntu 18.04.02LTS

- runtime、benchmark、time_profiler编译依赖
  - [CMake](https://cmake.org/download/) >= 3.14.1
  - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
  - [Android_NDK](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip) >= r20
  - [Git](https://git-scm.com/downloads) >= 2.28.0

- converter编译依赖
  - [CMake](https://cmake.org/download/) >= 3.14.1
  - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
  - [Android_NDK](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip) >= r20
  - [Git](https://git-scm.com/downloads) >= 2.28.0
  - [Autoconf](http://ftp.gnu.org/gnu/autoconf/) >= 2.69
  - [Libtool](https://www.gnu.org/software/libtool/) >= 2.4.6
  - [LibreSSL](http://www.libressl.org/) >= 3.1.3
  - [Automake](https://www.gnu.org/software/automake/) >= 1.11.6
  - [Libevent](https://libevent.org) >= 2.0
  - [M4](https://www.gnu.org/software/m4/m4.html) >= 1.4.18
  - [OpenSSL](https://www.openssl.org/) >= 1.1.1 

> - 当安装完依赖项Android_NDK后，需配置环境变量:`export ANDROID_NDK={$NDK_PATH}/android-ndk-r20b`。
> - 编译脚本中会执行`git clone`获取第三方依赖库的代码，请提前确保git的网络设置正确可用。

### 编译选项

MindSpore Lite提供编译脚本`build.sh`用于一键式编译，位于MindSpore根目录下，该脚本可用于MindSpore训练及推理的编译。下面对MindSpore Lite的编译选项进行说明。

| 选项  |  参数说明  | 取值范围 | 是否必选 |
| -------- | ----- | ---- | ---- |
| **-I** | **选择适用架构，编译MindSpore Lite此选项必选** | **arm64、arm32、x86_64** | **是** |
| -d | 设置该参数，则编译Debug版本，否则编译Release版本 | 无 | 否 |
| -i | 设置该参数，则进行增量编译，否则进行全量编译 | 无 | 否 |
| -j[n] | 设定编译时所用的线程数，否则默认设定为8线程 | Integer | 否 |
| -e | 选择除CPU之外的其他内置算子类型，仅在ARM架构下适用，当前仅支持GPU | GPU | 否 |
| -h | 显示编译帮助信息 | 无 | 否 |

> 在`-I`参数变动时，如`-I x86_64`变为`-I arm64`，添加`-i`参数进行增量编译不生效。

### 编译示例

首先，在进行编译之前，需从MindSpore代码仓下载源码。

```bash
git clone https://gitee.com/mindspore/mindspore.git
```

然后，在源码根目录下执行如下命令，可编译不同版本的MindSpore Lite。

- 编译x86_64架构Debug版本。
    ```bash
    bash build.sh -I x86_64 -d
    ```

- 编译x86_64架构Release版本，同时设定线程数。
    ```bash
    bash build.sh -I x86_64 -j32
    ```

- 增量编译ARM64架构Release版本，同时设定线程数。
    ```bash
    bash build.sh -I arm64 -i -j32
    ```

- 编译ARM64架构Release版本，同时编译内置的GPU算子。
    ```bash
    bash build.sh -I arm64 -e gpu
    ```

### 编译输出

编译完成后，进入`mindspore/output/`目录，可查看编译后生成的文件。文件分为两部分：
- `mindspore-lite-{version}-converter-{os}.tar.gz`：包含模型转换工具converter。
- `mindspore-lite-{version}-runtime-{os}-{device}.tar.gz`：包含模型推理框架runtime、基准测试工具benchmark和性能分析工具time_profiler。

> version：输出件版本号，与所编译的分支代码对应的版本一致。
>
> device：当前分为cpu（内置CPU算子）和gpu（内置CPU和GPU算子）。
>
> os：输出件应部署的操作系统。

执行解压缩命令，获取编译后的输出件：

```bash
tar -xvf mindspore-lite-{version}-converter-{os}.tar.gz
tar -xvf mindspore-lite-{version}-runtime-{os}-{device}.tar.gz
```

#### 模型转换工具converter目录结构说明

转换工具仅在`-I x86_64`编译选项下获得，内容包括以下几部分：

```
|
├── mindspore-lite-{version}-converter-{os} 
│   └── converter # 模型转换工具
│   └── third_party # 第三方库头文件和库
│       ├── protobuf # Protobuf的动态库

```

#### 模型推理框架runtime及其他工具目录结构说明

推理框架可在`-I x86_64`、`-I arm64`和`-I arm32`编译选项下获得，内容包括以下几部分：

- 当编译选项为`-I x86_64`时：
    ```
    |
    ├── mindspore-lite-{version}-runtime-x86-cpu 
    │   └── benchmark # 基准测试工具
    │   └── lib # 推理框架动态库
    │       ├── libmindspore-lite.so # MindSpore Lite推理框架的动态库
    │   └── third_party # 第三方库头文件和库
    │       ├── flatbuffers # FlatBuffers头文件
    │   └── include # 推理框架头文件  
    │   └── time_profile # 模型网络层耗时分析工具
    
    ```

- 当编译选项为`-I arm64`时：
    ```
    |
    ├── mindspore-lite-{version}-runtime-arm64-cpu
    │   └── benchmark # 基准测试工具
    │   └── lib # 推理框架动态库
    │       ├── libmindspore-lite.so # MindSpore Lite推理框架的动态库
    │       ├── liboptimize.so # MindSpore Lite算子性能优化库  
    │   └── third_party # 第三方库头文件和库
    │       ├── flatbuffers # FlatBuffers头文件
    │   └── include # 推理框架头文件  
    │   └── time_profile # 模型网络层耗时分析工具
      
    ```

- 当编译选项为`-I arm32`时：
    ```
    |
    ├── mindspore-lite-{version}-runtime-arm64-cpu
    │   └── benchmark # 基准测试工具
    │   └── lib # 推理框架动态库
    │       ├── libmindspore-lite.so # MindSpore Lite推理框架的动态库
    │   └── third_party # 第三方库头文件和库
    │       ├── flatbuffers # FlatBuffers头文件
    │   └── include # 推理框架头文件  
    │   └── time_profile # 模型网络层耗时分析工具
      
    ```

> 1. `liboptimize.so`仅在runtime-arm64的输出包中存在，仅在ARMv8.2和支持fp16特性的CPU上使用。
> 2. 编译ARM64默认可获得arm64-cpu的推理框架输出件，若添加`-e gpu`则获得arm64-gpu的推理框架输出件，此时包名为`mindspore-lite-{version}-runtime-arm64-gpu.tar.gz`，编译ARM32同理。
> 3. 运行converter、benchmark或time_profile目录下的工具前，都需配置环境变量，将MindSpore Lite和Protobuf的动态库所在的路径配置到系统搜索动态库的路径中。以0.7.0-beta版本下编译为例：配置converter：`export LD_LIBRARY_PATH=./output/mindspore-lite-0.7.0-converter-ubuntu/third_party/protobuf/lib:${LD_LIBRARY_PATH}`；配置benchmark和timeprofiler：`export LD_LIBRARY_PATH=./output/mindspore-lite-0.7.0-runtime-x86-cpu/lib:${LD_LIBRARY_PATH}`
