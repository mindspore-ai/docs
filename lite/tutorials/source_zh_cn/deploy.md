# 部署

<!-- TOC -->

- [部署](#部署)
    - [Linux环境部署](#linux环境部署)
        - [环境要求](#环境要求)
        - [编译选项](#编译选项)
        - [输出件说明](#输出件说明)
        - [编译示例](#编译示例)
    - [Windows环境部署](#windows环境部署)  
        - [环境要求](#环境要求-1)
        - [编译选项](#编译选项-1)
        - [输出件说明](#输出件说明-1)
        - [编译示例](#编译示例-1)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/lite/tutorials/source_zh_cn/deploy.md" target="_blank"><img src="./_static/logo_source.png"></a>

本文档介绍如何在Ubuntu和Windows系统上快速安装MindSpore Lite。

## Linux环境部署

### 环境要求

- 编译环境仅支持x86_64版本的Linux：推荐使用Ubuntu 18.04.02LTS

- 编译依赖（基本项）
  - [CMake](https://cmake.org/download/) >= 3.14.1
  - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
  - [Python](https://www.python.org/) >= 3.7
  - [Android_NDK r20b](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip)
  
  > - 仅在编译ARM版本时需要安装`Android_NDK`，编译x86_64版本可跳过此项。
  > - 如果安装并使用`Android_NDK`，需配置环境变量，命令参考：`export ANDROID_NDK={$NDK_PATH}/android-ndk-r20b`。
                                                                              
- 编译依赖（MindSpore Lite模型转换工具所需附加项，仅编译x86_64版本时需要）
  - [Autoconf](http://ftp.gnu.org/gnu/autoconf/) >= 2.69
  - [Libtool](https://www.gnu.org/software/libtool/) >= 2.4.6
  - [LibreSSL](http://www.libressl.org/) >= 3.1.3
  - [Automake](https://www.gnu.org/software/automake/) >= 1.11.6
  - [Libevent](https://libevent.org) >= 2.0
  - [M4](https://www.gnu.org/software/m4/m4.html) >= 1.4.18
  - [OpenSSL](https://www.openssl.org/) >= 1.1.1 
  
  
### 编译选项

MindSpore Lite提供多种编译方式，用户可根据需要选择不同的编译选项。

| 参数  |  参数说明  | 取值范围 | 是否必选 |
| -------- | ----- | ---- | ---- |
| -d | 设置该参数，则编译Debug版本，否则编译Release版本 | - | 否 |
| -i | 设置该参数，则进行增量编译，否则进行全量编译 | - | 否 |
| -j[n] | 设定编译时所用的线程数，否则默认设定为8线程 | - | 否 |
| -I | 选择适用架构 | arm64、arm32、x86_64 | 是 |
| -e | 在ARM架构下，选择后端算子，设置`gpu`参数，会同时编译框架内置的GPU算子 | gpu | 否 |
| -h | 设置该参数，显示编译帮助信息 | - | 否 |

> 在`-I`参数变动时，即切换适用架构时，无法使用`-i`参数进行增量编译。

### 输出件说明

编译完成后，进入源码的`mindspore/output`目录，可查看编译后生成的文件，命名为`mindspore-lite-{version}-{function}-{OS}.tar.gz`。解压后，即可获得编译后的工具包，名称为`mindspore-lite-{version}-{function}-{OS}`。

> version：输出件版本，与所编译的MindSpore版本一致。
>
> function：输出件功能，`convert`表示为转换工具的输出件，`runtime`表示为推理框架的输出件。
>
> OS：输出件应部署的操作系统。

```bash
tar -xvf mindspore-lite-{version}-{function}-{OS}.tar.gz
```
编译x86可获得转换工具`converter`与推理框架`runtime`功能的输出件，编译ARM仅能获得推理框架`runtime`。

输出件中包含以下几类子项，功能不同所含内容也会有所区别。

> 编译ARM64默认可获得`arm64-cpu`的推理框架输出件，若添加`-e gpu`则获得`arm64-gpu`的推理框架输出件，编译ARM32同理。

| 目录 | 说明 | converter | runtime |
| --- | --- | --- | --- |
| include | 推理框架头文件 | 无 | 有 |
| lib | 推理框架动态库 | 无 | 有 | 
| benchmark | 基准测试工具 | 无 | 有 | 
| time_profiler | 模型网络层耗时分析工具 | 无 | 有 | 
| converter | 模型转换工具 | 有 | 无 | 
| third_party | 第三方库头文件和库 | 有 | 有 | 

以0.7.0-beta版本，CPU编译为例，不同包名下，`third party`与`lib`的内容不同：
  
- `mindspore-lite-0.7.0-converter-ubuntu`：包含`protobuf`（Protobuf的动态库）。
- `mindspore-lite-0.7.0-runtime-x86-cpu`：`third party`包含`flatbuffers`（FlatBuffers头文件），`lib`包含`libmindspore-lite.so`（MindSpore Lite的动态库）。
- `mindspore-lite-0.7.0-runtime-arm64-cpu`：`third party`包含`flatbuffers`（FlatBuffers头文件），`lib`包含`libmindspore-lite.so`（MindSpore Lite的动态库）和`liboptimize.so`。
TODO：补全文件内容

> 运行converter、benchmark或time_profiler目录下的工具前，都需配置环境变量，将MindSpore Lite和Protobuf的动态库所在的路径配置到系统搜索动态库的路径中。以0.7.0-beta版本为例：`export LD_LIBRARY_PATH=./mindspore-lite-0.7.0/lib:./mindspore-lite-0.7.0/third_party/protobuf/lib:${LD_LIBRARY_PATH}`。

### 编译示例

首先，从MindSpore代码仓下载源码。

```bash
git clone https://gitee.com/mindspore/mindspore.git
```

然后，在源码根目录下，执行如下命令，可编译不同版本的MindSpore Lite。

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
    
> `build.sh`中会执行`git clone`获取第三方依赖库的代码，请提前确保git的网络设置正确可用。
   
以0.7.0-beta版本为例，x86_64架构Release版本编译完成之后，进入`mindspore/output`目录，执行如下解压缩命令，即可获取输出件`include`、`lib`、`benchmark`、`time_profiler`、`converter`和`third_party`。
   
```bash
tar -xvf mindspore-lite-0.7.0-converter-ubuntu.tar.gz
tar -xvf mindspore-lite-0.7.0-runtime-x86-cpu.tar.gz
```

## Windows环境部署

### 环境要求

- 编译环境仅支持32位或64位Windows系统

- 编译依赖（基本项）
  - [CMake](https://cmake.org/download/) >= 3.14.1
  - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

### 编译选项

MindSpore Lite的编译选项如下。

| 参数  |  参数说明  | 取值范围 | 是否必选 |
| -------- | ----- | ---- | ---- |
| lite | 设置该参数，则对Mindspore Lite工程进行编译，否则对Mindspore工程进行编译 | - | 是 |
| [n] | 设定编译时所用的线程数，否则默认设定为6线程 | - | 否 |

### 输出件说明

编译完成后，进入源码的`mindspore/output/`目录，可查看编译后生成的文件，命名为命名为`mindspore-lite-{version}-converter-win-{process_unit}.zip`。解压后，即可获得编译后的工具包，名称为`mindspore-lite-{version}`。

> version：输出件版本，与所编译的MindSpore版本一致。
> process_unit：输出件应部署的处理器类型。

### 编译示例

首先，使用git工具从MindSpore代码仓下载源码。

```bash
git clone https://gitee.com/mindspore/mindspore.git
```

然后，使用cmd工具在源码根目录下，执行如下命令即可编译MindSpore Lite。

- 以默认线程数（6线程）编译Windows版本。
    ```bash
    call build.bat lite
    ```
- 以指定线程数8编译Windows版本。
    ```bash
    call build.bat lite 8
    ```
   
> `build.bat`中会执行`git clone`获取第三方依赖库的代码，请提前确保git的网络设置正确可用。
   
编译完成之后，进入`mindspore/output/`目录，解压后即可获取输出件`converter`。
