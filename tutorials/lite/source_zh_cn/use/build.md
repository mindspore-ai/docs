# 编译MindSpore Lite

<!-- TOC -->

- [编译MindSpore Lite](#编译mindspore-lite)
    - [Linux环境编译](#linux环境编译)
        - [环境要求](#环境要求)
        - [编译选项](#编译选项)
        - [编译示例](#编译示例)
        - [编译输出](#编译输出)
            - [模型转换工具converter目录结构说明](#模型转换工具converter目录结构说明)
            - [模型推理框架runtime及其他工具目录结构说明](#模型推理框架runtime及其他工具目录结构说明)
            - [图像处理库目录结构说明](#图像处理库目录结构说明)
    - [Windows环境编译](#windows环境编译)  
        - [环境要求](#环境要求-1)
        - [编译选项](#编译选项-1)
        - [编译示例](#编译示例-1)
        - [编译输出](#编译输出-1)  
            - [模型转换工具converter目录结构说明](#模型转换工具converter目录结构说明-1)
            - [基准测试工具benchmark目录结构说明](#基准测试工具benchmark目录结构说明-1)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/lite/source_zh_cn/use/build.md" target="_blank"><img src="../_static/logo_source.png"></a>

本章节介绍如何快速编译出MindSpore Lite，其包含的模块如下：

| 模块 | 支持平台 | 说明 |
| --- | ---- | ---- |
| converter | Linux、Windows | 模型转换工具 |
| runtime(cpp、java) | Linux、Android | 模型推理框架(cpp、java) |
| benchmark | Linux、Windows、Android | 基准测试工具 |
| lib_cropper        | Linux          | libmindspore-lite.a静态库裁剪工具 |
| imageprocess | Linux、Android | 图像处理库 |

## Linux环境编译

### 环境要求

- 系统环境：Linux x86_64，推荐使用Ubuntu 18.04.02LTS
- runtime(cpp)、benchmark编译依赖
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
    - [Android_NDK](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip) >= r20
    - [Git](https://git-scm.com/downloads) >= 2.28.0
- converter编译依赖
    - [CMake](https://cmake.org/download/) >= 3.18.3
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
- runtime(java)编译依赖
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
    - [Android_NDK](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip) >= r20
    - [Git](https://git-scm.com/downloads) >= 2.28.0
    - [Android SDK](https://developer.android.com/studio?hl=zh-cn#cmdline-tools)
    - [Gradle](https://gradle.org/releases/) >= 6.6.1
    - [OpenJDK](https://openjdk.java.net/install/) >= 1.8

> - 当安装完依赖项Android_NDK后，需配置环境变量：`export ANDROID_NDK=${NDK_PATH}/android-ndk-r20b`。
> - 编译脚本中会执行`git clone`获取第三方依赖库的代码，请提前确保git的网络设置正确可用。
> - 当安装完依赖项Gradle后，需将其安装路径增加到PATH当中：`export PATH=${GRADLE_PATH}/bin:$PATH`。
> - 通过`Android command line tools`安装Android SDK，首先需要创建一个新目录，并将其路径配置到环境变量`${ANDROID_SDK_ROOT}`中，然后通过`sdkmanager`创建SDK：`./sdkmanager --sdk_root=${ANDROID_SDK_ROOT} "cmdline-tools;latest"`，最后通过`${ANDROID_SDK_ROOT}`目录下的`sdkmanager`接受许可证：`yes | ./sdkmanager --licenses`。
> - 编译AAR需要依赖Android SDK Build-Tools、Android SDK Platform-Tools等Android SDK相关组件，如果环境中的Android SDK不存在相关组件，编译时会自动下载所需依赖。

### 编译选项

MindSpore Lite提供编译脚本`build.sh`用于一键式编译，位于MindSpore根目录下，该脚本可用于MindSpore训练及推理的编译。下面对MindSpore Lite的编译选项进行说明。

| 选项  |  参数说明  | 取值范围 | 是否必选 |
| -------- | ----- | ---- | ---- |
| -I | 选择适用架构，若编译MindSpore Lite c++版本，则此选项必选 | arm64、arm32、x86_64 | 否 |
| -d | 设置该参数，则编译Debug版本，否则编译Release版本 | 无 | 否 |
| -i | 设置该参数，则进行增量编译，否则进行全量编译 | 无 | 否 |
| -j[n] | 设定编译时所用的线程数，否则默认设定为8线程 | Integer | 否 |
| -e | 选择除CPU之外的其他内置算子类型，仅在ARM架构下适用，当前仅支持GPU | GPU | 否 |
| -h | 显示编译帮助信息 | 无 | 否 |
| -n | 指定编译轻量级图片处理模块 | lite_cv | 否 |
| -A | 指定编译语言，默认cpp。设置为java时，则编译AAR包 | cpp、java | 否 |
| -C | 设置该参数，则编译模型转换工具，默认为on | on、off | 否 |
| -o | 设置该参数，则编译基准测试工具，默认为on | on、off | 否 |
| -t | 设置该参数，则编译测试用例，默认为off | on、off | 否 |

> 在`-I`参数变动时，如`-I x86_64`变为`-I arm64`，添加`-i`参数进行增量编译不生效。
>
> 编译AAR包时，必须添加`-A java`参数，且无需添加`-I`参数。

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

- 编译x86_64架构Release版本，同时编译测试用例。

    ```bash
    bash build.sh -I x86_64 -t on
    ```

- 增量编译ARM64架构Release版本，同时设定线程数。

    ```bash
    bash build.sh -I arm64 -i -j32
    ```

- 编译ARM64架构Release版本，同时编译内置的GPU算子。

    ```bash
    bash build.sh -I arm64 -e gpu
    ```

- 编译ARM64带图像预处理模块。

    ```bash
    bash build.sh -I arm64 -n lite_cv
    ```

- 增量编译MindSpore Lite AAR。

  ```bash
  bash build.sh -A java -i
  ```

  > 开启增量编译后，若arm64、arm32的runtime已经存在于`mindspore/output/`目录，将不会重新编译对应版本的runtime。

- 编译x86_64架构Release版本，同时编译模型转换工具。

  ```bash
  bash build.sh -I x86_64 -C on
  ```

- 编译x86_64架构Release版本，同时编译基准测试工具、库裁剪工具。

  ```bash
  bash build.sh -I x86_64 -o on
  ```

### 编译输出

编译完成后，进入`mindspore/output/`目录，可查看编译后生成的文件。文件分为以下几种：

- `mindspore-lite-{version}-converter-{os}.tar.gz`：包含模型转换工具converter。
- `mindspore-lite-{version}-runtime-{os}-{device}.tar.gz`：包含模型推理框架runtime、基准测试工具benchmark、库裁剪工具lib_cropper。
- `mindspore-lite-{version}-minddata-{os}-{device}.tar.gz`：包含图像处理库imageprocess。
- `mindspore-lite-maven-{version}.zip`：包含模型推理框架runtime(java)的AAR。

> version：输出件版本号，与所编译的分支代码对应的版本一致。
>
> device：当前分为cpu（内置CPU算子）和gpu（内置CPU和GPU算子）。
>
> os：输出件应部署的操作系统。

执行解压缩命令，获取编译后的输出件：

```bash
tar -xvf mindspore-lite-{version}-converter-{os}.tar.gz
tar -xvf mindspore-lite-{version}-runtime-{os}-{device}.tar.gz
tar -xvf mindspore-lite-{version}-minddata-{os}-{device}.tar.gz
unzip mindspore-lite-maven-{version}.zip
```

#### 模型转换工具converter目录结构说明

转换工具仅在`-I x86_64`编译选项下获得，内容包括以下几部分：

```text
|
├── mindspore-lite-{version}-converter-{os}
│   └── converter # 模型转换工具
|       ├── converter_lite # 可执行程序
│   └── lib # 转换工具依赖的动态库
|       ├── libmindspore_gvar.so # 存储某些全局变量的动态库
│   └── third_party # 第三方库头文件和库
|       ├── glog # Glog的动态库
```

#### 模型推理框架runtime及其他工具目录结构说明

推理框架可在`-I x86_64`、`-I arm64`、`-I arm32`和`-A java`编译选项下获得，内容包括以下几部分：

- 当编译选项为`-I x86_64`时：

    ```text
    |
    ├── mindspore-lite-{version}-runtime-x86-cpu
    │   └── benchmark # 基准测试工具
    |   └── lib_cropper # 库裁剪工具
    │       ├── lib_cropper  # 库裁剪工具可执行文件
    │       ├── cropper_mapping_cpu.cfg # 裁剪cpu库所需的配置文件
    │   └── include # 推理框架头文件
    │   └── lib # 推理框架库
    │       ├── libmindspore-lite.a  # MindSpore Lite推理框架的静态库
    │       ├── libmindspore-lite.so # MindSpore Lite推理框架的动态库
    │   └── minddata # 图像处理动态库
    │       └── include # 头文件
    │           └── lite_cv # 图像处理库头文件
    │               ├── image_process.h # 图像处理函数头文件
    │               ├── lite_mat.h # 图像数据类结构头文件
    │       └── lib # 图像处理动态库
    │           ├── libminddata-lite.so # 图像处理动态库文件
    │   └── third_party # 第三方库头文件和库
    │       ├── flatbuffers # FlatBuffers头文件
    ```

- 当编译选项为`-I arm64`时：

    ```text
    |
    ├── mindspore-lite-{version}-runtime-arm64-cpu
    │   └── benchmark # 基准测试工具
    │   └── include # 推理框架头文件  
    │   └── lib # 推理框架库
    │       ├── libmindspore-lite.a  # MindSpore Lite推理框架的静态库
    │       ├── libmindspore-lite.so # MindSpore Lite推理框架的动态库
    │   └── minddata # 图像处理动态库
    │       └── include # 头文件
    │           └── lite_cv # 图像处理库头文件
    │               ├── image_process.h # 图像处理函数头文件
    │               ├── lite_mat.h # 图像数据类结构头文件
    │       └── lib # 图像处理动态库
    │           ├── libminddata-lite.so # 图像处理动态库文件
    │   └── third_party # 第三方库头文件和库
    │       ├── flatbuffers # FlatBuffers头文件
    ```

- 当编译选项为`-I arm32`时：

    ```text
    |
    ├── mindspore-lite-{version}-runtime-arm32-cpu
    │   └── benchmark # 基准测试工具
    │   └── include # 推理框架头文件  
    │   └── lib # 推理框架库
    │       ├── libmindspore-lite.a  # MindSpore Lite推理框架的静态库
    │       ├── libmindspore-lite.so # MindSpore Lite推理框架的动态库
    │   └── minddata # 图像处理动态库
    │       └── include # 头文件
    │           └── lite_cv # 图像处理库头文件
    │               ├── image_process.h # 图像处理函数头文件
    │               ├── lite_mat.h # 图像数据类结构头文件
    │       └── lib # 图像处理动态库
    │           ├── libminddata-lite.so # 图像处理动态库文件
    │   └── third_party # 第三方库头文件和库
    │       ├── flatbuffers # FlatBuffers头文件
    ```

- 当编译选项为`-A java`时：

  ```text
  |
  ├── mindspore-lite-maven-{version}
  │   └── mindspore
  │       └── mindspore-lite
  |           └── {version}
  │               ├── mindspore-lite-{version}.aar # MindSpore Lite推理框架aar包
  ```

> 1. 编译ARM64默认可获得arm64-cpu的推理框架输出件，若添加`-e gpu`则获得arm64-gpu的推理框架输出件，此时包名为`mindspore-lite-{version}-runtime-arm64-gpu.tar.gz`，编译ARM32同理。
> 2. 运行converter、benchmark目录下的工具前，都需配置环境变量，将MindSpore Lite的动态库所在的路径配置到系统搜索动态库的路径中。

配置converter：

```bash
export LD_LIBRARY_PATH=./output/mindspore-lite-{version}-converter-ubuntu/lib:./output/mindspore-lite-{version}-converter-ubuntu/third_party/glog/lib:${LD_LIBRARY_PATH}
```

配置benchmark：

```bash
export LD_LIBRARY_PATH=./output/mindspore-lite-{version}-runtime-x86-cpu/lib:${LD_LIBRARY_PATH}
```

#### 图像处理库目录结构说明

图像处理库在`-I arm64 -n lite_cv`编译选项下获得，内容包括以下几部分：

```text
|
├── mindspore-lite-{version}-runtime-{os}-cpu
│   └── benchmark # 基准测试工具
│   └── lib # 推理框架态库
│       ├── libmindspore-lite.a  # MindSpore Lite推理框架的静态库
│       ├── libmindspore-lite.so # MindSpore Lite推理框架的动态库
│   └── minddata # 图像处理动态库
│       └── include # 头文件
│           └── lite_cv # 图像处理库头文件
│               ├── image_process.h # 图像处理函数头文件
│               ├── lite_mat.h # 图像数据类结构头文件
│       └── lib # 图像处理动态库
│           ├── libminddata-lite.so # 图像处理动态库文件
│   └── third_party # 第三方库头文件和库
│       ├── flatbuffers # Flatbuffers的动态库
```

## Windows环境编译

### 环境要求

- 系统环境：Windows 7，Windows 10；64位。

- 编译依赖
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [MinGW GCC](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/7.3.0/threads-posix/seh/x86_64-7.3.0-release-posix-seh-rt_v5-rev0.7z/download) >= 7.3.0

> 编译脚本中会执行`git clone`获取第三方依赖库的代码，请提前确保git的网络设置正确可用。

### 编译选项

MindSpore Lite提供编译脚本build.bat用于一键式编译，位于MindSpore根目录下，该脚本可用于MindSpore训练及推理的编译。下面对MindSpore Lite的编译选项进行说明。

| 参数  |  参数说明  | 是否必选 |
| -------- | ----- | ---- |
| lite | 设置该参数，则对MindSpore Lite工程进行编译 | 是 |
| [n] | 设定编译时所用的线程数，否则默认设定为6线程  | 否 |

### 编译示例

首先，使用git工具，从MindSpore代码仓下载源码。

```bat
git clone https://gitee.com/mindspore/mindspore.git
```

然后，使用cmd工具在源码根目录下，执行如下命令即可编译MindSpore Lite。

- 以默认线程数（6线程）编译Windows版本。

```bat
call build.bat lite
```

- 以指定线程数8编译Windows版本。

```bat
call build.bat lite 8
```

### 编译输出

编译完成后，进入`mindspore/output/`目录，可查看编译后生成的文件。文件分为以下几种：

- `mindspore-lite-{version}-converter-win-cpu.zip`：包含模型转换工具converter。
- `mindspore-lite-{version}-win-runtime-x86-cpu.zip`：包含基准测试工具benchmark。

> version：输出件版本号，与所编译的分支代码对应的版本一致。

执行解压缩命令，获取编译后的输出件：

```bat
unzip mindspore-lite-{version}-converter-win-cpu.zip
unzip mindspore-lite-{version}-win-runtime-x86-cpu.zip
```

#### 模型转换工具converter目录结构说明

转换工具的内容包括以下几部分：

```text
|
├── mindspore-lite-{version}-converter-win-cpu
│   └── converter # 模型转换工具
|       ├── converter_lite.exe # 可执行程序
|       ├── libglog.dll # Glog的动态库
|       ├── libmindspore_gvar.dll # 存储某些全局变量的动态库
|       ├── libgcc_s_seh-1.dll # MinGW动态库
|       ├── libssp-0.dll # MinGW动态库
|       ├── libstdc++-6.dll # MinGW动态库
|       ├── libwinpthread-1.dll # MinGW动态库
```

#### 基准测试工具benchmark目录结构说明

基准测试工具的内容包括以下几部分：

```text
|
├── mindspore-lite-{version}-win-runtime-x86-cpu
│   └── benchmark # 基准测试工具
│       ├── benchmark.exe # 可执行程序
│       ├── libmindspore-lite.a  # MindSpore Lite推理框架的静态库
│       ├── libmindspore-lite.dll # MindSpore Lite推理框架的动态库
│       ├── libmindspore-lite.dll.a # MindSpore Lite推理框架的动态库的链接文件
│       ├── libgcc_s_seh-1.dll # MinGW动态库
|       ├── libssp-0.dll # MinGW动态库
|       ├── libstdc++-6.dll # MinGW动态库
|       ├── libwinpthread-1.dll # MinGW动态库
│   └── include # 推理框架头文件  
│   └── third_party # 第三方库头文件和库
│       ├── flatbuffers # FlatBuffers头文件
```
