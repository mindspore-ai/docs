# 编译MindSpore Lite

`Windows` `Linux` `Mac` `Android` `iOS` `环境准备` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/lite/docs/source_zh_cn/use/build.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

本章节介绍如何快速编译出MindSpore Lite。

MindSpore Lite包含模块：

| 模块               | 支持平台                | 说明                              |
| ------------------ | ----------------------- | --------------------------------- |
| converter          | Linux, Windows          | 模型转换工具                      |
| runtime(cpp、java) | Linux, Windows, Android, iOS | 模型推理框架（Windows平台不支持java版runtime） |
| benchmark          | Linux, Windows, Android | 基准测试工具                      |
| benchmark_train    | Linux, Android          | 性能测试和精度校验工具              |
| cropper            | Linux                   | libmindspore-lite.a静态库裁剪工具 |
| minddata           | Linux, Android          | 图像处理库                        |
| codegen            | Linux                   | 模型推理代码生成工具               |
| obfuscator         | Linux                   | 模型混淆工具                      |

## Linux环境编译

### 环境要求

- 系统环境：Linux x86_64，推荐使用Ubuntu 18.04.02LTS
- C++编译依赖
    - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [Git](https://git-scm.com/downloads) >= 2.28.0
    - [Android_NDK](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip) >= r20
        - 配置环境变量：`export ANDROID_NDK=NDK路径`
- Java编译需要的额外依赖
    - [Gradle](https://gradle.org/releases/) >= 6.6.1
        - 配置环境变量：`export GRADLE_HOME=GRADLE路径`
        - 将bin目录添加到PATH中：`export PATH=${GRADLE_HOME}/bin:$PATH`
    - [OpenJDK](https://openjdk.java.net/install/) >= 1.8
        - 配置环境变量：`export JAVA_HOME=JDK路径`
        - 将bin目录添加到PATH中：`export PATH=${JAVA_HOME}/bin:$PATH`
    - [Android SDK](https://developer.android.com/studio?hl=zh-cn#cmdline-tools)
        - 创建一个新目录，配置环境变量`export ANDROID_SDK_ROOT=新建的目录`
        - 下载`SDK Tools`，通过`sdkmanager`创建SDK：`./sdkmanager --sdk_root=${ANDROID_SDK_ROOT} "cmdline-tools;latest"`
        - 通过`${ANDROID_SDK_ROOT}`目录下的`sdkmanager`接受许可证：`yes | ./sdkmanager --licenses`

> 也可直接使用已配置好上述依赖的Docker编译镜像。
>
> - 下载镜像：`docker pull swr.cn-south-1.myhuaweicloud.com/mindspore-build/mindspore-lite:ubuntu18.04.2-20210530`
> - 创建容器：`docker run -tid --net=host --name=docker01 swr.cn-south-1.myhuaweicloud.com/mindspore-build/mindspore-lite:ubuntu18.04.2-20210530`
> - 进入容器：`docker exec -ti -u 0 docker01 bash`

### 编译选项

MindSpore根目录下的`build.sh`脚本可用于MindSpore Lite的编译。

#### `build.sh`的编译参数

| 参数  |  参数说明  | 取值范围 | 默认值 |
| -------- | ----- | ---- | ---- |
| -I | 选择目标架构 | arm64、arm32、x86_64 | 无 |
| -A | 编译AAR包(包含arm32和arm64) | on、off | off |
| -d | 设置该参数，则编译Debug版本，否则编译Release版本 | 无 | 无 |
| -i | 设置该参数，则进行增量编译，否则进行全量编译 | 无 | 无 |
| -j[n] | 设定编译时所用的线程数，否则默认设定为8线程 | Integer | 8 |
| -a | 是否启用AddressSanitizer | on、off | off |

> - 编译x86_64版本时，若配置了JAVA_HOME环境变量并安装了Gradle，则同时编译JAR包。
> - 在`-I`参数变动时，如`-I x86_64`变为`-I arm64`，添加`-i`参数进行增量编译不生效。
> - 编译AAR包时，必须添加`-A on`参数，且无需添加`-I`参数。

#### `mindspore/lite/CMakeLists.txt`的选项

| 选项  |  参数说明  | 取值范围 | 默认值 |
| -------- | ----- | ---- | ---- |
| MSLITE_GPU_BACKEND | 设置GPU后端，在`-I arm64`时仅opencl有效，在`-I x86_64`时仅tensorrt有效 | opencl、tensorrt、off | 在`-I arm64`时为opencl， 在`-I x86_64`时为off |
| MSLITE_ENABLE_NPU | 是否编译NPU算子，仅在`-I arm64`或`-I arm32`时有效 | on、off | off |
| MSLITE_ENABLE_TRAIN | 是否编译训练版本 | on、off | on |
| MSLITE_ENABLE_SSE | 是否启用SSE指令集，仅在`-I x86_64`时有效 | on、off | off |
| MSLITE_ENABLE_AVX | 是否启用AVX指令集，仅在`-I x86_64`时有效 | on、off | off |
| MSLITE_ENABLE_CONVERTER | 是否编译模型转换工具，仅在`-I x86_64`时有效 | on、off | on |
| MSLITE_ENABLE_TOOLS | 是否编译配套工具 | on、off | on |
| MSLITE_ENABLE_TESTCASES | 是否编译测试用例 | on、off | off |

> - 以上选项可通过设置同名环境变量或者`mindspore/lite/CMakeLists.txt`文件修改。
> - 修改选项后，添加`-i`参数进行增量编译不生效。

### 编译示例

首先，在进行编译之前，需从MindSpore代码仓下载源码。

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.3
```

然后，在源码根目录下执行如下命令，可编译不同版本的MindSpore Lite。

- 编译x86_64架构版本，同时设定线程数。

    ```bash
    bash build.sh -I x86_64 -j32
    ```

- 编译ARM64架构版本，不编译训练相关的代码。

    ```bash
    export MSLITE_ENABLE_TRAIN=off
    bash build.sh -I arm64 -j32
    ```

    或者修改`mindspore/lite/CMakeLists.txt`将 MSLITE_ENABLE_TRAIN 设置为 off 后，执行命令:

    ```bash
    bash build.sh -I arm64 -j32
    ```

- 编译包含aarch64和aarch32的AAR包。

    ```bash
    bash build.sh -A on -j32
    ```

最后，会在`output/`目录中生成如下文件：

- `mindspore-lite-{version}-{os}-{arch}.tar.gz`：包含runtime和配套工具。

- `mindspore-lite-maven-{version}.zip`：包含runtime(java)的AAR包。

> - version: 输出件版本号，与所编译的分支代码对应的版本一致。
> - os: 输出件应部署的操作系统。
> - arch: 输出件应部署的系统架构。

### 目录结构

- 当编译选项为`-I x86_64`时：

    ```text
    mindspore-lite-{version}-linux-x64
    ├── runtime
    │   ├── include
    │   ├── lib
    │   │   ├── libminddata-lite.a         # 图像处理静态库
    │   │   ├── libminddata-lite.so        # 图像处理动态库
    │   │   ├── libmindspore-lite.a        # MindSpore Lite推理框架的静态库
    │   │   ├── libmindspore-lite-jni.so   # MindSpore Lite推理框架的jni动态库
    │   │   ├── libmindspore-lite.so       # MindSpore Lite推理框架的动态库
    │   │   ├── libmindspore-lite-train.a  # MindSpore Lite训练框架的静态库
    │   │   ├── libmindspore-lite-train.so # MindSpore Lite训练框架的动态库
    │   │   ├── libmsdeobfuscator-lite.so  # 混淆模型加载动态库文件，需开启`ENABLE_MODEL_OBF`选项。
    │   │   └── mindspore-lite-java.jar    # MindSpore Lite推理框架jar包
    │   └── third_party
    │       └── libjpeg-turbo
    └── tools
        ├── benchmark       # 基准测试工具
        ├── benchmark_train # 训练模型性能与精度调测工具
        ├── codegen         # 代码生成工具
        ├── converter       # 模型转换工具
        ├── obfuscator      # 模型混淆工具
        └── cropper         # 库裁剪工具
    ```

- 当编译选项为`-I arm64`或`-I arm32`时：

    ```text
    mindspore-lite-{version}-android-{arch}
    ├── runtime
    │   ├── include
    │   ├── lib
    │   │   ├── libminddata-lite.a         # 图像处理静态库
    │   │   ├── libminddata-lite.so        # 图像处理动态库
    │   │   ├── libmindspore-lite.a        # MindSpore Lite推理框架的静态库
    │   │   ├── libmindspore-lite.so       # MindSpore Lite推理框架的动态库
    │   │   ├── libmindspore-lite-train.a  # MindSpore Lite训练框架的静态库
    │   │   ├── libmindspore-lite-train.so # MindSpore Lite训练框架的动态库
    │   │   └── libmsdeobfuscator-lite.so  # 混淆模型加载动态库文件，需开启`ENABLE_MODEL_OBF`选项。
    │   └── third_party
    │       ├── hiai_ddk
    │       └── libjpeg-turbo
    └── tools
        ├── benchmark       # 基准测试工具
        ├── benchmark_train # 训练模型性能与精度调测工具
        └── codegen         # 代码生成工具
    ```

- 当编译选项为`-A on`时：

    ```text
    mindspore-lite-maven-{version}
    └── mindspore
        └── mindspore-lite
            └── {version}
                └── mindspore-lite-{version}.aar # MindSpore Lite推理框架aar包
    ```

## Windows环境编译

### 环境要求

- 系统环境：Windows 7，Windows 10；64位。

- 编译依赖
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [MinGW GCC](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/7.3.0/threads-posix/seh/x86_64-7.3.0-release-posix-seh-rt_v5-rev0.7z/download) = 7.3.0

> - 编译脚本中会执行`git clone`获取第三方依赖库的代码。
> - 如果要编译32位Mindspore Lite，请使用32位[MinGW](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win32/Personal%20Builds/mingw-builds/7.3.0/threads-posix/dwarf/i686-7.3.0-release-posix-dwarf-rt_v5-rev0.7z)编译。

### 编译选项

MindSpore根目录下的`build.bat`脚本可用于MindSpore Lite的编译。

#### `build.bat`的编译参数

| 参数  |  参数说明  | 是否必选 |
| -------- | ----- | ---- |
| lite | 设置该参数，则对MindSpore Lite工程进行编译 | 是 |
| [n] | 设定编译时所用的线程数，否则默认设定为6线程  | 否 |

#### `mindspore/lite/CMakeLists.txt`的选项

| 选项  |  参数说明  | 取值范围 | 默认值 |
| -------- | ----- | ---- | ---- |
| MSLITE_ENABLE_SSE | 是否启用SSE指令集 | on、off | off |
| MSLITE_ENABLE_AVX | 是否启用AVX指令集 | on、off | off |
| MSLITE_ENABLE_CONVERTER | 是否编译模型转换工具 | on、off | on |
| MSLITE_ENABLE_TOOLS | 是否编译配套工具 | on、off | on |
| MSLITE_ENABLE_TESTCASES | 是否编译测试用例 | on、off | off |

> - 以上选项可通过设置同名环境变量或者`mindspore/lite/CMakeLists.txt`文件修改。

### 编译示例

首先，使用git工具，从MindSpore代码仓下载源码。

```bat
git clone https://gitee.com/mindspore/mindspore.git -b r1.3
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

最后，会在`output/`目录中生成如下文件：

- `mindspore-lite-{version}-win-x64.zip`：包含模型推理框架runtime和配套工具。

> version：输出件版本号，与所编译的分支代码对应的版本一致。

### 目录结构

```text
mindspore-lite-{version}-win-x64
├── runtime
│   ├── include
│   └── lib
│       ├── libgcc_s_seh-1.dll      # MinGW动态库
│       ├── libmindspore-lite.a     # MindSpore Lite推理框架的静态库
│       ├── libmindspore-lite.dll   # MindSpore Lite推理框架的动态库
│       ├── libmindspore-lite.dll.a # MindSpore Lite推理框架的动态库的链接文件
│       ├── libssp-0.dll            # MinGW动态库
│       ├── libstdc++-6.dll         # MinGW动态库
│       └── libwinpthread-1.dll     # MinGW动态库
└── tools
    ├── benchmark # 基准测试工具
    └── converter # 模型转换工具
```

> 暂不支持在Windows进行端侧训练。

## macOS环境编译

### 环境要求

- 系统环境：macOS 10.15.4及以上；64位。

- 编译依赖
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [Xcode](https://developer.apple.com/xcode/download/cn) == 11.4.1
    - [Git](https://git-scm.com/downloads) >= 2.28.0

> - 编译脚本中会执行`git clone`获取第三方依赖库的代码。

### 编译选项

MindSpore根目录下的`build.sh`脚本可用于MindSpore Lite的编译。

#### `build.sh`的编译参数

| 参数  |  参数说明  | 取值范围 | 默认值 |
| -------- | ----- | ---- | ---- |
| -I | 选择目标架构 | arm64、arm32 | 无 |
| -j[n] | 设定编译时所用的线程数，否则默认设定为8线程 | Integer | 8 |

### 编译示例

首先，在进行编译之前，需从MindSpore代码仓下载源码。

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.3
```

然后，在源码根目录下执行如下命令即可编译MindSpore Lite。

- 编译ARM64架构版本。

    ```bash
    bash build.sh -I arm64 -j8
    ```

- 编译ARM32架构版本。

    ```bash
    bash build.sh -I arm32 -j8
    ```

最后，会在`output/`目录中生成如下文件：

- `mindspore-lite-{version}-{os}-{arch}.tar.gz`：包含模型推理框架runtime。

> - version: 输出件版本号，与所编译的分支代码对应的版本一致。
> - os: 输出件应部署的操作系统。
> - arch: 输出件应部署的系统架构。

### 目录结构

```text
mindspore-lite.framework
└── runtime
    ├── Headers        # 推理框架头文件
    ├── Info.plist     # 配置文件
    └── mindspore-lite # 静态库
```

> 暂不支持在macOS进行端侧训练与转换工具。
