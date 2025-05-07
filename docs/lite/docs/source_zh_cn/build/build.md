# 端侧编译

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/build/build.md)

本章节介绍如何快速编译出MindSpore Lite。

MindSpore Lite包含模块：

| 模块               | 支持平台                | 说明                              |
| ------------------ | ----------------------- | --------------------------------- |
| converter          | Linux、Windows          | 模型转换工具                      |
| runtime(cpp、java) | Linux、Windows、Android、iOS、OpenHarmony(OHOS) | 模型推理框架（Windows平台不支持java版runtime） |
| benchmark          | Linux、Windows、Android、OpenHarmony(OHOS) | 基准测试工具                      |
| benchmark_train    | Linux、Android          | 性能测试和精度校验工具              |
| cropper            | Linux                   | libmindspore-lite.a静态库裁剪工具 |
| minddata           | Linux、Android          | 图像处理库                        |
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
    - [OpenHarmony_NDK](http://ci.openharmony.cn/workbench/cicd/dailybuild/dailylist)
        - 下载OpenHarmony NDK：前往OpenHarmony的每日构建，选择"形态组件"为"ohos-sdk"，任选一个成功构建的包进行下载。下载之后解压压缩包，其中以native开头的文件即为OpenHarmony NDK。
        - 配置环境变量：`export OHOS_NDK=NDK路径`、`export TOOLCHAIN_NAME=ohos`
- Java API模块的编译依赖（可选），未设置JAVA_HOME环境变量则不编译该模块。
    - [Gradle](https://gradle.org/releases/) >= 6.6.1
        - 配置环境变量：`export GRADLE_HOME=GRADLE路径`和`export GRADLE_USER_HOME=GRADLE路径`
        - 将bin目录添加到PATH中：`export PATH=${GRADLE_HOME}/bin:$PATH`
    - [Maven](https://archive.apache.org/dist/maven/maven-3/) >= 3.3.1
        - 配置环境变量：`export MAVEN_HOME=MAVEN路径`
        - 将bin目录添加到PATH中：`export PATH=${MAVEN_HOME}/bin:$PATH`
    - [OpenJDK](https://openjdk.java.net/install/) 1.8 到 1.15
        - 配置环境变量：`export JAVA_HOME=JDK路径`
        - 将bin目录添加到PATH中：`export PATH=${JAVA_HOME}/bin:$PATH`
    - [Android SDK](https://developer.android.com/studio?hl=zh-cn#cmdline-tools)
        - 创建一个新目录，配置环境变量`export ANDROID_SDK_ROOT=新建的目录`
        - 下载`SDK Tools`，解压缩后进入`cmdline-tools/bin`目录，通过`sdkmanager`创建SDK：`./sdkmanager --sdk_root=${ANDROID_SDK_ROOT} "cmdline-tools;latest"`
        - 通过`${ANDROID_SDK_ROOT}`目录下的`sdkmanager`接受许可证：`yes | ./sdkmanager --licenses`
- Python API模块的编译依赖（可选），未安装Python3或者NumPy则不编译该模块。
    - [Python](https://www.python.org/) >= 3.7.0
    - [NumPy](https://numpy.org/) >= 1.17.0 (如果用pip安装失败，请先升级pip版本：`python -m pip install -U pip`)
    - [wheel](https://pypi.org/project/wheel/) >= 0.32.0 (如果用pip安装失败，请先升级pip版本：`python -m pip install -U pip`)

> Gradle建议采用[gradle-6.6.1-complete](https://gradle.org/next-steps/?version=6.6.1&format=all)版本，配置其他版本gradle将会采用gradle wrapper机制自动下载`gradle-6.6.1-complete`。
>
> 也可直接使用已配置好上述依赖的Docker编译镜像。
>
> - 下载镜像：`docker pull swr.cn-south-1.myhuaweicloud.com/mindspore-build/mindspore-lite:ubuntu18.04.2-20210530`
> - 创建容器：`docker run -tid --net=host --name=docker01 swr.cn-south-1.myhuaweicloud.com/mindspore-build/mindspore-lite:ubuntu18.04.2-20210530`
> - 进入容器：`docker exec -ti -u 0 docker01 bash`

### 编译选项

MindSpore根目录下的`build.sh`脚本可用于MindSpore Lite的编译。

#### `build.sh`的参数使用说明

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

#### 模块构建编译选项

模块的构建通过环境变量进行控制，用户可通过声明相关环境变量，控制编译构建的模块。在修改编译选项后，为使选项生效，在使用`build.sh`脚本进行编译时，不可添加`-i`参数进行增量编译。

- 通用模块编译选项

    | 选项  |  参数说明  | 取值范围 | 默认值 |
    | -------- | ----- | ---- | ---- |
    | MSLITE_GPU_BACKEND | 设置GPU后端，在非OpenHarmony系统且`-I arm64`时仅opencl有效，在`-I x86_64`时仅tensorrt有效 | opencl、tensorrt、off | 在`-I arm64`时为opencl， 在`-I x86_64`时为off |
    | MSLITE_ENABLE_NPU | 是否编译NPU算子，仅在非OpenHarmony系统且`-I arm64`或`-I arm32`时有效 | on、off | off |
    | MSLITE_ENABLE_TRAIN | 是否编译训练版本 | on、off | on |
    | MSLITE_ENABLE_SSE | 是否启用SSE指令集，仅在`-I x86_64`时有效 | on、off | off |
    | MSLITE_ENABLE_AVX | 是否启用AVX指令集，仅在`-I x86_64`时有效 | on、off | off |
    | MSLITE_ENABLE_AVX512 | 是否启用AVX512指令集，仅在`-I x86_64`时有效 | on、off | off |
    | MSLITE_ENABLE_CONVERTER | 是否编译模型转换工具，仅在`-I x86_64`时有效 | on、off | on |
    | MSLITE_ENABLE_TOOLS | 是否编译配套工具 | on、off | on |
    | MSLITE_ENABLE_TESTCASES | 是否编译测试用例 | on、off | off |
    | MSLITE_ENABLE_MODEL_ENCRYPTION | 是否支持模型加解密 | on、off | off |
    | MSLITE_ENABLE_MODEL_PRE_INFERENCE | 是否启用模型编译时预推理 | on、off | off |
    | MSLITE_ENABLE_GITEE_MIRROR | 是否使能三方库从码云镜像下载 | on、off | off |

    > - TensorRT 和 NPU 的编译环境配置，参考[专用芯片集成说明](https://www.mindspore.cn/lite/docs/zh-CN/master/advanced/third_party/asic.html)。
    > - 启用AVX指令集时，需要运行环境的CPU同时支持avx特性和fma特性。
    > - 模型转换工具的编译时间较长，若非必要，建议通过`MSLITE_ENABLE_CONVERTER`关闭转换工具编译，以加快编译速度。
    > - 解密所需的OpenSSL加密库crypto支持的版本为1.1.1k，需要用户自行下载编译，相关方法可参考：<https://github.com/openssl/openssl#build-and-install>。此外，还需要将libcrypto.so.1.1文件的路径加入到LD_LIBRARY_PATH中。
    > - 当启用模型编译时预推理时，对于非加密模型，用户调用Build接口时，推理框架会创建一个子进程进行预推理，子进程成功返回之后，主进程会正式执行图编译的流程。
    > - 目前OpenHarmony系统仅支持CPU推理，不支持GPU推理。

- runtime功能裁剪编译选项

    若用户对框架包大小敏感，可通过配置以下选项，对runtime模型推理框架进行功能裁剪，以减少包大小，之后，用户可再通过[裁剪工具](https://www.mindspore.cn/lite/docs/zh-CN/master/tools/cropper_tool.html)进行算子裁剪以进一步减少包大小。

    | 选项  |  参数说明  | 取值范围 | 默认值 |
    | -------- | ----- | ---- | ---- |
    | MSLITE_ENABLE_STRING_KERNEL | 是否支持string数据推理模型，如smart_reply.tflite模型 | on、off | on |
    | MSLITE_ENABLE_CONTROLFLOW | 是否支持控制流模型 | on、off | on |
    | MSLITE_ENABLE_WEIGHT_DECODE | 是否支持权重量化模型推理 | on、off | on |
    | MSLITE_ENABLE_CUSTOM_KERNEL | 是否支持南向算子注册 | on、off | on |
    | MSLITE_ENABLE_DELEGATE | 是否支持Delegate机制 | on、off | on |
    | MSLITE_ENABLE_FP16 | 是否支持FP16算子 | on、off | 在`-I x86_64`时off，在`-I arm64`时on，在`-I arm32`时，若Android_NDK版本大于r21e，则on，否则off |
    | MSLITE_ENABLE_INT8 | 是否支持INT8算子 | on、off | on |

    > - 由于NPU和TensorRT的实现依赖于Delegate机制，所以在使用NPU或TensorRT时无法关闭Delegate机制，如果关闭了Delegate机制，则相关功能也必须关闭。

### 编译示例

首先，在进行编译之前，需从MindSpore代码仓下载源码。

```bash
git clone https://gitee.com/mindspore/mindspore.git
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

- 编译OpenHarmony系统的aarch32或aarch64的包：

   编译aarch32

    ```bash
    export OHOS_NDK=NDK路径
    export TOOLCHAIN_NAME=ohos
    bash build.sh -I arm32 -j32
    ```

    编译aarch64

    ```bash
    export OHOS_NDK=NDK路径
    export TOOLCHAIN_NAME=ohos
    bash build.sh -I arm64 -j32
    ```

最后，会在`output/`目录中生成如下文件：

- `mindspore-lite-{version}-{os}-{arch}.tar.gz`：包含runtime和配套工具。

- `mindspore-lite-maven-{version}.zip`：包含runtime(java)的AAR包。

- `mindspore-lite-{version}-{python}-{os}-{arch}.whl`：包含runtime(Python)的Whl包。

> - version: 输出件版本号，与所编译的分支代码对应的版本一致。
> - python: 输出件Python版本, 如：Python3.7为`cp37-cp37m`。
> - os: 输出件应部署的操作系统。
> - arch: 输出件应部署的系统架构。

若要体验python接口，需要移动到`output/`目录下，使用以下命令进行安装Whl安装包。

```bash
pip install `mindspore-lite-{version}-{python}-{os}-{arch}.whl`
```

安装后可以使用以下命令检查是否安装成功：若无报错，则表示安装成功。

```bash
python -c "import mindspore_lite"
```

安装成功后，可使用`pip show mindspore_lite`命令查看MindSpore Lite的Python模块的安装位置。

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
    │   │   ├── libmsdeobfuscator-lite.so  # 混淆模型加载动态库文件，需开启`MSLITE_ENABLE_MODEL_OBF`选项。
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
    │   │   └── libmsdeobfuscator-lite.so  # 混淆模型加载动态库文件，需开启`MSLITE_ENABLE_MODEL_OBF`选项。
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

- 当编译选项为`-I arm64`或`-I arm32`，并且指定`TOOLCHAIN_NAME=ohos`时：

    ```text
    mindspore-lite-{version}-ohos-{arch}
    ├── runtime
    │   ├── include
    │   └── lib
    │       ├── libmindspore-lite.a  # MindSpore Lite推理框架的静态库
    │       └── libmindspore-lite.so # MindSpore Lite推理框架的动态库
    └── tools
        └── benchmark                # 基准测试工具
    ```

## Windows环境编译

### 环境要求

- 系统环境：Windows 7，Windows 10；64位。

- MinGW 编译依赖
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - 编译64位：[MinGW-W64 x86_64](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/7.3.0/threads-posix/seh/x86_64-7.3.0-release-posix-seh-rt_v5-rev0.7z) = GCC-7.3.0
    - 编译32位：[MinGW-W64 i686](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win32/Personal%20Builds/mingw-builds/7.3.0/threads-posix/dwarf/i686-7.3.0-release-posix-dwarf-rt_v5-rev0.7z) = GCC-7.3.0

- Visual Studio 编译依赖
    - [Visual Studio](https://visualstudio.microsoft.com/zh-hans/vs/older-downloads/) = 2017，已自带cmake。
    - 编译64位：进入开始菜单，点击“适用于 VS 2017 的 x64 本机工具命令提示”，或者打开cmd窗口，执行`call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Profession\VC\Auxiliary\Build\vcvars64.bat"`。
    - 编译32位：进入开始菜单，点击“VS 2017的 x64_x86 交叉工具命令提示符”，或者打开cmd窗口，执行`call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Profession\VC\Auxiliary\Build\vcvarsamd64_x86.bat"`。

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
| MSLITE_ENABLE_AVX | 是否启用AVX指令集（该选项暂不支持Visual Studio编译器） | on、off | off |
| MSLITE_ENABLE_AVX512 | 是否启用AVX512指令集（该选项暂不支持Visual Studio编译器） | on、off | off |
| MSLITE_ENABLE_CONVERTER | 是否编译模型转换工具（该选项暂不支持Visual Studio编译器） | on、off | on |
| MSLITE_ENABLE_TOOLS | 是否编译配套工具 | on、off | on |
| MSLITE_ENABLE_TESTCASES | 是否编译测试用例 | on、off | off |
| MSLITE_ENABLE_GITEE_MIRROR | 是否使能三方库从码云镜像下载 | on、off | off |

> - 以上选项可通过设置同名环境变量或者`mindspore/lite/CMakeLists.txt`文件修改。
> - 模型转换工具的编译时间较长，若非必要，建议通过`MSLITE_ENABLE_CONVERTER`关闭转换工具编译，以加快编译速度。

### 编译示例

首先，使用git工具，从MindSpore代码仓下载源码。

```bat
git clone https://gitee.com/mindspore/mindspore.git
```

然后，使用cmd工具在源码根目录下，执行如下命令即可编译MindSpore Lite。

- 打开SSE指令集优化，以8线程编译。

```bat
set MSLITE_ENABLE_SSE=on
call build.bat lite 8
```

最后，会在`output/`目录中生成如下文件：

- `mindspore-lite-{version}-win-x64.zip`：包含模型推理框架runtime和配套工具。

> version：输出件版本号，与所编译的分支代码对应的版本一致。

### 目录结构

- 当编译器为 MinGW 时：

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

- 当编译器为 Visual Studio 时：

    ```text
    mindspore-lite-{version}-win-x64
    ├── runtime
    │   ├── include
    │   └── lib
    │       ├── libmindspore-lite.dll     # MindSpore Lite推理框架的动态库
    │       ├── libmindspore-lite.dll.lib # MindSpore Lite推理框架的动态库的导入库
    │       └── libmindspore-lite.lib     # MindSpore Lite推理框架的静态库
    └── tools
        └── benchmark # 基准测试工具
    ```

> - 链接 MinGW 编译出的静态库时，需要在链接选项中，加`-Wl,--whole-archive mindspore-lite -Wl,--no-whole-archive`。
> - 链接 Visual Studio 编译出的静态库时，需要在“属性->链接器->命令行->其他选项”中，加`/WHOLEARCHIVE:libmindspore-lite.lib`。
> - 使用 Visual Studio 编译器时，读入 model 流必须加 std::ios::binary，否则会出现读取模型文件不完整的问题。
> - 暂不支持在 Windows 进行端侧训练。

## macOS环境编译

### 环境要求

- 系统环境：macOS 10.15.4及以上；64位。

- 编译依赖
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [Xcode](https://developer.apple.com/xcode/) == 11.4.1
    - [Git](https://git-scm.com/downloads) >= 2.28.0

> - 编译脚本中会执行`git clone`获取第三方依赖库的代码。

### 编译选项

MindSpore根目录下的`build.sh`脚本可用于MindSpore Lite的编译。

#### `build.sh`的编译参数

| 参数  |  参数说明  | 取值范围 | 默认值 |
| -------- | ----- | ---- | ---- |
| -I | 选择目标架构 | arm64、arm32 | 无 |
| -j[n] | 设定编译时所用的线程数，否则默认设定为8线程 | Integer | 8 |

#### `mindspore/lite/CMakeLists.txt`的选项

| 选项                   | 参数说明           | 取值范围 | 默认值 |
|----------------------|----------------| ---- | ---- |
| MSLITE_ENABLE_COREML | 是否启用CoreML后端推理 | on、off | off |

### 编译示例

首先，在进行编译之前，需从MindSpore代码仓下载源码。

```bash
git clone https://gitee.com/mindspore/mindspore.git
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

> 暂不支持在macOS构建端侧训练与模型转换。
