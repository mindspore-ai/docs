# Building MindSpore Lite

`Windows` `Linux` `Android` `Environment Preparation` `Intermediate` `Expert`

<!-- TOC -->

- [Building MindSpore Lite](#building-mindspore-lite)
    - [Linux Environment Compilation](#linux-environment-compilation)
        - [Environment Requirements](#environment-requirements)
        - [Compilation Options](#compilation-options)
        - [Compilation Example](#compilation-example)
        - [Inference Output Description](#inference-output-description)
            - [Description of Converter's Directory Structure](#description-of-converters-directory-structure)
            - [Description of Obfuscator's Directory Structure](#description-of-obfuscators-directory-structure)
            - [Description of Runtime and Other tools' Directory Structure](#description-of-runtime-and-other-tools-directory-structure)
        - [Training Output Description](#training-output-description)
            - [Description of Training Runtime and Related Tools' Directory Structure](#description-of-training-runtime-and-related-tools-directory-structure)
    - [Windows Environment Compilation](#windows-environment-compilation)
        - [Environment Requirements](#environment-requirements-1)
        - [Compilation Options](#compilation-options-1)
        - [Compilation Example](#compilation-example-1)
        - [Output Description](#output-description)
            - [Description of Runtime and Related Tools' Directory Structure](#description-of-runtime-and-related-tools-directory-structure)
    - [Docker Environment Compilation](#docker-environment-compilation)
        - [Environmental Preparation](#environmental-preparation)
            - [Download the docker image](#download-the-docker-image)
            - [Create a container](#create-a-container)
            - [Enter the container](#enter-the-container)
        - [Compilation Options](#compilation-options-2)
        - [Compilation Example](#compilation-example-2)
        - [Output Description](#output-description-1)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/lite/source_en/use/build.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

This chapter introduces how to quickly compile MindSpore Lite, which includes the following modules:

Modules in inference version:

| Module | Support Platform | Description |
| --- | ---- | ---- |
| converter | Linux, Windows | Model Conversion Tool |
| runtime(cpp, java) | Linux, Windows, Android | Model Inference Framework(Windows platform does not support java version runtime) |
| benchmark | Linux, Windows, Android | Benchmarking Tool |
| cropper | Linux | Static library crop tool for libmindspore-lite.a |
| minddata | Linux, Android | Image Processing Library |

Modules in training version:

| Module          | Support Platform | Description                                      |
| --------------- | ---------------- | ------------------------------------------------ |
| converter       | Linux            | Model Conversion Tool                            |
| runtime(cpp)    | Linux, Android   | Model Train Framework(java is not support)       |
| cropper         | Linux            | Static library crop tool for libmindspore-lite.a |
| minddata        | Linux, Android   | Image Processing Library                         |
| benchmark_train | Linux, Android   | Performance and Accuracy Validation              |
| obfuscator      | Linux            | Model Obfuscation Tool                           |

## Linux Environment Compilation

### Environment Requirements

- The compilation environment supports Linux x86_64 only. Ubuntu 18.04.02 LTS is recommended.

- Compilation dependencies of runtime(cpp):
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
    - [Android_NDK](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip) >= r20
    - [Git](https://git-scm.com/downloads) >= 2.28.0
- Compilation dependencies of converter:
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
    - [Android_NDK](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip) >= r20
    - [Git](https://git-scm.com/downloads) >= 2.28.0
    - [Autoconf](http://ftp.gnu.org/gnu/autoconf/) >= 2.69
    - [Libtool](https://www.gnu.org/software/libtool/) >= 2.4.6
    - [LibreSSL](http://www.libressl.org/) >= 3.1.3
    - [Automake](https://www.gnu.org/software/automake/) >= 1.11.6
    - [Libevent](https://libevent.org) >= 2.0
    - [OpenSSL](https://www.openssl.org/) >= 1.1.1

- Compilation dependencies of runtime(java)
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
    - [Android_NDK](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip) >= r20
    - [Git](https://git-scm.com/downloads) >= 2.28.0
    - [Android SDK](https://developer.android.com/studio?hl=zh-cn#cmdline-tools)
    - [Gradle](https://gradle.org/releases/) >= 6.6.1
    - [OpenJDK](https://openjdk.java.net/install/) >= 1.8

> - To install and use `Android_NDK`, you need to configure environment variables. The command example is `export ANDROID_NDK=${NDK_PATH}/android-ndk-r20b`.
> - After Gradle is installed, you need to add its installation path to the PATH: `export PATH=${GRADLE_PATH}/bin:$PATH`.
> - To install the Android SDK via `Android command line tools`, you need to create a new directory first and configure its path to the environment variable in `${ANDROID_SDK_ROOT}`, then create SDK via `sdkmanager`: `./sdkmanager --sdk_root=$ {ANDROID_SDK_ROOT} "cmdline-tools;latest"`, and finally accept the license through `sdkmanager` under the `${ANDROID_SDK_ROOT}` directory: `yes | ./sdkmanager --licenses`.
> - Compiling AAR relies on Android SDK Build-Tools, Android SDK Platform-Tools and other Android SDK related components. If the Android SDK in the environment does not have related components, the required dependencies will be automatically downloaded during compilation.
> - When compiling the NPU operator, you need to download [DDK V500.010](https://developer.huawei.com/consumer/cn/doc/development/hiai-Library/ddk-download-0000001053590180), the directory where the compressed package is decompressed Set to the environment variable `${HWHIAI_DDK}`.

### Compilation Options

MindSpore Lite provides a compilation script `build.sh` for one-click compilation, located in the root directory of MindSpore. This script can be used to compile the code of training and inference.

The following describes the compilation parameter of `build.sh` and the options of `mindspore/lite/CMakeLists.txt`.

#### The compilation parameter of `build.sh`

| Parameter  |  Parameter Description  | Value Range | Defaults |
| -------- | ----- | ---- | ---- |
| -I | Selects an applicable architecture. | arm64, arm32, or x86_64 | None |
| -A | Compile AAR package (including arm32 and arm64). | on, off | off |
| -d | If this parameter is set, the debug version is compiled. Otherwise, the release version is compiled. | None | None |
| -i | If this parameter is set, incremental compilation is performed. Otherwise, full compilation is performed. | None | None |
| -j[n] | Sets the number of threads used during compilation. Otherwise, the number of threads is set to 8 by default. | Integer | 8 |
| -a | Whether to enable AddressSanitizer | on、off | off |

> - When compiling the x86_64 version, if the JAVA_HOME environment variable is configured and Gradle is installed, the JAR package will be compiled at the same time.
> - When the `-I` parameter changes, such as `-I x86_64` is converted to `-I arm64`, adding `-i` for parameter compilation does not take effect.
> - When compiling the AAR package, the `-A on` parameter must be added, and there is no need to add the `-I` parameter.

#### The options of `mindspore/lite/CMakeLists.txt`

| Option  |  Parameter Description  | Value Range | Defaults |
| -------- | ----- | ---- | ---- |
| MSLITE_GPU_BACKEND | Set the GPU backend, only valid when `-I arm64` | opencl, vulkan, cuda, off | opencl |
| MSLITE_ENABLE_NPU | Whether to compile NPU operator, only valid when `-I arm64` or `-I arm32` | on、off | on |
| MSLITE_ENABLE_TRAIN | Whether to compile the training version | on、off | on |
| MSLITE_ENABLE_SSE | Whether to enable SSE instruction set, only valid when `-I x86_64` | on、off | off |
| MSLITE_ENABLE_AVX | Whether to enable AVX instruction set, only valid when `-I x86_64` | on、off | off |
| MSLITE_ENABLE_CONVERTER | Whether to compile the model conversion tool, only valid when `-I x86_64` | on、off | on |
| MSLITE_ENABLE_TOOLS | Whether to compile supporting tools | on、off | on |
| MSLITE_ENABLE_TESTCASES | Whether to compile test cases | on、off | off |

> - The above options can be modified by setting the environment variable with the same name or the file `mindspore/lite/CMakeLists.txt`.
> - After modifying the Option, adding the `-i` parameter for incremental compilation will not take effect.

### Compilation Example

First, download source code from the MindSpore code repository.

```bash
git clone https://gitee.com/mindspore/mindspore.git
```

Then, run the following commands in the root directory of the source code to compile MindSpore Lite of different versions:

- Compile the x86_64 architecture version and set the number of threads at the same time.

    ```bash
    bash build.sh -I x86_64 -j32
    ```

- Compile the ARM64 architecture version without compiling the NPU operator.

    ```bash
    export MSLITE_ENABLE_NPU=off
    bash build.sh -I arm64 -j32
    ```

    Or modify `mindspore/lite/CMakeLists.txt` to set MSLITE_ENABLE_NPU to off and execute the command:

    ```bash
    bash build.sh -I arm64 -j32
    ```

- Compile the AAR package containing aarch64 and aarch32.

    ```bash
    bash build.sh -A on -j32
    ```

### Inference Output Description

After the compilation is complete, go to the `mindspore/output` directory of the source code to view the file generated after compilation. The file is divided into the following parts.

- `mindspore-lite-{version}-{os}-{arch}.tar.gz`: Contains model inference framework runtime (cpp), and related tools.
- `mindspore-lite-maven-{version}.zip`: Contains model reasoning framework AAR package.

> - version: Version of the output, consistent with that of the MindSpore.
> - os: Operating system on which the output will be deployed.
> - arch: System architecture on which the output will be deployed.

Execute the decompression command to obtain the compiled output:

```bash
tar -xvf mindspore-lite-{version}-{os}-{arch}.tar.gz
unzip mindspore-lite-maven-{version}.zip
```

#### Description of Converter's Directory Structure

The conversion tool is only available under the `-I x86_64` compilation option, and the content includes the following parts:

```text
mindspore-lite-{version}-linux-x64
└── tools
    └── converter
        ├── include
        │   └── registry             # Header files of customized op, parser and pass registration
        ├── converter                # Model conversion tool
        │   └── converter_lite       # Executable program
        └── lib                      # The dynamic link library that converter depends
            ├── libglog.so.0         # Dynamic library of Glog
            └── libmslite_converter_plugin.so  # Dynamic library of plugin registry
```

#### Description of CodeGen's Directory Structure

The codegen executable program is only available under the `-I x86_64` compilation option, and only the operator library required by the inference code generated by codegen is generated under the `-I arm64` and `-I arm32` compilation options.

- When the compilation option is `-I x86_64`:

    ```text
    mindspore-lite-{version}-linux-x64
    └── tools
        └── codegen # Code generation tool
            ├── codegen          # Executable program
            ├── include          # Header files of inference framework
            │   ├── nnacl        # nnacl operator header file
            │   └── wrapper
            ├── lib
            │   └── libwrapper.a # MindSpore Lite CodeGen generates code dependent operator static library
            └── third_party
                ├── include
                │   └── CMSIS    # ARM CMSIS NN operator header files
                └── lib
                    └── libcmsis_nn.a # ARM CMSIS NN operator static library
    ```

- When the compilation option is `-I arm64` or `-I arm32`:

    ```text
    mindspore-lite-{version}-android-{arch}
    └── tools
        └── codegen # Code generation tool
            └── operator_library # Operator library
                ├── include   # Header files of inference framework
                │   ├── nnacl # nnacl operator header file
                │   └── wrapper
                └── lib       # Inference framework library
                    └── libwrapper.a # MindSpore Lite CodeGen generates code dependent static library
    ```

#### Description of Obfuscator's Directory Structure

The obfuscation tool is only available under the `-I x86_64` compilation option and the `ENABLE_MODEL_OBF` compilation option in `mindspore/mindspore/lite/CMakeLists.txt` is turned on, the content includes the following parts:

```text
mindspore-lite-{version}-linux-x64
└── tools
    └── obfuscator # Model obfuscation tool
        └── msobfuscator          # Executable program
```

#### Description of Runtime and Other tools' Directory Structure

The inference framework can be obtained under `-I x86_64`, `-I arm64` and `-I arm32` compilation options, and the content includes the following parts:

- When the compilation option is `-I x86_64`:

    ```text
    mindspore-lite-{version}-linux-x64
    ├── runtime
    │   ├── include  # Header files of inference framework
    │   │   └── registry # Header files of customized op registration
    │   └── lib      # Inference framework library
    │       ├── libminddata-lite.so        # The files of image processing dynamic library
    │       ├── libmindspore-lite.a        # Static library of inference framework in MindSpore Lite
    │       ├── libmindspore-lite-jni.so   # Dynamic library of inference framework jni in MindSpore Lite
    │       ├── libmindspore-lite.so       # Dynamic library of inference framework in MindSpore Lite
    │       ├── libmsdeobfuscator-lite.so  # The files of obfuscated model loading dynamic library, need to open the `ENABLE_MODEL_OBF` option.
    │       └── mindspore-lite-java.jar    # Jar of inference framework in MindSpore Lite
    └── tools
        ├── benchmark # Benchmarking tool
        │   └── benchmark # Executable program
        ├── codegen   # Code generation tool
        │   ├── codegen   # Executable program
        │   ├── include   # operator header file
        │   ├── lib       # operator static library
        │   └── third_party # ARM CMSIS NN static library
        ├── converter  # Model conversion tool
        ├── obfuscator # Model obfuscation tool
        └── cropper    # Static library crop tool
            ├── cropper                 # Executable file of static library crop tool
            └── cropper_mapping_cpu.cfg # Crop cpu library related configuration files
    ```

- When the compilation option is `-I arm64` or `-I arm32`:

    ```text
    mindspore-lite-{version}-android-{arch}
    ├── runtime
    │   ├── include     # Header files of inference framework
    │   │   └── registry # Header files of customized op registration
    │   ├── lib         # Inference framework library
    │   │   ├── libminddata-lite.so       # The files of image processing dynamic library
    │   │   ├── libmindspore-lite.a       # Static library of inference framework in MindSpore Lite
    │   │   ├── libmindspore-lite.so      # Dynamic library of inference framework in MindSpore Lite
    │   │   └── libmsdeobfuscator-lite.so # The files of obfuscated model loading dynamic library, need to open the `ENABLE_MODEL_OBF` option.
    │   └── third_party
    │       └── hiai_ddk # NPU library, only exists in arm64 package
    └── tools
        ├── benchmark # Benchmarking tool
        │   └── benchmark
        └── codegen   # Code generation tool
            ├── include  # operator header file
            └── lib      # operator static library
    ```

- When the compilation option is `-A on`:

    ```text
    mindspore-lite-maven-{version}
    └── mindspore
        └── mindspore-lite
            └── {version}
                └── mindspore-lite-{version}.aar # MindSpore Lite runtime aar
    ```

### Training Output Description

If the MSLITE_ENABLE_TRAIN option is turned on, default on, the training Runtime and related tools will be generated, as follows:

- `mindspore-lite-{version}-{os}-{arch}.tar.gz`: Contains model training framework and related tool.

> - version: Version of the output, consistent with that of the MindSpore.
> - os: Operating system on which the output will be deployed.
> - arch: System architecture on which the output will be deployed.

Execute the decompression command to obtain the compiled output:

```bash
tar -xvf mindspore-lite-{version}-{os}-{arch}.tar.gz
```

#### Description of Training Runtime and Related Tools' Directory Structure

The MindSpore Lite training framework can be obtained under `-I x86_64`, `-I arm64` and `-I arm32` compilation options, and the content includes the following parts:

- When the compilation option is `-I x86_64`:

    ```text
    mindspore-lite-{version}-linux-x64
    ├── tools
    │   ├── benchmark_train # Training model benchmark tool
    │   ├── converter       # Model conversion tool
    │   └── cropper         # Static library crop tool
    │       ├── cropper                 # Executable file of static library crop tool
    │       └── cropper_mapping_cpu.cfg # Crop cpu library related configuration files
    └── runtime
        ├── include  # Header files of training framework
        │   └── registry # Header files of customized op registration
        ├── lib      # Inference framework library
        │   ├── libminddata-lite.so        # The files of image processing dynamic library
        │   ├── libmindspore-lite-jni.so   # Dynamic library of training framework jni in MindSpore Lite
        │   ├── libmindspore-lite-train.a  # Static library of training framework in MindSpore Lite
        │   ├── libmindspore-lite-train.so # Dynamic library of training framework in MindSpore Lite
        │   └── mindspore-lite-java.jar    # Jar of inference framework in MindSpore Lite
        └── third_party
            └── libjpeg-turbo
    ```

- When the compilation option is `-I arm64` or `-I arm32`:  

    ```text
    mindspore-lite-{version}-android-{arch}
    ├── tools
    │   ├── benchmark       # Benchmarking tool
    │   ├── benchmark_train # Training model benchmark tool
    └── runtime
        ├── include # Header files of training framework
        │   └── registry # Header files of customized op registration
        ├── lib     # Training framework library
        │   ├── libminddata-lite.so # The files of image processing dynamic library
        │   ├── libmindspore-lite-train.a  # Static library of training framework in MindSpore Lite
        │   └── libmindspore-lite-train.so # Dynamic library of training framework in MindSpore Lite
        └── third_party
            ├── hiai_ddk  # NPU library, only exists in arm64 package
            └── libjpeg-turbo
    ```

## Windows Environment Compilation

### Environment Requirements

- System environment: Windows 7, Windows 10; 64-bit.

- Compilation dependencies are:
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [MinGW GCC](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/7.3.0/threads-posix/seh/x86_64-7.3.0-release-posix-seh-rt_v5-rev0.7z/download) = 7.3.0

> - The compilation script will execute `git clone` to obtain the code of the third-party dependent libraries.
> - If you want to compile 32-bit Mindspore Lite, please use 32-bit [MinGW](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win32/Personal%20Builds/mingw-builds/7.3.0/threads-posix/dwarf/i686-7.3.0-release-posix-dwarf-rt_v5-rev0.7z) to compile.

### Compilation Options

The compilation options of MindSpore Lite are as follows:

| Parameter  |  Parameter Description   | Mandatory or Not |
| -------- | ----- | ---- |
| lite | Set this parameter to compile the MindSpore Lite project. | Yes |
| [n] | Set the number of threads used during compilation, otherwise the default is set to 6 threads.  | No |

### Compilation Example

First, use the git tool to download the source code from the MindSpore code repository.

```bat
git clone https://gitee.com/mindspore/mindspore.git
```

Then, use the cmd tool to compile MindSpore Lite in the root directory of the source code and execute the following commands.

- Compile the Windows version with the default number of threads (6 threads).

```bat
call build.bat lite
```

- Compile the Windows version with the specified number of 8 threads.

```bat
call build.bat lite 8
```

### Output Description

After the compilation is complete, go to the `mindspore/output` directory of the source code to view the file generated after compilation. The file is divided into the following parts.

- `mindspore-lite-{version}-win-x64.zip`: Contains model inference framework and related tool.

> version: Version of the output, consistent with that of the MindSpore.

Execute the decompression command to obtain the compiled output:

```bat
unzip mindspore-lite-{version}-win-x64.zip
```

#### Description of Runtime and Related Tools' Directory Structure

The content includes the following parts:

```text
mindspore-lite-{version}-win-x64
├── runtime
│   ├── include # Header files of inference framework
│   │   └── registry     # Header files of customized op registration
│   └── lib
│       ├── libgcc_s_seh-1.dll      # Dynamic library of MinGW
│       ├── libmindspore-lite.a     # Static library of inference framework in MindSpore Lite
│       ├── libmindspore-lite.dll   # Dynamic library of inference framework in MindSpore Lite
│       ├── libmindspore-lite.dll.a # Link file of dynamic library of inference framework in MindSpore Lite
│       ├── libssp-0.dll            # Dynamic library of MinGW
│       ├── libstdc++-6.dll         # Dynamic library of MinGW
│       └── libwinpthread-1.dll     # Dynamic library of MinGW
└── tools
    ├── benchmark # Benchmarking tool
    │   └── benchmark.exe # Executable program
    └── converter # Model conversion tool
        ├── include
        │   └── registry             # Header files of customized op, parser and pass registration
        ├── converter
        │   └── converter_lite.exe    # Executable program
        └── lib
            ├── libgcc_s_seh-1.dll    # Dynamic library of MinGW
            ├── libglog.dll           # Dynamic library of Glog
            ├── libmslite_converter_plugin.dll   # Dynamic library of plugin registry
            ├── libmslite_converter_plugin.dll.a # Link file of Dynamic library of plugin registry
            ├── libssp-0.dll          # Dynamic library of MinGW
            ├── libstdc++-6.dll       # Dynamic library of MinGW
            └── libwinpthread-1.dll   # Dynamic library of MinGW
```

> Currently, MindSpore Lite is not supported on Windows.

## Docker Environment Compilation

### Environmental Preparation

#### Download the docker image

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore-build/mindspore-lite:ubuntu18.04.2-20210530
```

> - Before downloading the image, please make sure docker has been installed.
> - Docker image does not currently support Windows version compilation.
> - Third-party libraries that compile dependencies have been installed in the image and environment variables have been configured.

#### Create a container

```bash
docker run -tid --net=host --name=docker01 swr.cn-south-1.myhuaweicloud.com/mindspore-build/mindspore-lite:ubuntu18.04.2-20210530
```

#### Enter the container

```bash
docker exec -ti -u 0 docker01 bash
```

### Compilation Options

Refer to [Linux Environment Compilation](https://www.mindspore.cn/tutorial/lite/en/master/use/build.html#linux-environment-compilation)

### Compilation Example

Refer to [Linux Environment Compilation](https://www.mindspore.cn/tutorial/lite/en/master/use/build.html#linux-environment-compilation)

### Output Description

Refer to [Linux Environment Compilation](https://www.mindspore.cn/tutorial/lite/en/master/use/build.html#linux-environment-compilation)
