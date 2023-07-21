# Building MindSpore Lite

`Windows` `Linux` `Android` `Environment Preparation` `Intermediate` `Expert`

[![View Source On Gitee](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/tutorials/lite/source_en/use/build.md)

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
| benchmark       | Linux, Android   | Benchmarking Tool                                |
| cropper         | Linux            | Static library crop tool for libmindspore-lite.a |
| minddata        | Linux, Android   | Image Processing Library                         |
| benchmark_train | Linux, Android   | Performance and Accuracy Validation              |

## Linux Environment Compilation

### Environment Requirements

- The compilation environment supports Linux x86_64 only. Ubuntu 18.04.02 LTS is recommended.

- Compilation dependencies of runtime(cpp):
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
    - [Android_NDK r20b](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip)
    - [Git](https://git-scm.com/downloads) >= 2.28.0

- Compilation dependencies of converter:
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
    - [Android_NDK r20b](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip)
    - [Git](https://git-scm.com/downloads) >= 2.28.0
    - [Autoconf](http://ftp.gnu.org/gnu/autoconf/) >= 2.69
    - [Libtool](https://www.gnu.org/software/libtool/) >= 2.4.6
    - [LibreSSL](http://www.libressl.org/) >= 3.1.3
    - [Automake](https://www.gnu.org/software/automake/) >= 1.11.6
    - [Libevent](https://libevent.org) >= 2.0
    - [M4](https://www.gnu.org/software/m4/m4.html) >= 1.4.18
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
> - In the `build.sh` script, run the `git clone` command to obtain the code in the third-party dependency library. Ensure that the network settings of Git are correct.
> - After Gradle is installed, you need to add its installation path to the PATH: `export PATH=${GRADLE_PATH}/bin:$PATH`.
> - To install the Android SDK via `Android command line tools`, you need to create a new directory first and configure its path to the environment variable in `${ANDROID_SDK_ROOT}`, then create SDK via `sdkmanager`: `./sdkmanager --sdk_root=$ {ANDROID_SDK_ROOT} "cmdline-tools;latest"`, and finally accept the license through `sdkmanager` under the `${ANDROID_SDK_ROOT}` directory: `yes | ./sdkmanager --licenses`.
> - Compiling AAR relies on Android SDK Build-Tools, Android SDK Platform-Tools and other Android SDK related components. If the Android SDK in the environment does not have related components, the required dependencies will be automatically downloaded during compilation.
> - When compiling the NPU operator, you need to download [DDK V500.010](https://developer.huawei.com/consumer/cn/doc/development/hiai-Library/ddk-download-0000001053590180), the directory where the compressed package is decompressed Set to the environment variable `${HWHIAI_DDK}`.

### Compilation Options

MindSpore Lite provides a compilation script `build.sh` for one-click compilation, located in the root directory of MindSpore. This script can be used to compile the code of training and inference. The following describes the compilation options of MindSpore Lite.

| Parameter  |  Parameter Description  | Value Range | Mandatory or No |
| -------- | ----- | ---- | ---- |
| -I | Selects an applicable architecture. This option is required when compile MindSpore Lite. | arm64, arm32, or x86_64 | No |
| -d | If this parameter is set, the debug version is compiled. Otherwise, the release version is compiled. | None | No |
| -i | If this parameter is set, incremental compilation is performed. Otherwise, full compilation is performed. | None | No |
| -j[n] | Sets the number of threads used during compilation. Otherwise, the number of threads is set to 8 by default. | Integer | No |
| -e | In the ARM architecture, select the backend operator. Otherwise, all operator of the framework is compiled at the same time. | cpu, gpu, npu | No |
| -h | Displays the compilation help information. | None | No |
| -n | Specifies to compile the lightweight image processing module. | lite_cv | No |
| -A | Language used by mindspore lite, default cpp. If the parameter is set to java，the AAR is compiled. | cpp, java | No |
| -C | If this parameter is set, the converter is compiled, default on. | on, off | No |
| -o | If this parameter is set, the benchmark and static library crop tool are compiled, default on. | on, off | No |
| -t | If this parameter is set, the testcase is compiled, default off. | on, off | No |
| -T | If this parameter is set, MindSpore Lite training version is compiled, i.e., this option is required when compiling, default off. | on, off | No |

> When the `-I` parameter changes, such as `-I x86_64` is converted to `-I arm64`, adding `-i` for parameter compilation does not take effect.
>
> When compiling the AAR package, the `-A java` parameter must be added, and there is no need to add the `-I` parameter. By default, the built-in CPU and GPU operators are compiled at the same time.
>
> The compiler will only generate training packages when `-T` is opened.
>
> Any `-e` compilation option, the CPU operators will be compiled into it.

### Compilation Example

First, download source code from the MindSpore code repository.

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.1
```

Then, run the following commands in the root directory of the source code to compile MindSpore Lite of different versions:

- Debug version of the x86_64 architecture:

    ```bash
    bash build.sh -I x86_64 -d
    ```

- Release version of the x86_64 architecture, with the number of threads set:

    ```bash
    bash build.sh -I x86_64 -j32
    ```

- Release version of the x86_64 architecture, with the testcase compiled:

    ```bash
    bash build.sh -I x86_64 -t on
    ```

- Release version of the ARM 64-bit architecture in the incremental compilation mode, with the number of threads set:

    ```bash
    bash build.sh -I arm64 -i -j32
    ```

- Release version of the ARM 64-bit architecture, with the built-in CPU operators compiled:

    ```bash
    bash build.sh -I arm64 -e cpu
    ```

- Release version of the ARM 64-bit architecture, with the built-in CPU and GPU operators compiled:

    ```bash
    bash build.sh -I arm64 -e gpu
    ```

- Release version of the ARM 64-bit architecture, with the built-in CPU and NPU operators compiled:

    ```bash
    bash build.sh -I arm64 -e npu
    ```

- Compile ARM64 with image preprocessing module:

    ```bash
    bash build.sh -I arm64 -n lite_cv
    ```

- Compile MindSpore Lite AAR, with the built-in CPU and GPU operators compiled:

    ```bash
    bash build.sh -A java
    ```

- Compile MindSpore Lite AAR, with the built-in CPU operators compiled:

    ```bash
    bash build.sh -A java -e cpu
    ```

- Release version of the x86_64 architecture, with the benchmark, cropper and converter compiled:

    ```bash
    bash build.sh -I x86_64
    ```

- Release version of the x86_64 architecture, with the converter compiled and train on device enabled:

    ```bash
    bash build.sh -I x86_64 -T on
    ```

### Inference Output Description

After the compilation is complete, go to the `mindspore/output` directory of the source code to view the file generated after compilation. The file is divided into the following parts.

- `mindspore-lite-{version}-converter-{os}-{arch}.tar.gz`: Model converter.
- `mindspore-lite-{version}-inference-{os}-{arch}.tar.gz`: Contains model inference framework, benchmarking tool, performance analysis tool and library crop tool.
- `mindspore-lite-maven-{version}.zip`: Contains model reasoning framework AAR package.

> version: Version of the output, consistent with that of the MindSpore.
>
> os: Operating system on which the output will be deployed.
>
> arch: System architecture on which the output will be deployed.

Execute the decompression command to obtain the compiled output:

```bash
tar -xvf mindspore-lite-{version}-converter-{os}-{arch}.tar.gz
tar -xvf mindspore-lite-{version}-inference-{os}-{arch}.tar.gz
unzip mindspore-lite-maven-{version}.zip
```

#### Description of Converter's Directory Structure

The conversion tool is only available under the `-I x86_64` compilation option, and the content includes the following parts:

```text
│
├── mindspore-lite-{version}-converter-{os}-{arch}
│   └── converter # Model conversion Ttool
│       ├── converter_lite # Executable program
│   └── lib # The dynamic link library that converter depends
│       ├── libmindspore_gvar.so # A dynamic library that stores some global variables
│   └── third_party # Header files and libraries of third party libraries
│       ├── glog # Dynamic library of Glog
```

#### Description of Runtime and Other tools' Directory Structure

The inference framework can be obtained under `-I x86_64`, `-I arm64` and `-I arm32` compilation options, and the content includes the following parts:

- When the compilation option is `-I x86_64`:

    ```text
    │
    ├── mindspore-lite-{version}-inference-linux-x64
    │   └── benchmark # Benchmarking Tool
    │   └── cropper # Static library crop tool
    │       ├── cropper  # Executable file of static library crop tool
    │       ├── cropper_mapping_cpu.cfg # Crop cpu library related configuration files
    │   └── include # Header files of inference framework
    │   └── lib # Inference framework library
    │       ├── libmindspore-lite.a  # Static library of infernece framework in MindSpore Lite
    │       ├── libmindspore-lite.so # Dynamic library of infernece framework in MindSpore Lite
    │   └── minddata # Image processing dynamic library
    │       └── include # Header files
    │           └── lite_cv # The Header files of image processing dynamic library
    │               ├── image_process.h # The Header files of image processing function
    │               ├── lite_mat.h # The Header files of image data class structure
    │       └── lib # Image processing dynamic library
    │           ├── libminddata-lite.so # The files of image processing dynamic library
    ```

- When the compilation option is `-I arm64` or `-I arm32`:

    ```text
    │
    ├── mindspore-lite-{version}-inference-android-{arch}
    │   └── benchmark # Benchmarking Tool
    │   └── include # Header files of inference framework
    │   └── lib # Inference framework library
    │       ├── libmindspore-lite.a  # Static library of infernece framework in MindSpore Lite
    │       ├── libmindspore-lite.so # Dynamic library of infernece framework in MindSpore Lite
    │   └── minddata # Image processing dynamic library
    │       └── include # Header files
    │           └── lite_cv # The Header files of image processing dynamic library
    │               ├── image_process.h # The Header files of image processing function
    │               ├── lite_mat.h # The Header files of image data class structure
    │       └── lib # Image processing dynamic library
    │           ├── libminddata-lite.so # The files of image processing dynamic library
    ```

- When the compilation option is `-A java`:

  ```text
  │
  ├── mindspore-lite-maven-{version}
  │   └── mindspore
  │       └── mindspore-lite
  │           └── {version}
  │               ├── mindspore-lite-{version}.aar # MindSpore Lite runtime aar
  ```

> 1. Compile ARM64 to get the inference framework output of cpu/gpu/npu by default, if you add `-e gpu`, you will get the inference framework output of cpu/gpu, ARM32 only supports CPU.
>
> 2. Before running the tools in the converter, benchmark directory, you need to configure environment variables, and configure the path where the dynamic libraries of MindSpore Lite are located to the path where the system searches for dynamic libraries.

Configure converter:

```bash
export LD_LIBRARY_PATH=./output/mindspore-lite-{version}-converter-{os}-{arch}/lib:./output/mindspore-lite-{version}-converter-{os}-{arch}/third_party/glog/lib:${LD_LIBRARY_PATH}
```

Configure benchmark:

```bash
export LD_LIBRARY_PATH= ./output/mindspore-lite-{version}-inference-{os}-{arch}/lib:${LD_LIBRARY_PATH}
```

### Training Output Description

If the `-T on` is added to the MindSpore Lite, go to the `mindspore/output` directory of the source code to view the file generated after compilation. The file is divided into the following parts.

- `mindspore-lite-{version}-train-converter-{os}-{arch}.tar.gz`: Model converter, only support MINDIR format model file.
- `mindspore-lite-{version}-train-{os}-{arch}.tar.gz`: Contains model training framework, performance analysis tool.

> version: Version of the output, consistent with that of the MindSpore.
>
> os: Operating system on which the output will be deployed.
>
> arch: System architecture on which the output will be deployed.

Execute the decompression command to obtain the compiled output:

```bash
tar -xvf mindspore-lite-{version}-train-converter-{os}-{arch}.tar.gz
tar -xvf mindspore-lite-{version}-train-{os}-{arch}.tar.gz
```

#### Description of Training Converter's Directory Structure

The training model conversion tool is only available under the `-I x86_64` compilation option, and the content includes the following parts:

```text
│
├── mindspore-lite-{version}-train-converter-linux-x64
│   └── converter # Model conversion Ttool
│       ├── converter_lite # Executable program
│   └── lib # The dynamic link library that converter depends
│       ├── libmindspore_gvar.so # A dynamic library that stores some global variables
│   └── third_party # Header files and libraries of third party libraries
│       ├── glog # Dynamic library of Glog
```

#### Description of Training Runtime and Other tools' Directory Structure

The MindSpore Lite training framework can be obtained under `-I x86_64`, `-I arm64` and `-I arm32` compilation options, and the content includes the following parts:

- When the compilation option is `-I x86_64`:

    ```text
    │
    ├── mindspore-lite-{version}-train-linux-x64
    │   └── cropper # Static library crop tool
    │       ├── cropper  # Executable file of static library crop tool
    │       ├── cropper_mapping_cpu.cfg # Crop cpu library related configuration files
    │   └── include # Header files of training framework
    │   └── lib # Inference framework library
    │       ├── libmindspore-lite.a  # Static library of training framework in MindSpore Lite
    │       ├── libmindspore-lite.so # Dynamic library of training framework in MindSpore Lite
    │   └── minddata # Image processing dynamic library
    │       └── include # Header files
    │           └── lite_cv # The Header files of image processing dynamic library
    │               ├── image_process.h # The Header files of image processing function
    │               ├── lite_mat.h # The Header files of image data class structure
    │       └── lib # Image processing dynamic library
    │           ├── libminddata-lite.so # The files of image processing dynamic library
    │   └── benchmark_train
    │       ├── benchmark_train # training model benchmark tool
    ```

- When the compilation option is `-I arm64` or `-I arm32`:  

    ```text
    │
    ├── mindspore-lite-{version}-train-android-{arch}
    │   └── include # Header files of training framework
    │   └── lib # Training framework library
    │       ├── libmindspore-lite.a  # Static library of training framework in MindSpore Lite
    │       ├── libmindspore-lite.so # Dynamic library of training framework in MindSpore Lite
    │   └── minddata # Image processing dynamic library
    │       └── include # Header files
    │           └── lite_cv # The Header files of image processing dynamic library
    │               ├── image_process.h # The Header files of image processing function
    │               ├── lite_mat.h # The Header files of image data class structure
    │       └── lib # Image processing dynamic library
    │           ├── libminddata-lite.so # The files of image processing dynamic library
    │   └── benchmark_train
    │       ├── benchmark_train # training model benchmark tool
    ```

> Before running the tools in the converter and the benchmark_train directory, you need to configure environment variables, and configure the path where the dynamic libraries of MindSpore Lite are located to the path where the system searches for dynamic libraries.

Configure converter:

```bash
export LD_LIBRARY_PATH=./output/mindspore-lite-{version}-train-converter-{os}-{arch}/lib:./output/mindspore-lite-{version}-train-converter-{os}-{arch}/third_party/glog/lib:${LD_LIBRARY_PATH}
```

Configure benchmark_train:

```bash
export LD_LIBRARY_PATH= ./output/mindspore-lite-{version}-train-{os}-{arch}/lib:${LD_LIBRARY_PATH}
```

## Windows Environment Compilation

### Environment Requirements

- System environment: Windows 7, Windows 10; 64-bit.

- Compilation dependencies are:
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [MinGW GCC](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/7.3.0/threads-posix/seh/x86_64-7.3.0-release-posix-seh-rt_v5-rev0.7z/download) = 7.3.0

> The compilation script will execute `git clone` to obtain the code of the third-party dependent libraries. Please make sure that the git network settings are correct and available in advance.

### Compilation Options

The compilation options of MindSpore Lite are as follows:

| Parameter  |  Parameter Description   | Mandatory or Not |
| -------- | ----- | ---- |
| lite | Set this parameter to compile the MindSpore Lite project. | Yes |
| [n] | Set the number of threads used during compilation, otherwise the default is set to 6 threads.  | No |

### Compilation Example

First, use the git tool to download the source code from the MindSpore code repository.

```bat
git clone https://gitee.com/mindspore/mindspore.git -b r1.1
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

- `mindspore-lite-{version}-converter-win-x64.zip`: Contains model conversion tool.
- `mindspore-lite-{version}-inference-win-x64.zip`: Contains model inference framework and benchmarking tool.

> version: Version of the output, consistent with that of the MindSpore.

Execute the decompression command to obtain the compiled output:

```bat
unzip mindspore-lite-{version}-converter-win-x64.zip
unzip mindspore-lite-{version}-inference-win-x64.zip
```

#### Description of Converter's Directory Structure

The content includes the following parts:

```text
│
├── mindspore-lite-{version}-converter-win-x64
│   └── converter # Model conversion Ttool
│       ├── converter_lite # Executable program
│       ├── libglog.dll # Dynamic library of Glog
│       ├── libmindspore_gvar.dll # A dynamic library that stores some global variables
│       ├── libgcc_s_seh-1.dll # Dynamic library of MinGW
│       ├── libssp-0.dll # Dynamic library of MinGW
│       ├── libstdc++-6.dll # Dynamic library of MinGW
│       ├── libwinpthread-1.dll # Dynamic library of MinGW
```

#### Description of Benchmark's Directory Structure

The content includes the following parts:

```text
│
├── mindspore-lite-{version}-inference-win-x64
│   └── benchmark # Benchmarking Tool
│       ├── benchmark.exe # Executable program
│       ├── libmindspore-lite.a  # Static library of infernece framework in MindSpore Lite
│       ├── libmindspore-lite.dll # Dynamic library of infernece framework in MindSpore Lite
│       ├── libmindspore-lite.dll.a # Link file of dynamic library of infernece framework in MindSpore Lite
│       ├── libgcc_s_seh-1.dll # Dynamic library of MinGW
│       ├── libssp-0.dll # Dynamic library of MinGW
│       ├── libstdc++-6.dll # Dynamic library of MinGW
│       ├── libwinpthread-1.dll # Dynamic library of MinGW
│   └── include # Header files of inference framework
```

> Currently, MindSpore Lite is not supported on Windows.
