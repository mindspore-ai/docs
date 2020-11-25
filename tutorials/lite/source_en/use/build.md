# Building MindSpore Lite

<!-- TOC -->

- [Building MindSpore Lite](#building-mindspore-lite)
    - [Linux Environment Compilation](#linux-environment-compilation)
        - [Environment Requirements](#environment-requirements)
        - [Compilation Options](#compilation-options)
        - [Compilation Example](#compilation-example)
        - [Output Description](#output-description)
            - [Description of Converter's Directory Structure](#description-of-converters-directory-structure)
            - [Description of Runtime and Other tools' Directory Structure](#description-of-runtime-and-other-tools-directory-structure)
            - [Description of Imageprocess's Directory Structure](#description-of-imageprocesss-directory-structure)
    - [Windows Environment Compilation](#windows-environment-compilation)
        - [Environment Requirements](#environment-requirements-1)
        - [Compilation Options](#compilation-options-1)
        - [Compilation Example](#compilation-example-1)
        - [Output Description](#output-description-1)
            - [Description of Converter's Directory Structure](#description-of-converters-directory-structure-1)
            - [Description of Benchmark Directory Structure](#description-of-benchmark-directory-structure-1)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/lite/source_en/use/build.md" target="_blank"><img src="../_static/logo_source.png"></a>

This chapter introduces how to quickly compile MindSpore Lite, which includes the following modules:

| Module | Support Platform | Description |
| --- | ---- | ---- |
| converter | Linux, Windows | Model Conversion Tool |
| runtime(cpp, java) | Linux, Android | Model Inference Framework(cpp, java) |
| benchmark | Linux, Windows, Android | Benchmarking Tool |
| lib_cropper | Linux | libmindspore-lite.a static library crop tool |
| imageprocess | Linux, Android | Image Processing Library |

## Linux Environment Compilation

### Environment Requirements

- The compilation environment supports Linux x86_64 only. Ubuntu 18.04.02 LTS is recommended.

- Compilation dependencies of runtime(cpp), benchmark:
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

### Compilation Options

MindSpore Lite provides a compilation script `build.sh` for one-click compilation, located in the root directory of MindSpore. This script can be used to compile the code of training and inference. The following describes the compilation options of MindSpore Lite.

| Parameter  |  Parameter Description  | Value Range | Mandatory or No |
| -------- | ----- | ---- | ---- |
| -I | Selects an applicable architecture. This option is required when compile MindSpore Lite. | arm64, arm32, or x86_64 | No |
| -d | If this parameter is set, the debug version is compiled. Otherwise, the release version is compiled. | None | No |
| -i | If this parameter is set, incremental compilation is performed. Otherwise, full compilation is performed. | None | No |
| -j[n] | Sets the number of threads used during compilation. Otherwise, the number of threads is set to 8 by default. | Integer | No |
| -e | In the ARM architecture, select the backend operator and set the `gpu` parameter. The built-in GPU operator of the framework is compiled at the same time. | GPU | No |
| -h | Displays the compilation help information. | None | No |
| -n | Specifies to compile the lightweight image processing module. | lite_cv | No |
| -A | Language used by mindspore lite, default cpp. If the parameter is set to java，the AAR is compiled. | cpp, java | No |
| -C | If this parameter is set, the converter is compiled, default on. | on, off | No |
| -o | If this parameter is set, the benchmark is compiled, default on. | on, off | No |
| -t | If this parameter is set, the testcase is compiled, default off. | on, off | No |

> When the `-I` parameter changes, such as `-I x86_64` is converted to `-I arm64`, adding `-i` for parameter compilation does not take effect.
>
> When compiling the AAR package, the `-A java` parameter must be added, and there is no need to add the `-I` parameter.

### Compilation Example

First, download source code from the MindSpore code repository.

```bash
git clone https://gitee.com/mindspore/mindspore.git
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

- Release version of the ARM 64-bit architecture in the incremental compilation mode, with the built-in GPU operator compiled:

    ```bash
    bash build.sh -I arm64 -e gpu
    ```

- Compile ARM64 with image preprocessing module:

    ```bash
    bash build.sh -I arm64 -n lite_cv
    ```

- Compile MindSpore Lite AAR in the incremental compilation mode:

    ```bash
    bash build.sh -A java -i
    ```

  > Turn on the incremental compilation mode. If the ARM64 or ARM32 runtime already exists in the `mindspore/output/` directory, the corresponding version of the runtime will not be recompiled.

- Release version of the x86_64 architecture, with the converter compiled:

    ```bash
    bash build.sh -I x86_64 -C on
    ```

- Release version of the x86_64 architecture, with the benchmark compiled:

    ```bash
    bash build.sh -I x86_64 -o on
    ```

### Output Description

After the compilation is complete, go to the `mindspore/output` directory of the source code to view the file generated after compilation. The file is divided into the following parts.

- `mindspore-lite-{version}-converter-{os}.tar.gz`: Contains model conversion tool.
- `mindspore-lite-{version}-runtime-{os}-{device}.tar.gz`: Contains model inference framework, benchmarking tool, performance analysis tool and library crop tool.
- `mindspore-lite-{version}-minddata-{os}-{device}.tar.gz`: Contains image processing library ImageProcess.
- `mindspore-lite-maven-{version}.zip`: Contains model reasoning framework AAR package.

> version: Version of the output, consistent with that of the MindSpore.
>
> device: Currently divided into cpu (built-in CPU operator) and gpu (built-in CPU and GPU operator).
>
> os: Operating system on which the output will be deployed.

Execute the decompression command to obtain the compiled output:

```bash
tar -xvf mindspore-lite-{version}-converter-{os}.tar.gz
tar -xvf mindspore-lite-{version}-runtime-{os}-{device}.tar.gz
tar -xvf mindspore-lite-{version}-minddata-{os}-{device}.tar.gz
unzip mindspore-lite-maven-{version}.zip
```

#### Description of Converter's Directory Structure

The conversion tool is only available under the `-I x86_64` compilation option, and the content includes the following parts:

```text
|
├── mindspore-lite-{version}-converter-{os}
│   └── converter # Model conversion Ttool
|       ├── converter_lite # Executable program
│   └── lib # The dynamic link library that converter depends
|       ├── libmindspore_gvar.so # A dynamic library that stores some global variables
│   └── third_party # Header files and libraries of third party libraries
│       ├── glog # Dynamic library of Glog
```

#### Description of Runtime and Other tools' Directory Structure

The inference framework can be obtained under `-I x86_64`, `-I arm64` and `-I arm32` compilation options, and the content includes the following parts:

- When the compilation option is `-I x86_64`:

    ```text
    |
    ├── mindspore-lite-{version}-runtime-x86-cpu
    │   └── benchmark # Benchmarking Tool
    |   └── lib_cropper # Static library crop tool
    │       ├── lib_cropper  # Executable file of static library crop tool
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
    │   └── third_party # Header files and libraries of third party libraries
    │       ├── flatbuffers # Header files of FlatBuffers
    ```

- When the compilation option is `-I arm64`:  

    ```text
    |
    ├── mindspore-lite-{version}-runtime-arm64-cpu
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
    │   └── third_party # Header files and libraries of third party libraries
    │       ├── flatbuffers # Header files of FlatBuffers
    ```

- When the compilation option is `-I arm32`:  

    ```text
    |
    ├── mindspore-lite-{version}-runtime-arm32-cpu
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
    │   └── third_party # Header files and libraries of third party libraries
    │       ├── flatbuffers # Header files of FlatBuffers
    ```

- When the compilation option is `-A java`:

  ```text
  |
  ├── mindspore-lite-maven-{version}
  │   └── mindspore
  │       └── mindspore-lite
  |           └── {version}
  │               ├── mindspore-lite-{version}.aar # MindSpore Lite runtime aar
  ```

> 1. Compile ARM64 to get the inference framework output of arm64-cpu by default, if you add `-e gpu`, you will get the inference framework output of arm64-gpu, and the package name is `mindspore-lite-{version}-runtime-arm64-gpu.tar.gz`, compiling ARM32 is in the same way.
> 2. Before running the tools in the converter, benchmark directory, you need to configure environment variables, and configure the path where the dynamic libraries of MindSpore Lite are located to the path where the system searches for dynamic libraries.

Configure converter:

```bash
export LD_LIBRARY_PATH=./output/mindspore-lite-{version}-converter-ubuntu/lib:./output/mindspore-lite-{version}-converter-ubuntu/third_party/glog/lib:${LD_LIBRARY_PATH}
```

Configure benchmark:

```bash
export LD_LIBRARY_PATH= ./output/mindspore-lite-{version}-runtime-x86-cpu/lib:${LD_LIBRARY_PATH}
```

#### Description of Imageprocess's Directory Structure

The image processing library is only available under the `-I arm64 -n lite_cv` compilation option, and the content includes the following parts:

```text
|
├── mindspore-lite-{version}-runtime-{os}-cpu
│   └── benchmark # Benchmarking Tool
│   └── include # Header files (Image processing files are not involved here, and will not be displayed)
│   └── lib # Inference framework dynamic library
│       ├── libmindspore-lite.a  # Static library of infernece framework in MindSpore Lite
│       ├── libmindspore-lite.so # Dynamic library of infernece framework in MindSpore Lite
│   └── minddata # Image processing dynamic library
│       └── include # Header files
│           └── lite_cv # The Header files of image processing dynamic library
│               ├── image_process.h # The Header files of image processing function
│               ├── lite_mat.h # The Header files of image data class structure
│       └── lib # Image processing dynamic library
│           ├── libminddata-lite.so # The files of image processing dynamic library
│   └── third_party # Third-party Iibrary header files and libraries
│       ├── flatbuffers # The Header files of FlatBuffers
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

- `mindspore-lite-{version}-converter-win-cpu.zip`: Contains model conversion tool.
- `mindspore-lite-{version}-win-runtime-x86-cpu.zip`: Contains benchmarking tool.

> version: Version of the output, consistent with that of the MindSpore.

Execute the decompression command to obtain the compiled output:

```bat
unzip mindspore-lite-{version}-converter-win-cpu.zip
unzip mindspore-lite-{version}-win-runtime-x86-cpu.zip
```

#### Description of Converter's Directory Structure

The content includes the following parts:

```text
|
├── mindspore-lite-{version}-converter-win-cpu
│   └── converter # Model conversion Ttool
|       ├── converter_lite # Executable program
|       ├── libglog.dll # Dynamic library of Glog
|       ├── libmindspore_gvar.dll # A dynamic library that stores some global variables
|       ├── libgcc_s_seh-1.dll # Dynamic library of MinGW
|       ├── libssp-0.dll # Dynamic library of MinGW
|       ├── libstdc++-6.dll # Dynamic library of MinGW
|       ├── libwinpthread-1.dll # Dynamic library of MinGW
```

#### Description of Benchmark's Directory Structure

The content includes the following parts:

```text
|
├── mindspore-lite-{version}-win-runtime-x86-cpu
│   └── benchmark # Benchmarking Tool
│       ├── benchmark.exe # Executable program
│       ├── libmindspore-lite.a  # Static library of infernece framework in MindSpore Lite
│       ├── libmindspore-lite.dll # Dynamic library of infernece framework in MindSpore Lite
│       ├── libmindspore-lite.dll.a # Link file of dynamic library of infernece framework in MindSpore Lite
│       ├── libgcc_s_seh-1.dll # Dynamic library of MinGW
|       ├── libssp-0.dll # Dynamic library of MinGW
|       ├── libstdc++-6.dll # Dynamic library of MinGW
|       ├── libwinpthread-1.dll # Dynamic library of MinGW
│   └── include # Header files of inference framework
│   └── third_party # Header files and libraries of third party libraries
│       ├── flatbuffers # Header files of FlatBuffers
```
