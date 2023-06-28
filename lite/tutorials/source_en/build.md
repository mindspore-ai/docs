# Build

<a href="https://gitee.com/mindspore/docs/blob/r0.7/lite/tutorials/source_en/build.md" target="_blank"><img src="./_static/logo_source.png"></a>

This chapter introduces how to quickly compile MindSpore Lite, which includes the following modules:

| Module | Support Platform | Description |
| --- | ---- | ---- |
| converter | Linux | Model Conversion Tool |
| runtime | Linux、Android | Model Inference Framework |
| benchmark | Linux、Android | Benchmarking Tool |
| time_profiler | Linux、Android | Performance Analysis Tool |

## Linux Environment Compilation

### Environment Requirements

- The compilation environment supports Linux x86_64 only. Ubuntu 18.04.02 LTS is recommended.

- Compilation dependencies of runtime、benchmark and time_profiler:
  - [CMake](https://cmake.org/download/) >= 3.14.1
  - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
  - [Android_NDK r20b](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip)
  - [Git](https://git-scm.com/downloads) >= 2.28.0

- Compilation dependencies of converter:
  - [CMake](https://cmake.org/download/) >= 3.14.1
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
  
> - To install and use `Android_NDK`, you need to configure environment variables. The command example is `export ANDROID_NDK={$NDK_PATH}/android-ndk-r20b`.
> - In the `build.sh` script, run the `git clone` command to obtain the code in the third-party dependency library. Ensure that the network settings of Git are correct.

### Compilation Options

MindSpore Lite provides a compilation script `build.sh` for one-click compilation, located in the root directory of MindSpore. This script can be used to compile the code of training and inference. The following describes the compilation options of MindSpore Lite.

| Parameter  |  Parameter Description  | Value Range | Mandatory or Not |
| -------- | ----- | ---- | ---- |
| **-I** | **Selects an applicable architecture. This option is required when compile MindSpore Lite.** | **arm64, arm32, or x86_64** | **Yes** |
| -d | If this parameter is set, the debug version is compiled. Otherwise, the release version is compiled. | None | No |
| -i | If this parameter is set, incremental compilation is performed. Otherwise, full compilation is performed. | None | No |
| -j[n] | Sets the number of threads used during compilation. Otherwise, the number of threads is set to 8 by default. | Integer | No |
| -e | In the Arm architecture, select the backend operator and set the `gpu` parameter. The built-in GPU operator of the framework is compiled at the same time. | GPU | No |
| -h | Displays the compilation help information. | None | No |

> When the `-I` parameter changes, such as `-I x86_64` is converted to `-I arm64`, adding `-i` for parameter compilation does not take effect.

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

- Release version of the Arm 64-bit architecture in incremental compilation mode, with the number of threads set:
    ```bash
    bash build.sh -I arm64 -i -j32
    ```

- Release version of the Arm 64-bit architecture in incremental compilation mode, with the built-in GPU operator compiled:
    ```bash
    bash build.sh -I arm64 -e gpu
    ```

### Output Description

After the compilation is complete, go to the `mindspore/output` directory of the source code to view the file generated after compilation. The file is divided into two parts.
- `mindspore-lite-{version}-converter-{os}.tar.gz`：Contains model conversion tool.
- `mindspore-lite-{version}-runtime-{os}-{device}.tar.gz`：Contains model inference framework、benchmarking tool and performance analysis tool.

> version: version of the output, consistent with that of the MindSpore.
>
> device: Currently divided into cpu (built-in CPU operator) and gpu (built-in CPU and GPU operator).
>
> os: Operating system on which the output will be deployed.

Execute the decompression command to obtain the compiled output:

```bash
tar -xvf mindspore-lite-{version}-converter-{os}.tar.gz
tar -xvf mindspore-lite-{version}-runtime-{os}-{device}.tar.gz
```
#### Description of Converter's Directory Structure

The conversion tool is only available under the `-I x86_64` compilation option, and the content includes the following parts:

```
|
├── mindspore-lite-{version}-converter-{os} 
│   └── converter # Model conversion Ttool
│   └── third_party # Header files and libraries of third party libraries
│       ├── protobuf # Dynamic library of Protobuf

```

#### Description of Runtime and Other tools' Directory Structure

The inference framework can be obtained under `-I x86_64`, `-I arm64` and `-I arm32` compilation options, and the content includes the following parts:

- When the compilation option is `-I x86_64`:
    ```
    |
    ├── mindspore-lite-{version}-runtime-x86-cpu 
    │   └── benchmark # Benchmarking Tool
    │   └── lib # Inference framework dynamic library
    │       ├── libmindspore-lite.so # Dynamic library of infernece framework in MindSpore Lite
    │   └── third_party # Header files and libraries of third party libraries
    │       ├── flatbuffers # Header files of FlatBuffers
    │   └── include # Header files of inference framework
    │   └── time_profile # Model network layer time-consuming analysis tool
    
    ```
  
- When the compilation option is `-I arm64`:  
    ```
    |
    ├── mindspore-lite-{version}-runtime-arm64-cpu
    │   └── benchmark # Benchmarking Tool
    │   └── lib # Inference framework dynamic library
    │       ├── libmindspore-lite.so # Dynamic library of infernece framework in MindSpore Lite
    │       ├── liboptimize.so # Operator performance optimization library in MindSpore Lite  
    │   └── third_party # Header files and libraries of third party libraries
    │       ├── flatbuffers # Header files of FlatBuffers
    │   └── include # Header files of inference framework
    │   └── time_profile # Model network layer time-consuming analysis tool
      
    ```

- When the compilation option is `-I arm32`:  
    ```
    |
    ├── mindspore-lite-{version}-runtime-arm64-cpu
    │   └── benchmark # Benchmarking Tool
    │   └── lib # Inference framework dynamic library
    │       ├── libmindspore-lite.so # Dynamic library of infernece framework in MindSpore Lite
    │   └── third_party # Header files and libraries of third party libraries
    │       ├── flatbuffers # Header files of FlatBuffers
    │   └── include # Header files of inference framework
    │   └── time_profile # Model network layer time-consuming analysis tool
      
    ```

> 1. `liboptimize.so` only exists in the output package of runtime-arm64 and is only used on ARMv8.2 and CPUs that support fp16.
> 2. Compile ARM64 to get the inference framework output of arm64-cpu by default, if you add `-e gpu`, you will get the inference framework output of arm64-gpu, and the package name is `mindspore-lite-{version}-runtime-arm64-gpu.tar.gz`, compiling ARM32 is in the same way.
> 3. Before running the tools in the converter, benchmark or time_profile directory, you need to configure environment variables, and configure the path where the dynamic libraries of MindSpore Lite and Protobuf are located to the path where the system searches for dynamic libraries. Take the compiled under version 0.7.0-beta as an example: configure converter: `export LD_LIBRARY_PATH=./output/mindspore-lite-0.7.0-converter-ubuntu/third_party/protobuf/lib:${LD_LIBRARY_PATH}`; configure benchmark and timeprofiler: `export LD_LIBRARY_PATH= ./output/mindspore-lite-0.7.0-runtime-x86-cpu/lib:${LD_LIBRARY_PATH}`.
