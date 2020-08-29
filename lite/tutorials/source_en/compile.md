# Compile

<!-- TOC -->

- [compilation](#compilation)
    - [Environment Requirements](#environment-requirements)
    - [Compilation Options](#compilation-options)
    - [Output Description](#output-description)
    - [Compilation Example](#compilation-example)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/lite/tutorials/source_en/compile.md" target="_blank"><img src="./_static/logo_source.png"></a>

This document describes how to quickly install MindSpore Lite on the Ubuntu system.

## Environment Requirements

- The compilation environment supports Linux x86_64 only. Ubuntu 18.04.02 LTS is recommended.

- Compilation dependencies (basics):
  - [CMake](https://cmake.org/download/) >= 3.14.1
  - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
  - [Android_NDK r20b](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip)

  > - `Android_NDK` needs to be installed only when the Arm version is compiled. Skip this dependency when the x86_64 version is compiled.
  > - To install and use `Android_NDK`, you need to configure environment variables. The command example is `export ANDROID_NDK={$NDK_PATH}/android-ndk-r20b`.

- Compilation dependencies (additional dependencies required by the MindSpore Lite model conversion tool, which is required only for compilation of the x86_64 version)
  - [Autoconf](http://ftp.gnu.org/gnu/autoconf/) >= 2.69
  - [Libtool](https://www.gnu.org/software/libtool/) >= 2.4.6
  - [LibreSSL](http://www.libressl.org/) >= 3.1.3
  - [Automake](https://www.gnu.org/software/automake/) >= 1.11.6
  - [Libevent](https://libevent.org) >= 2.0
  - [M4](https://www.gnu.org/software/m4/m4.html) >= 1.4.18
  - [OpenSSL](https://www.openssl.org/) >= 1.1.1


## Compilation Options

MindSpore Lite provides multiple compilation options. You can select different compilation options as required.

| Parameter  |  Parameter Description  | Value Range | Mandatory or Not |
| -------- | ----- | ---- | ---- |
| -d | If this parameter is set, the debug version is compiled. Otherwise, the release version is compiled. | - | No |
| -i | If this parameter is set, incremental compilation is performed. Otherwise, full compilation is performed. | - | No |
| -j[n] | Sets the number of threads used during compilation. Otherwise, the number of threads is set to 8 by default. | - | No |
| -I | Selects an applicable architecture. | arm64, arm32, or x86_64 | Yes |
| -e | In the Arm architecture, select the backend operator and set the `gpu` parameter. The built-in GPU operator of the framework is compiled at the same time. | GPU | No |
| -h | Displays the compilation help information. | - | No |

> When the `-I` parameter changes, that is, the applicable architecture is changed, the `-i` parameter cannot be used for incremental compilation.

## Output Description

After the compilation is complete, go to the `mindspore/output` directory of the source code to view the file generated after compilation. The file is named `mindspore-lite-{version}-{function}-{OS}.tar.gz`. After decompression, the tool package named `mindspore-lite-{version}-{function}-{OS}` can be obtained.

> version: version of the output, consistent with that of the MindSpore.
>
> function: function of the output. `converter` indicates the output of the conversion tool and `runtime` indicates the output of the inference framework.
>
> OS: OS on which the output will be deployed.

```bash
tar -xvf mindspore-lite-{version}-{function}-{OS}.tar.gz
```

For the x86 architecture, you can obtain the output of the conversion tool and inference framework；But for the ARM architecture, you only get inference framework.

Generally, the compiled output files include the following types. The architecture selection affects the types of output files.

> For the Arm 64-bit architecture, you can obtain the output of the `arm64-cpu` inference framework. If `-e gpu` is added, you can obtain the output of the `arm64-gpu` inference framework. The compilation for arm 64-bit is the same as that for arm 32-bit.

| Directory | Description | converter | runtime |
| --- | --- | --- | --- |
| include | Inference framework header file | No | Yes |
| lib | Inference framework dynamic library | No | Yes |
| benchmark | Benchmark test tool | No | Yes |
| time_profiler | Time consumption analysis tool at the model network layer| No | Yes |
| converter | Model conversion tool  | Yes | No | No |
| third_party | Header file and library of the third-party library | Yes | Yes |

Take the 0.7.0-beta version and CPU as an example. The contents of `third party` and `lib` vary depending on the architecture as follows:  
- `mindspore-lite-0.7.0-converter-ubuntu`: `third party`include `protobuf` (Protobuf dynamic library).
- `mindspore-lite-0.7.0-runtime-x86-cpu`: `third party`include `flatbuffers` (FlatBuffers header file), `lib`include`libmindspore-lite.so`(Dynamic library of MindSpore Lite inference framework). 
- `mindspore-lite-0.7.0-runtime-arm64-cpu`: `third party`include `flatbuffers` (FlatBuffers header file), `lib`include`libmindspore-lite.so`(Dynamic library of MindSpore Lite inference framework) and `liboptimize.so`(Dynamic library of MindSpore Lite advanced operators).

> `liboptimize.so` only exits in runtime-arm64 outputs, and only can be used in the CPU which supports armv8.2 and fp16.

> Before running the tools in the `converter`, `benchmark`, or `time_profiler` directory, you need to configure environment variables and set the paths of the dynamic libraries of MindSpore Lite and Protobuf to the paths of the system dynamic libraries. The following uses the 0.7.0-beta version as an example: `export LD_LIBRARY_PATH=./mindspore-lite-0.7.0/lib:./mindspore-lite-0.7.0/third_party/protobuf/lib:${LD_LIBRARY_PATH}`.

## Compilation Example

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

> - In the `build.sh` script, run the `git clone` command to obtain the code in the third-party dependency library. Ensure that the network settings of Git are correct.

Take the 0.7.0-beta version as an example. After the release version of the x86_64 architecture is compiled, go to the `mindspore/output` directory and run the following decompression command to obtain the output files `include`, `lib`, `benchmark`, `time_profiler`, `converter`, and `third_party`:

```bash
tar -xvf mindspore-lite-0.7.0-converter-ubuntu.tar.gz
tar -xvf mindspore-lite-0.7.0-runtime-x86-cpu.tar.gz
```
