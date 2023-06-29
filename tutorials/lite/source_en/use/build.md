# Building MindSpore Lite

`Windows` `Linux` `Android` `Environment Preparation` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.2/tutorials/lite/source_en/use/build.md" target="_blank"><img src="../_static/logo_source.png"></a>

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
| -A | Language used by mindspore lite, default cpp. If the parameter is set to java, the AAR and JAR for Linux x86 are compiled. | cpp, java | No |
| -C | If this parameter is set, the converter is compiled, default on. | on, off | No |
| -o | If this parameter is set, the benchmark and static library crop tool are compiled, default on. | on, off | No |
| -t | If this parameter is set, the testcase is compiled, default off. | on, off | No |
| -T | If this parameter is set, MindSpore Lite training version is compiled, i.e., this option is required when compiling, default off. | on, off | No |
| -W | Enable x86_64 SSE or AVX instruction set, default off. | sse, avx, off | No |

> - When the `-I` parameter changes, such as `-I x86_64` is converted to `-I arm64`, adding `-i` for parameter compilation does not take effect.
> - When compiling the AAR package, the `-A java` parameter must be added, and there is no need to add the `-I` parameter. By default, the built-in CPU and GPU operators are compiled at the same time.
> - The compiler will only generate training packages when `-T` is opened.
> - Any `-e` compilation option, the CPU operators will be compiled into it.

### Compilation Example

First, download source code from the MindSpore code repository.

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.2
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

- Compile MindSpore Lite AAR and JAR for Linux x86, AAR compiles the built-in CPU and GPU operators at the same time, but JAR only compiles the built-in CPU:

    ```bash
    bash build.sh -A java
    ```

- Compile MindSpore Lite AAR and JAR for Linux x86, with the built-in CPU operators compiled:

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

- `mindspore-lite-{version}-inference-{os}-{arch}.tar.gz`: Contains model inference framework runtime (cpp), and related tools.
- `mindspore-lite-maven-{version}.zip`: Contains model reasoning framework AAR package.

> - version: Version of the output, consistent with that of the MindSpore.
> - os: Operating system on which the output will be deployed.
> - arch: System architecture on which the output will be deployed.

Execute the decompression command to obtain the compiled output:

```bash
tar -xvf mindspore-lite-{version}-inference-{os}-{arch}.tar.gz
unzip mindspore-lite-maven-{version}.zip
```

#### Description of Converter's Directory Structure

The conversion tool is only available under the `-I x86_64` compilation option, and the content includes the following parts:

```text
mindspore-lite-{version}-inference-linux-x64
└── tools
    └── converter
        ├── converter                # Model conversion tool
        │   └── converter_lite       # Executable program
        ├── lib                      # The dynamic link library that converter depends
        │   └── libmindspore_gvar.so # A dynamic library that stores some global variables
        └── third_party              # The dynamic link library that converter depends
            └── glog
                └── lib
                    └── libglog.so.0 # Dynamic library of Glog
```

#### Description of CodeGen's Directory Structure

The codegen executable program is only available under the `-I x86_64` compilation option, and only the operator library required by the inference code generated by codegen is generated under the `-I arm64` and `-I arm32` compilation options.

- When the compilation option is `-I x86_64`:

    ```text
    mindspore-lite-{version}-inference-linux-x64
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
    mindspore-lite-{version}-inference-android-{arch}
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
mindspore-lite-{version}-inference-linux-x64
└── tools
    └── obfuscator # Model obfuscation tool
        └── msobfuscator          # Executable program
```

#### Description of Runtime and Other tools' Directory Structure

The inference framework can be obtained under `-I x86_64`, `-I arm64` and `-I arm32` compilation options, and the content includes the following parts:

- When the compilation option is `-I x86_64`:

    ```text
    mindspore-lite-{version}-inference-linux-x64
    ├── inference
    │   ├── include  # Header files of inference framework
    │   ├── lib      # Inference framework library
    │   │   ├── libmindspore-lite.a     # Static library of infernece framework in MindSpore Lite
    │   │   ├── libmindspore-lite.so    # Dynamic library of infernece framework in MindSpore Lite
    │   │   └── libmsdeobfuscator-lite.so  # The files of obfuscated model loading dynamic library
    │   └── minddata # Image processing dynamic library
    │       ├── include
    │       └── lib
    │           └── libminddata-lite.so # The files of image processing dynamic library
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
    mindspore-lite-{version}-inference-android-{arch}
    ├── inference
    │   ├── include     # Header files of inference framework
    │   ├── lib         # Inference framework library
    │   │   ├── libmindspore-lite.a  # Static library of infernece framework in MindSpore Lite
    │   │   ├── libmindspore-lite.so # Dynamic library of infernece framework in MindSpore Lite
    │   │   └── libmsdeobfuscator-lite.so # The files of obfuscated model loading dynamic library
    │   ├── minddata    # Image processing dynamic library
    │   │   ├── include
    │   │   └── lib
    │   │       └── libminddata-lite.so # The files of image processing dynamic library
    │   └── third_party # NPU library
    │       └── hiai_ddk
    └── tools
        ├── benchmark # Benchmarking tool
        │   └── benchmark
        │   └── benchmark
        └── codegen   # Code generation tool
            ├── include  # operator header file
            └── lib      # operator static library
    ```

- When the compilation option is `-A java`:

    ```text
    mindspore-lite-maven-{version}
    └── mindspore
        └── mindspore-lite
            └── {version}
                └── mindspore-lite-{version}.aar # MindSpore Lite runtime aar
    ```

    ```text
    mindspore-lite-{version}-inference-linux-x64-jar
    └── jar
        ├── libmindspore-lite-jni.so # Dynamic library of MindSpore Lite inference framework
        ├── libmindspore-lite.so     # MindSpore Lite JNI dynamic library
        └── mindspore-lite-java.jar  # MindSpore Lite inference framework jar package
    ```

> - Compile ARM64 to get the inference framework output of cpu/gpu/npu by default, if you add `-e gpu`, you will get the inference framework output of cpu/gpu, ARM32 only supports CPU.
> - Before running the tools in the converter, benchmark directory, you need to configure environment variables, and configure the path where the dynamic libraries of MindSpore Lite are located to the path where the system searches for dynamic libraries.
> - The dynamic library `libmsdeobfuscator-lite.so` is only available if the `ENABLE_MODEL_OBF` compilation option in `mindspore/mindspore/lite/CMakeLists.txt` is turned on.

Configure converter:

```bash
export LD_LIBRARY_PATH=./output/mindspore-lite-{version}-inference-{os}-{arch}/tools/converter/lib:./output/mindspore-lite-{version}-inference-{os}-{arch}/tools/converter/third_party/glog/lib:${LD_LIBRARY_PATH}
```

Configure benchmark:

```bash
export LD_LIBRARY_PATH=./output/mindspore-lite-{version}-inference-{os}-{arch}/inference/lib:${LD_LIBRARY_PATH}
```

### Training Output Description

If the `-T on` is added to the MindSpore Lite, go to the `mindspore/output` directory of the source code to view the file generated after compilation. The file is divided into the following parts.

- `mindspore-lite-{version}-train-{os}-{arch}.tar.gz`: Contains model training framework, performance analysis tool.

> - version: Version of the output, consistent with that of the MindSpore.
> - os: Operating system on which the output will be deployed.
> - arch: System architecture on which the output will be deployed.

Execute the decompression command to obtain the compiled output:

```bash
tar -xvf mindspore-lite-{version}-train-{os}-{arch}.tar.gz
```

#### Description of Training Runtime and Related Tools' Directory Structure

The MindSpore Lite training framework can be obtained under `-I x86_64`, `-I arm64` and `-I arm32` compilation options, and the content includes the following parts:

- When the compilation option is `-I x86_64`:

    ```text
    mindspore-lite-{version}-train-linux-x64
    ├── tools
    │   ├── benchmark_train # Training model benchmark tool
    │   ├── converter       # Model conversion tool
    │   └── cropper         # Static library crop tool
    │       ├── cropper                 # Executable file of static library crop tool
    │       └── cropper_mapping_cpu.cfg # Crop cpu library related configuration files
    └── train
        ├── include  # Header files of training framework
        ├── lib      # Inference framework library
        │   ├── libmindspore-lite-train.a  # Static library of training framework in MindSpore Lite
        │   └── libmindspore-lite-train.so # Dynamic library of training framework in MindSpore Lite
        └── minddata # Image processing dynamic library
            ├── include
            ├── lib
            │   └── libminddata-lite.so # The files of image processing dynamic library
            └── third_party
    ```

- When the compilation option is `-I arm64` or `-I arm32`:  

    ```text
    mindspore-lite-{version}-train-android-{arch}
    ├── tools
    │   ├── benchmark       # Benchmarking tool
    │   ├── benchmark_train # Training model benchmark tool
    └── train
        ├── include # Header files of training framework
        ├── lib     # Training framework library
        │   ├── libmindspore-lite-train.a  # Static library of training framework in MindSpore Lite
        │   └── libmindspore-lite-train.so # Dynamic library of training framework in MindSpore Lite
        ├── minddata # Image processing dynamic library
        │   ├── include
        │   ├── lib
        │   │   └── libminddata-lite.so # The files of image processing dynamic library
        │   └── third_party
        └── third_party
    ```

> Before running the tools in the converter and the benchmark_train directory, you need to configure environment variables, and configure the path where the dynamic libraries of MindSpore Lite are located to the path where the system searches for dynamic libraries.

Configure converter:

```bash
export LD_LIBRARY_PATH=./output/mindspore-lite-{version}-train-{os}-{arch}/tools/converter/lib:./output/mindspore-lite-{version}-train-{os}-{arch}/tools/converter/third_party/glog/lib:${LD_LIBRARY_PATH}
```

Configure benchmark_train:

```bash
export LD_LIBRARY_PATH=./output/mindspore-lite-{version}-train-{os}-{arch}/train/lib:./output/mindspore-lite-{version}-train-{os}-{arch}/train/minddata/lib:./output/mindspore-lite-{version}-train-{os}-{arch}/train/minddata/third_party:${LD_LIBRARY_PATH}
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
git clone https://gitee.com/mindspore/mindspore.git -b r1.2
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

- `mindspore-lite-{version}-inference-win-x64.zip`: Contains model inference framework and related tool.

> version: Version of the output, consistent with that of the MindSpore.

Execute the decompression command to obtain the compiled output:

```bat
unzip mindspore-lite-{version}-inference-win-x64.zip
```

#### Description of Runtime and Related Tools' Directory Structure

The content includes the following parts:

```text
mindspore-lite-{version}-inference-win-x64
├── inference
│   ├── include # Header files of inference framework
│   └── lib
│       ├── libgcc_s_seh-1.dll      # Dynamic library of MinGW
│       ├── libmindspore-lite.a     # Static library of infernece framework in MindSpore Lite
│       ├── libmindspore-lite.dll   # Dynamic library of infernece framework in MindSpore Lite
│       ├── libmindspore-lite.dll.a # Link file of dynamic library of infernece framework in MindSpore Lite
│       ├── libssp-0.dll            # Dynamic library of MinGW
│       ├── libstdc++-6.dll         # Dynamic library of MinGW
│       └── libwinpthread-1.dll     # Dynamic library of MinGW
└── tools
    ├── benchmark # Benchmarking tool
    │   └── benchmark.exe # Executable program
    └── converter # Model conversion tool
        ├── converter
        │   └── converter_lite.exe    # Executable program
        └── lib
            ├── libgcc_s_seh-1.dll    # Dynamic library of MinGW
            ├── libglog.dll           # Dynamic library of Glog
            ├── libmindspore_gvar.dll # A dynamic library that stores some global variables
            ├── libssp-0.dll          # Dynamic library of MinGW
            ├── libstdc++-6.dll       # Dynamic library of MinGW
            └── libwinpthread-1.dll   # Dynamic library of MinGW
```

> Before running the tools in the converter and the benchmark directory, you need to configure environment variables, and configure the path where the dynamic libraries of MindSpore Lite are located to the path where the system searches for dynamic libraries.

Configure converter:

```bash
set PATH=./output/mindspore-lite-{version}-inference-win-x64/tools/converter/lib:%PATH%
```

Configure benchmark:

```bash
set PATH=./output/mindspore-lite-{version}-inference-win-x64/inference/lib:%PATH%
```

> Currently, MindSpore Lite is not supported on Windows.

## Docker Environment Compilation

### Environmental Preparation

#### Download the docker image

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore-build/mindspore-lite:ubuntu18.04.2-20210323
```

> - Before downloading the image, please make sure docker has been installed.
> - Docker image does not currently support Windows version compilation.

#### Create a container

```bash
docker run -tid --net=host --name=docker01 swr.cn-south-1.myhuaweicloud.com/mindspore-build/mindspore-lite:ubuntu18.04.2-20210323
```

#### Enter the container

```bash
docker exec -ti -u 0 docker01 bash
```

### Compilation Options

Refer to [Linux Environment Compilation](https://www.mindspore.cn/tutorial/lite/en/r1.2/use/build.html#linux-environment-compilation)

### Compilation Example

Refer to [Linux Environment Compilation](https://www.mindspore.cn/tutorial/lite/en/r1.2/use/build.html#linux-environment-compilation)

### Output Description

Refer to [Linux Environment Compilation](https://www.mindspore.cn/tutorial/lite/en/r1.2/use/build.html#linux-environment-compilation)
