# Building MindSpore Lite

`Windows` `macOS` `Linux` `iOS` `Android` `Environment Preparation` `Intermediate` `Expert`

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/lite/docs/source_en/use/build.md)

This chapter introduces how to quickly compile MindSpore Lite, which includes the following modules:

Modules in MindSpore Lite:

| Module | Support Platform | Description |
| --- | ---- | ---- |
| converter          | Linux, Windows               | Model Conversion Tool    |
| runtime(cpp, java) | Linux, Windows, Android, iOS | Model Inference Framework(Windows platform does not support java version runtime) |
| benchmark          | Linux, Windows, Android      | Benchmarking Tool        |
| benchmark_train    | Linux, Android               | Performance and Accuracy Validation              |
| cropper            | Linux                        | Static library crop tool for libmindspore-lite.a |
| minddata           | Linux, Android               | Image Processing Library |
| codegen            | Linux                        | Model inference code generation tool |
| obfuscator         | Linux                        | Model Obfuscation Tool   |

## Linux Environment Compilation

### Environment Requirements

- The compilation environment supports Linux x86_64 only. Ubuntu 18.04.02 LTS is recommended.
- Compilation dependencies of cpp:
    - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [Git](https://git-scm.com/downloads) >= 2.28.0
    - [Android_NDK](https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip) >= r20
        - Configure environment variables: `export ANDROID_NDK=NDK path`.
- Additional compilation dependencies of Java:
    - [Gradle](https://gradle.org/releases/) >= 6.6.1
        - Configure environment variables: `export GRADLE_HOME=GRADLE path`.
        - Add the bin directory to the PATH: `export PATH=${GRADLE_HOME}/bin:$PATH`.
    - [OpenJDK](https://openjdk.java.net/install/) >= 1.8
        - Configure environment variables: `export JAVA_HOME=JDK path`.
        - Add the bin directory to the PATH: `export PATH=${JAVA_HOME}/bin:$PATH`.
    - [Android SDK](https://developer.android.com/studio?hl=zh-cn#cmdline-tools)
        - Create a new directory, configure environment variables`export ANDROID_SDK_ROOT=new directory`.
        - Download `SDK Tools`, create SDK through `sdkmanager`: `./sdkmanager --sdk_root=${ANDROID_SDK_ROOT} "cmdline-tools;latest"`.
        - Accept the license through `sdkmanager` under the `${ANDROID_SDK_ROOT}` directory: `yes | ./sdkmanager --licenses`.

> You can also directly use the Docker compilation image that has been configured with the above dependencies.
>
> - Download the docker image: `docker pull swr.cn-south-1.myhuaweicloud.com/mindspore-build/mindspore-lite:ubuntu18.04.2-20210530`
> - Create a container: `docker run -tid --net=host --name=docker01 swr.cn-south-1.myhuaweicloud.com/mindspore-build/mindspore-lite:ubuntu18.04.2-20210530`
> - Enter the container: `docker exec -ti -u 0 docker01 bash`

### Compilation Options

The script `build.sh` in the root directory of MindSpore can be used to compile MindSpore Lite.

#### The compilation parameter of `build.sh`

| Parameter  |  Parameter Description  | Value Range | Defaults |
| -------- | ----- | ---- | ---- |
| -I | Selects an applicable architecture. | arm64, arm32, or x86_64 | None |
| -A | Compile AAR package (including arm32 and arm64). | on, off | off |
| -d | If this parameter is set, the debug version is compiled. Otherwise, the release version is compiled. | None | None |
| -i | If this parameter is set, incremental compilation is performed. Otherwise, full compilation is performed. | None | None |
| -j[n] | Sets the number of threads used during compilation. Otherwise, the number of threads is set to 8 by default. | Integer | 8 |
| -a | Whether to enable AddressSanitizer | on, off | off |

> - When compiling the x86_64 version, if the JAVA_HOME environment variable is configured and Gradle is installed, the JAR package will be compiled at the same time.
> - When the `-I` parameter changes, such as `-I x86_64` is converted to `-I arm64`, adding `-i` for parameter compilation does not take effect.
> - When compiling the AAR package, the `-A on` parameter must be added, and there is no need to add the `-I` parameter.

#### The options of `mindspore/lite/CMakeLists.txt`

| Option  |  Parameter Description  | Value Range | Defaults |
| -------- | ----- | ---- | ---- |
| MSLITE_GPU_BACKEND | Set the GPU backend, only opencl is valid when `-I arm64`, and only tensorrt is valid when `-I x86_64` | opencl, tensorrt, off | opencl when `-I arm64`, off when `-I x86_64` |
| MSLITE_ENABLE_NPU | Whether to compile NPU operator, only valid when `-I arm64` or `-I arm32` | on, off | off |
| MSLITE_ENABLE_TRAIN | Whether to compile the training version | on, off | on |
| MSLITE_ENABLE_SSE | Whether to enable SSE instruction set, only valid when `-I x86_64` | on, off | off |
| MSLITE_ENABLE_AVX | Whether to enable AVX instruction set, only valid when `-I x86_64` | on, off | off |
| MSLITE_ENABLE_CONVERTER | Whether to compile the model conversion tool, only valid when `-I x86_64` | on, off | on |
| MSLITE_ENABLE_TOOLS | Whether to compile supporting tools | on, off | on |
| MSLITE_ENABLE_TESTCASES | Whether to compile test cases | on, off | off |

> - The above options can be modified by setting the environment variable with the same name or the file `mindspore/lite/CMakeLists.txt`.
> - After modifying the Option, adding the `-i` parameter for incremental compilation will not take effect.

### Compilation Example

First, download source code from the MindSpore code repository.

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.3
```

Then, run the following commands in the root directory of the source code to compile MindSpore Lite of different versions:

- Compile the x86_64 architecture version and set the number of threads at the same time.

    ```bash
    bash build.sh -I x86_64 -j32
    ```

- Compile the ARM64 architecture version without compiling training-related code.

    ```bash
    export MSLITE_ENABLE_TRAIN=off
    bash build.sh -I arm64 -j32
    ```

    Or modify `mindspore/lite/CMakeLists.txt` to set MSLITE_ENABLE_TRAIN to off and execute the command:

    ```bash
    bash build.sh -I arm64 -j32
    ```

- Compile the AAR package containing aarch64 and aarch32.

    ```bash
    bash build.sh -A on -j32
    ```

Finally, the following files will be generated in the `output/` directory:

- `mindspore-lite-{version}-{os}-{arch}.tar.gz`: Contains runtime, and related tools.

- `mindspore-lite-maven-{version}.zip`: The AAR package which contains runtime (java).

> - version: Version of the output, consistent with that of the MindSpore.
> - os: Operating system on which the output will be deployed.
> - arch: System architecture on which the output will be deployed.

### Directory Structure

- When the compilation option is `-I x86_64`:

    ```text
    mindspore-lite-{version}-linux-x64
    ├── runtime
    │   ├── include
    │   ├── lib
    │   │   ├── libminddata-lite.a         # Static library of image processing
    │   │   ├── libminddata-lite.so        # Dynamic library of image processing
    │   │   ├── libmindspore-lite.a        # Static library of inference framework in MindSpore Lite
    │   │   ├── libmindspore-lite-jni.so   # Dynamic library of inference framework jni in MindSpore Lite
    │   │   ├── libmindspore-lite.so       # Dynamic library of inference framework in MindSpore Lite
    │   │   ├── libmindspore-lite-train.a  # Static library of training framework in MindSpore Lite
    │   │   ├── libmindspore-lite-train.so # Dynamic library of training framework in MindSpore Lite
    │   │   ├── libmsdeobfuscator-lite.so  # The files of obfuscated model loading dynamic library, need to open the `ENABLE_MODEL_OBF` option.
    │   │   └── mindspore-lite-java.jar    # Jar of inference framework in MindSpore Lite
    │   └── third_party
    │       └── libjpeg-turbo
    └── tools
        ├── benchmark       # Benchmarking tool
        ├── benchmark_train # Training model benchmark tool
        ├── codegen         # Code generation tool
        ├── converter       # Model conversion tool
        ├── obfuscator      # Model obfuscation tool
        └── cropper         # Static library crop tool
    ```

- When the compilation option is `-I arm64` or `-I arm32`:

    ```text
    mindspore-lite-{version}-android-{arch}
    ├── runtime
    │   ├── include
    │   ├── lib
    │   │   ├── libminddata-lite.a         # Static library of image processing
    │   │   ├── libminddata-lite.so        # Dynamic library of image processing
    │   │   ├── libmindspore-lite.a        # Static library of inference framework in MindSpore Lite
    │   │   ├── libmindspore-lite.so       # Dynamic library of inference framework in MindSpore Lite
    │   │   ├── libmindspore-lite-train.a  # Static library of training framework in MindSpore Lite
    │   │   └── libmindspore-lite-train.so # Dynamic library of training framework in MindSpore Lite
    │   │   └── libmsdeobfuscator-lite.so  # The files of obfuscated model loading dynamic library, need to open the `ENABLE_MODEL_OBF` option.
    │   └── third_party
    │       ├── hiai_ddk
    │       └── libjpeg-turbo
    └── tools
        ├── benchmark       # Benchmarking tool
        ├── benchmark_train # Training model benchmark tool
        └── codegen         # Code generation tool
    ```

- When the compilation option is `-A on`:

    ```text
    mindspore-lite-maven-{version}
    └── mindspore
        └── mindspore-lite
            └── {version}
                └── mindspore-lite-{version}.aar # MindSpore Lite runtime aar
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

The script `build.bat` in the root directory of MindSpore can be used to compile MindSpore Lite.

#### The compilation parameter of `build.bat`

| Parameter  |  Parameter Description   | Mandatory or Not |
| -------- | ----- | ---- |
| lite | Set this parameter to compile the MindSpore Lite project. | Yes |
| [n] | Set the number of threads used during compilation, otherwise the default is set to 6 threads.  | No |

#### The options of `mindspore/lite/CMakeLists.txt`

| Option  |  Parameter Description  | Value Range | Defaults |
| -------- | ----- | ---- | ---- |
| MSLITE_ENABLE_SSE | Whether to enable SSE instruction set | on, off | off |
| MSLITE_ENABLE_AVX | Whether to enable AVX instruction set | on, off | off |
| MSLITE_ENABLE_CONVERTER | Whether to compile the model conversion tool | on, off | on |
| MSLITE_ENABLE_TOOLS | Whether to compile supporting tools | on, off | on |
| MSLITE_ENABLE_TESTCASES | Whether to compile test cases | on, off | off |

> - The above options can be modified by setting the environment variable with the same name or the file `mindspore/lite/CMakeLists.txt`.

### Compilation Example

First, use the git tool to download the source code from the MindSpore code repository.

```bat
git clone https://gitee.com/mindspore/mindspore.git -b r1.3
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

Finally, the following files will be generated in the `output/` directory:

- `mindspore-lite-{version}-win-x64.zip`: Contains model inference framework and related tool.

> version: Version of the output, consistent with that of the MindSpore.

### Directory Structure

```text
mindspore-lite-{version}-win-x64
├── runtime
│   ├── include
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
    └── converter # Model conversion tool
```

> Currently, MindSpore Lite is not supported on Windows.

## macOS Environment Compilation

### Environment Requirements

- System environment: macOS 10.15.4 and above ; 64-bit.

- Compilation dependencies are:
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [Xcode](https://developer.apple.com/xcode/download/cn) == 11.4.1
    - [Git](https://git-scm.com/downloads) >= 2.28.0

> - The compilation script will execute `git clone` to obtain the code of the third-party dependent libraries.

### Compilation Options

The script `build.sh` in the root directory of MindSpore can be used to compile MindSpore Lite.

#### The compilation parameter of `build.sh`

| Parameter  |  Parameter Description  | Value Range | Defaults |
| -------- | ----- | ---- | ---- |
| -I | Selects an applicable architecture. | arm64, arm32 | None |
| -j[n] | Sets the number of threads used during compilation. Otherwise, the number of threads is set to 8 by default. | Integer | 8 |

### Compilation Example

First, use the git tool to download the source code from the MindSpore code repository.

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.3
```

Then, use the cmd tool to compile MindSpore Lite in the root directory of the source code and execute the following commands.

- Compile the ARM64 architecture version

    ```bash
    bash build.sh -I arm64 -j8
    ```

- Compile the ARM32 architecture version

    ```bash
    bash build.sh -I arm32 -j8
    ```

Finally, the following files will be generated in the `output/` directory:

- `mindspore-lite-{version}-{os}-{arch}.tar.gz`: Contains model inference framework.

> - version: Version of the output, consistent with that of the MindSpore.
> - os: Operating system on which the output will be deployed.
> - arch: System architecture on which the output will be deployed.

### Directory Structure

```text
mindspore-lite.framework
└── runtime
    ├── Headers        # 推理框架头文件
    ├── Info.plist     # 配置文件
    └── mindspore-lite # 静态库
```

> Currently, MindSpore Lite Train and converter are not supported on macOS.
