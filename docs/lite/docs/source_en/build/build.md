# Building Device-side

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/build/build.md)

This chapter introduces how to quickly compile MindSpore Lite, which includes the following modules:

Modules in MindSpore Lite:

| Module | Support Platform | Description |
| --- | ---- | ---- |
| converter          | Linux, Windows               | Model Conversion Tool    |
| runtime(cpp, java) | Linux, Windows, Android, iOS, OpenHarmony(OHOS) | Model Inference Framework(Windows platform does not support java version runtime) |
| benchmark          | Linux, Windows, Android, OpenHarmony(OHOS)      | Benchmarking Tool        |
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
    - [OpenHarmony_NDK](http://ci.openharmony.cn/workbench/cicd/dailybuild/dailylist)
        - Download OpenHarmony NDK: Go to OpenHarmony daily build web site, select "modality" as "ohos-sdk", then select any successfully built package to download. After downloading, the files in the unzipped package that begin with 'native' are OpenHarmony NDK.
        - Configure environment variables: `export OHOS_NDK=OHOS NDK path`, `export TOOLCHAIN_NAME=ohos`.
- Compilation dependency of the Java API module (optional). If the JAVA_HOME environment variable is not set, this module will not be compiled:
    - [Gradle](https://gradle.org/releases/) >= 6.6.1
        - Configure environment variables: `export GRADLE_HOME=GRADLE path` and `export GRADLE_USER_HOME=GRADLE path`.
        - Add the bin directory to the PATH: `export PATH=${GRADLE_HOME}/bin:$PATH`.
    - [Maven](https://archive.apache.org/dist/maven/maven-3/) >= 3.3.1
        - Configure environment variables: `export MAVEN_HOME=MAVEN path`.
        - Add the bin directory to the PATH: `export PATH=${MAVEN_HOME}/bin:$PATH`.
    - [OpenJDK](https://openjdk.java.net/install/)  1.8 to 1.15
        - Configure environment variables: `export JAVA_HOME=JDK path`.
        - Add the bin directory to the PATH: `export PATH=${JAVA_HOME}/bin:$PATH`.
    - [Android SDK](https://developer.android.com/studio?hl=zh-cn#cmdline-tools)
        - Create a new directory, configure environment variables`export ANDROID_SDK_ROOT=new directory`.
        - Download `SDK Tools`, uncompress and go to directory `cmdline-tools/bin`, create SDK through `sdkmanager`: `./sdkmanager --sdk_root=${ANDROID_SDK_ROOT} "cmdline-tools;latest"`.
        - Accept the license through `sdkmanager` under the `${ANDROID_SDK_ROOT}` directory: `yes | ./sdkmanager --licenses`.
- Compilation dependency of the Python API module (optional). If Python3 or NumPy is not installed, this module will not be compiled:
    - [Python](https://www.python.org/) >= 3.7.0
    - [NumPy](https://numpy.org/) >= 1.17.0 (If the installation with pip fails, please upgrade the pip version first: `python -m pip install -U pip`)
    - [wheel](https://pypi.org/project/wheel/) >= 0.32.0 (If the installation with pip fails, please upgrade the pip version first: `python -m pip install -U pip`)

> Gradle is recommended to use the [gradle-6.6.1-complete](https://gradle.org/next-steps/?version=6.6.1&format=all) version. If you configure other versions, gradle will automatically download  `gradle-6.6.1-complete` by gradle wrapper mechanism.
>
> You can also directly use the Docker compilation image that has been configured with the above dependencies.
>
> - Download the docker image: `docker pull swr.cn-south-1.myhuaweicloud.com/mindspore-build/mindspore-lite:ubuntu18.04.2-20210530`
> - Create a container: `docker run -tid --net=host --name=docker01 swr.cn-south-1.myhuaweicloud.com/mindspore-build/mindspore-lite:ubuntu18.04.2-20210530`
> - Enter the container: `docker exec -ti -u 0 docker01 bash`

### Compilation Options

The script `build.sh` in the root directory of MindSpore can be used to compile MindSpore Lite.

#### Instructions for Parameters of `build.sh`

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

#### Module Build Compilation Options

The construction of modules is controlled by environment variables. Users can control the modules built by compiling by declaring relevant environment variables. After the compilation option is modified, in order to make the option effective, the `-i` parameter cannot be added for incremental compilation when compiling with the `build.sh` script.

- General module compilation options

| Option  |  Parameter Description  | Value Range | Defaults |
| -------- | ----- | ---- | ---- |
| MSLITE_GPU_BACKEND | Set the GPU backend, only opencl is valid when the target OS is not OpenHarmony and `-I arm64`, and only tensorrt is valid when `-I x86_64` | opencl, tensorrt, off | opencl when `-I arm64`, off when `-I x86_64` |
| MSLITE_ENABLE_NPU | Whether to compile NPU operator, only valid when the target OS is not OpenHarmony  `-I arm64` or `-I arm32` | on, off | off |
| MSLITE_ENABLE_TRAIN | Whether to compile the training version | on, off | on |
| MSLITE_ENABLE_SSE | Whether to enable SSE instruction set, only valid when `-I x86_64` | on, off | off |
| MSLITE_ENABLE_AVX | Whether to enable AVX instruction set, only valid when `-I x86_64` | on, off | off |
| MSLITE_ENABLE_AVX512 | Whether to enable AVX512 instruction set, only valid when `-I x86_64` | on, off | off |
| MSLITE_ENABLE_CONVERTER | Whether to compile the model conversion tool, only valid when `-I x86_64` | on, off | on |
| MSLITE_ENABLE_TOOLS | Whether to compile supporting tools | on, off | on |
| MSLITE_ENABLE_TESTCASES | Whether to compile test cases | on, off | off |
| MSLITE_ENABLE_MODEL_ENCRYPTION | Whether to support model encryption and decryption | on, off | off |
| MSLITE_ENABLE_MODEL_PRE_INFERENCE | Whether to enable pre-inference during model compilation | on, off | off |
| MSLITE_ENABLE_GITEE_MIRROR | Whether to enable download third_party from gitee mirror | on, off | off |

> - For TensorRT and NPU compilation environment configuration, refer to [Application Specific Integrated Circuit Integration Instructions](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/third_party/asic.html).
> - When the AVX instruction set is enabled, the CPU of the running environment needs to support both AVX and FMA features.
> - The compilation time of the model conversion tool is long. If it is not necessary, it is recommended to use `MSLITE_ENABLE_CONVERTER` to turn off the compilation of the conversion tool to speed up the compilation.
> - The version supported by the OpenSSL encryption library is 1.1.1k, which needs to be downloaded and compiled by the user. For the compilation, please refer to: <https://github.com/openssl/openssl#build-and-install>. In addition, the path of libcrypto.so.1.1 should be added to LD_LIBRARY_PATH.
> - When pre-inference during model compilation is enabled, for the non-encrypted model, the inference framework will create a child process for pre-inference when Build interface is called. After the child process returns successfully, the main precess will formally execute the process of graph compilation.
> - At present, OpenHarmony only supports CPU reasoning, not GPU reasoning.

- Runtime feature compilation options

If the user is sensitive to the package size of the framework, the following options can be configured to reduce the package size by reducing the function of the runtime model reasoning framework. Then, the user can further reduce the package size by operator reduction through the [cropper tool](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/tools/cropper_tool.html).

| Option  |  Parameter Description  | Value Range | Defaults |
| -------- | ----- | ---- | ---- |
| MSLITE_STRING_KERNEL | Whether to support string data reasoning model, such as smart_reply.tflite | on,off | on |
| MSLITE_ENABLE_CONTROLFLOW | Whether to support control flow model | on,off | on |
| MSLITE_ENABLE_WEIGHT_DECODE | Whether to support weight quantitative model | on,off | on |
| MSLITE_ENABLE_CUSTOM_KERNEL | Whether to support southbound operator registration | on,off | on |
| MSLITE_ENABLE_DELEGATE | Whether to  support Delegate mechanism | on,off | on |
| MSLITE_ENABLE_FP16 | Whether to support FP16 operator | on,off | off when `-I x86_64`, on when `-I arm64`, when `-I arm32`, if the Android_NDK version is greater than r21e, it is on, otherwise it is off |
| MSLITE_ENABLE_INT8 | Whether to support INT8 operator | on,off | on |

> - Since the implementation of NPU and TensorRT depends on the Delegate mechanism, the Delegate mechanism cannot be turned off when using NPU or TensorRT. If the Delegate mechanism is turned off, the related functions must also be turned off.

### Compilation Example

First, download source code from the MindSpore code repository.

```bash
git clone -b v2.6.0rc1 https://gitee.com/mindspore/mindspore.git
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

- Compile the aarch32 or aarch64 package of the OpenHarmony OS:

    Compile aarch32 package

    ```bash
    export OHOS_NDK=OHOS NDK path
    export TOOLCHAIN_NAME=ohos
    bash build.sh -I arm32 -j32
    ```

    Compile aarch64 package

    ```bash
    export OHOS_NDK=OHOS NDK path
    export TOOLCHAIN_NAME=ohos
    bash build.sh -I arm64 -j32
    ```

Finally, the following files will be generated in the `output/` directory:

- `mindspore-lite-{version}-{os}-{arch}.tar.gz`: Contains runtime, and related tools.

- `mindspore-lite-maven-{version}.zip`: The AAR package which contains runtime (java).

- `mindspore-lite-{version}-{python}-{os}-{arch}.whl`: The Whl package which contains runtime (Python).

> - version: Version of the output, consistent with that of the MindSpore.
> - python: Python version of the output, for example, Python 3.7 is `cp37-cp37m`.
> - os: Operating system on which the output will be deployed.
> - arch: System architecture on which the output will be deployed.

To experience the Python API, you need to move to the 'output/' directory and use the following command to install the Whl installation package.

```bash
pip install `mindspore-lite-{version}-{python}-{os}-{arch}.whl`
```

After installation, you can use the following command to check whether the installation is successful: If no error is reported, the installation is successful.

```bash
python -c "import mindspore_lite"
```

After successful installation, you can use the command of `pip show mindspore_lite` to check the installation location of the Python module of MindSpot Lite.

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
    │   │   ├── libmsdeobfuscator-lite.so  # The files of obfuscated model loading dynamic library, need to open the `MSLITE_ENABLE_MODEL_OBF` option.
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
    │   │   ├── libmindspore-lite-train.so # Dynamic library of training framework in MindSpore Lite
    │   │   └── libmsdeobfuscator-lite.so  # The files of obfuscated model loading dynamic library, need to open the `MSLITE_ENABLE_MODEL_OBF` option.
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

- When the compilation option is `-I arm64` or `-I arm32`, and specifies`TOOLCHAIN_NAME=ohos`:

    ```text
    mindspore-lite-{version}-ohos-{arch}
    ├── runtime
    │   ├── include
    │   └── lib
    │       ├── libmindspore-lite.a  # Static library of inference framework in MindSpore Lite
    │       └── libmindspore-lite.so # Dynamic library of training framework in MindSpore Lite
    └── tools
        └── benchmark                # Benchmarking tool
    ```

## Windows Environment Compilation

### Environment Requirements

- System environment: Windows 7, Windows 10; 64-bit.

- MinGW compilation dependencies:
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - Compile 64-bit: [MinGW-W64 x86_64](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/7.3.0/threads-posix/seh/x86_64-7.3.0-release-posix-seh-rt_v5-rev0.7z) = GCC-7.3.0
    - Compile 32-bit: [MinGW-W64 i686](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win32/Personal%20Builds/mingw-builds/7.3.0/threads-posix/dwarf/i686-7.3.0-release-posix-dwarf-rt_v5-rev0.7z) = GCC-7.3.0

- Visual Studio compilation dependencies:
    - [Visual Studio](https://visualstudio.microsoft.com/vs/older-downloads/) = 2017, cmake is included.
    - Compile 64-bit: Enter the start menu, click "x64 Native Tools Command Prompt for VS 2017", or open the cmd window and execute `call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Profession\VC\Auxiliary\Build\vcvars64.bat"`.
    - Compile 32-bit: Enter the start menu, click "x64_x86 Cross Tools Command Prompt for VS 2017", or open the cmd window and execute `call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Profession\VC\Auxiliary\Build\vcvarsamd64_x86.bat"`.

### Compilation Options

The script `build.bat` in the root directory of MindSpore can be used to compile MindSpore Lite.

#### The Compilation Parameter of `build.bat`

| Parameter  |  Parameter Description   | Mandatory or Not |
| -------- | ----- | ---- |
| lite | Set this parameter to compile the MindSpore Lite project. | Yes |
| [n] | Set the number of threads used during compilation, otherwise the default is set to 6 threads.  | No |

#### The Options of `mindspore/lite/CMakeLists.txt`

| Option  |  Parameter Description  | Value Range | Defaults |
| -------- | ----- | ---- | ---- |
| MSLITE_ENABLE_SSE | Whether to enable SSE instruction set | on, off | off |
| MSLITE_ENABLE_AVX | Whether to enable AVX instruction set (This option does not currently support the Visual Studio compiler) | on, off | off |
| MSLITE_ENABLE_AVX512 | Whether to enable AVX512 instruction set (This option does not currently support the Visual Studio compiler) | on, off | off |
| MSLITE_ENABLE_CONVERTER | Whether to compile the model conversion tool (This option does not currently support the Visual Studio compiler) | on, off | on |
| MSLITE_ENABLE_TOOLS | Whether to compile supporting tools | on, off | on |
| MSLITE_ENABLE_TESTCASES | Whether to compile test cases | on, off | off |
| MSLITE_ENABLE_GITEE_MIRROR | Whether to enable download third_party from gitee mirror | on, off | off |

> - The above options can be modified by setting the environment variable with the same name or the file `mindspore/lite/CMakeLists.txt`.
> - The compilation time of the model conversion tool is long. If it is not necessary, it is recommended to use `MSLITE_ENABLE_CONVERTER` to turn off the compilation of the conversion tool to speed up the compilation.

### Compilation Example

First, use the git tool to download the source code from the MindSpore code repository.

```bat
git clone -b v2.6.0rc1 https://gitee.com/mindspore/mindspore.git
```

Then, use the cmd tool to compile MindSpore Lite in the root directory of the source code and execute the following commands.

- Turn on SSE instruction set optimization and compile with 8 threads.

```bat
set MSLITE_ENABLE_SSE=on
call build.bat lite 8
```

Finally, the following files will be generated in the `output/` directory:

- `mindspore-lite-{version}-win-x64.zip`: Contains model inference framework and related tool.

> version: Version of the output, consistent with that of the MindSpore.

### Directory Structure

- When the compiler is MinGW:

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

- When the compiler is Visual Studio:

    ```text
    mindspore-lite-{version}-win-x64
    ├── runtime
    │   ├── include
    │   └── lib
    │       ├── libmindspore-lite.dll     # Dynamic library of inference framework in MindSpore Lite
    │       ├── libmindspore-lite.dll.lib # Import library of dynamic library of inference framework in MindSpore Lite
    │       └── libmindspore-lite.lib     # Static library of inference framework in MindSpore Lite
    └── tools
        └── benchmark # Benchmarking tool
    ```

> - When linking the static library compiled by MinGW, you need to add `-Wl, --whole-archive mindspore-lite -Wl, --no-whole-archive` in the link options.
> - When linking the static library compiled by Visual Studio, you need to add `/WHOLEARCHIVE:libmindspore-lite.lib` in "Property Pages -> Linker -> Command Line -> Additional Options".
> - When using the Visual Studio compiler, std::ios::binary must be added to read the model stream, otherwise the problem of incomplete reading of the model file will occur.
> - Currently, MindSpore Lite is not supported on Windows.

## macOS Environment Compilation

### Environment Requirements

- System environment: macOS 10.15.4 and above ; 64-bit.

- Compilation dependencies are:
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [Xcode](https://developer.apple.com/xcode/) == 11.4.1
    - [Git](https://git-scm.com/downloads) >= 2.28.0

> - The compilation script will execute `git clone` to obtain the code of the third-party dependent libraries.

### Compilation Options

The script `build.sh` in the root directory of MindSpore can be used to compile MindSpore Lite.

#### The Compilation Parameter of `build.sh`

| Parameter  |  Parameter Description  | Value Range | Defaults |
| -------- | ----- | ---- | ---- |
| -I | Selects an applicable architecture. | arm64, arm32 | None |
| -j[n] | Sets the number of threads used during compilation. Otherwise, the number of threads is set to 8 by default. | Integer | 8 |

#### The Options of `mindspore/lite/CMakeLists.txt`

| Option               | Parameter Description            | Value Range | Defaults |
|----------------------|----------------------------------| ---- | ---- |
| MSLITE_ENABLE_COREML | Whether to enable CoreML backend | on, off | off |

### Compilation Example

First, use the git tool to download the source code from the MindSpore code repository.

```bash
git clone -b v2.6.0rc1 https://gitee.com/mindspore/mindspore.git
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
    ├── Headers        # The inference framework header files
    ├── Info.plist     # The configured files
    └── mindspore-lite # The static library
```

> Currently, device-side training and model conversion are not supported on macOS.
