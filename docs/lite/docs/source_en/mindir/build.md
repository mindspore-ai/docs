# Building Cloud-side MindSpore Lite

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/mindir/build.md)

This section describes how to quickly compile MindSpore Lite.

Cloud-side MindSpore Lite contains modules:

| Modules               | Supported Platforms | Description         |
| ------------------ | -------- | ------------ |
| converter          | Linux    | Model Converter |
| runtime(cpp, java) | Linux    | Model Inference Framework |
| benchmark          | Linux    | Benchmarking Tool |
| minddata           | Linux    | Image Processing Library   |
| akg                | Linux                   | Polyhedral-based deep learning operator compiler ([Auto Kernel Generator](https://gitee.com/mindspore/akg)) |

## Environment Requirements

- System Environment: Linux x86_64 or arm64, Ubuntu 18.04.02LTS recommended
- C++ compilation dependencies
    - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [Git](https://git-scm.com/downloads) >= 2.28.0
- Compilation dependency of the Java API module (optional), which is not compiled if the JAVA_HOME environment variable is not set.
    - [Gradle](https://gradle.org/releases/) >= 6.6.1
        - Configure environment variables: `export GRADLE_HOME=GRADLE path`, and `export GRADLE_USER_HOME=GRADLE path`
        - Add the bin directory to the PATH: `export PATH=${GRADLE_HOME}/bin:$PATH`
    - [Maven](https://archive.apache.org/dist/maven/maven-3/) >= 3.3.1
        - Configure environment variables: `export MAVEN_HOME=MAVEN path`
        - Add the bin directory to the PATH: `export PATH=${MAVEN_HOME}/bin:$PATH`
    - [OpenJDK](https://openjdk.java.net/install/) between 1.8 and 1.15
        - Configure environment variables: `export JAVA_HOME=JDK path`
        - Add the bin directory to the PATH: `export PATH=${JAVA_HOME}/bin:$PATH`
- Compilation dependency for the Python API module (optional), which is not compiled if Python3 or NumPy is not installed.
    - [Python](https://www.python.org/) >= 3.7.0
    - [NumPy](https://numpy.org/) >= 1.17.0 (If installation with pip fails, please upgrade the pip version first: `python -m pip install -U pip`)
    - [wheel](https://pypi.org/project/wheel/) >= 0.32.0 (If installation with pip fails, please upgrade the pip version first: `python -m pip install -U pip`)
- Compilation dependency for AKG (optional, compiled by default), which is not compiled if LLVM-12 or Python3 is not installed. To compile the AKG for the Ascend backend, git-lfs must be installed.
    - [llvm](#installing-llvm-optional) == 12.0.1
    - [git-lfs](https://git-lfs.com/)

> Gradle recommends using [gradle-6.6.1-complete](https://gradle.org/next-steps/?version=6.6.1&format=all), and configuring other versions of gradle will use the gradle wrapper mechanism to automatically download ` gradle-6.6.1-complete`.
>
> You can also directly use Docker compiling images that have been configured with the above dependencies.
>
> - Download images: `docker pull swr.cn-south-1.myhuaweicloud.com/mindspore-build/mindspore-lite:ubuntu18.04.2-20210530`
> - Create a container: `docker run -tid --net=host --name=docker01 swr.cn-south-1.myhuaweicloud.com/mindspore-build/mindspore-lite:ubuntu18.04.2-20210530`
> - Enter the container: `docker exec -ti -u 0 docker01 bash`

## Compilation Options

The `build.sh` script in the MindSpore root directory can be used to compile cloud-side MindSpore Lite.

### Instructions for Using the Parameters of `build.sh`

| Parameters  |  Description of the parameters  | Range of values  | Default values |
| -------- | ----- | ---- | ---- |
| -I | Select target architecture | arm64, x86_64 | None |
| -d | Set this parameter to compile the Debug version, otherwise compile the Release version  | None | None |
| -i | Set this parameter for incremental compilation, otherwise for full compilation | None | None |
| -j[n] | Set the number of threads used at compile time, otherwise the default setting is 8 threads | Integer | 8 |
| -K | Set whether to compile AKG during compilation, otherwise the default setting is on | on, off | on |

> - If the JAVA_HOME environment variable is configured and Gradle is installed, the JAR package is compiled at the same time.
> - Add the `-i` parameter for incremental compilation does not take effect when the `-I` parameter changes, e.g. `-I x86_64` becomes `-I arm64`.
> - Cross-compilation is not supported, i.e. the arm64 version needs to be compiled in the arm environment.

### Module Build Compilation Options

The building of modules is controlled through environment variables, and the users can control the build compilation modules by declaring the relevant environment variables. After modifying the compilation options, the `-i` parameter can not be added for incremental compilation when compiling with the `build.sh` script in order for the options to take effect.

General module compilation options:

| Options  |  Description of the parameters  | Range of values  | Default values |
| -------- | ----- | ---- | ---- |
| MSLITE_GPU_BACKEND                   | Set GPU backend, only tensorrt is valid at `-I x86_64`. | tensorrt, off | off in `-I x86_64` |
| MSLITE_ENABLE_TOOLS                  | Whether to compile the accompanying benchmarking tool          | on, off       | on                   |
| MSLITE_ENABLE_TESTCASES              | Whether to compile test cases                           | on, off       | off                  |
| MSLITE_ENABLE_ACL                    | Whether to enable Ascend ACL                            | on, off       | off                  |
| MSLITE_ENABLE_CLOUD_INFERENCE | Whether to enable cloud-side inference                           | on, off       | off                  |
| MSLITE_ENABLE_SSE | Whether to enable SSE instruction set, only valid for `-I x86_64` | on, off | off |
| MSLITE_ENABLE_AVX512 | Whether to enable AVX512 instruction set, only valid for `-I x86_64` | on, off | off |

> - The cloud-side inference version relies on the model converter, so when ``MSLITE_ENABLE_CLOUD_INFERENCE`` is configured to ``on``, it will compile ``converter`` at the same time.
> - If the environment only supports the SSE instruction set, the AVX512 instruction set needs to be configured as ``off``.

## Compilation Examples

First, you need to download the source code from the MindSpore code repository before compiling.

```bash
git clone https://gitee.com/mindspore/mindspore.git
```

### Environment Preparation

#### Ascend

- Verify the installation of the Ascend AI processor package.

    - The Ascend package is available in both commercial and community versions.

        1. Please refer to the [Ascend Data Center Solution 23.0.RC3 Installation Guide document](https://support.huawei.com/enterprise/en/doc/EDOC1100336282) for the download link and installation method of the commercial version.

        2. There is no restriction on downloading the Community Edition. Please go to [CANN Community Edition](https://www.hiascend.com/developer/download/community/result?module=cann), select `7.0.RC1.beta1` version, and get the corresponding firmware and driver installation packages from the [Firmware and Driver](https://www.hiascend.com/hardware/firmware-drivers?tag=community). For package selection and installation, please refer to the commercial version installation guide document above.
    - The default installation path for the installation package is `/usr/local/Ascend`. After installation, make sure the current user has permission to access the installation path of the Ascend AI processor companion package. If you don't have permission, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.
    - Install the whl package included with the Ascend AI processor companion software. If you have previously installed the package included with the Ascend AI processor, you need to uninstall the corresponding whl package first by using the following command.

    ```bash
    pip uninstall te topi -y
    ```

    The default installation path is installed via the following command. If the installation path is not the default path, you need to replace the path in the command with the installation path.

    ```bash
    pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/topi-{version}-py3-none-any.whl
    pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-{version}-py3-none-any.whl
    ```

- Configure environment variables

    - After installing Ascend package, you need to export Runtime-related environment variables. `/usr/local/Ascend` of the `/LOCAL_ASCEND=/usr/local/Ascend` in the following command indicates the installation path of the package, so you need to change it to the actual installation path of the package.

        ```bash
        # control log level. 0-EBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
        export GLOG_v=2

        # Conda environmental options
        LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

        # lib libraries that the run package depends on
        export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

        # Environment variables that must be configured
        export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
        export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
        export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/compiler/ccec_compiler/bin/:${PATH}                  # TBE operator compilation tool path
        export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                       # Python library that TBE implementation depends on
        ```

#### GPU

GPU environment compilation. Using TensorRT requires integration with CUDA, TensorRT. The current version is adapted to [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive) and [TensorRT 8.5.1](https://developer.nvidia.com/nvidia-tensorrt-8x-download).

Install the appropriate version of CUDA and set the installed directory to the environment variable `${CUDA_HOME}`. The build script will use this environment variable to find CUDA.

Download the corresponding version of the TensorRT archive and set the directory where the archive was unzipped to the environment variable `${TENSORRT_PATH}`. The build script will use this environment variable to find TensorRT.

#### CPU

Use x86_64 or ARM64 environment.

#### Installing LLVM-optional

The CPU backend of the graph kernel fusion in the converter needs to rely on LLVM-12. Run the following commands to install [LLVM](https://llvm.org/) to enable CPU backend. If LLVM-12 is not installed, the graph kernel fusion can only support GPU and Ascend backend.

```shell
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-12 main"
sudo apt-get update
sudo apt-get install llvm-12-dev -y
```

### Executing Compilation

Three-backend-unification packages need to configure the following environment variables:

```bash
export MSLITE_ENABLE_CLOUD_INFERENCE=on
export MSLITE_GPU_BACKEND=tensorrt
export MSLITE_ENABLE_ACL=on
```

> - If you don't need Ascend backend, you can configure ``export MSLITE_ENABLE_ACL=off``.
> - If you don't need GPU backend, you can configure ``export MSLITE_GPU_BACKEND=off``.

Execute the following command in the source root directory to compile different versions of MindSpore Lite.

- Compile the x86_64 architecture version while set the number of threads.

    ```bash
    bash build.sh -I x86_64 -j32
    ```

- Compile the arm64 architecture version while set the number of threads.

    ```bash
    bash build.sh -I arm64 -j32
    ```

- Compile the x86_64 architecture version while setting the number of threads, but do not compile AKG.

    ```bash
    bash build.sh -I x86_64 -j32 -K off
    ```

Finally, the following file will be generated in the `output/` directory:

- `mindspore-lite-{version}-{os}-{arch}.tar.gz`: contains runtime and companion tools.

- `mindspore-lite-{version}-{python}-{os}-{arch}.whl`: contains the Whl package for runtime (Python).

> - version: The version number of the output, which corresponds to the version of the branch code compiled.
> - python: The output Python version, e.g. Python 3.7 for `cp37-cp37m`.
> - os: The operating system to which the output piece should be deployed
> - arch: The system architecture in which the output pieces should be deployed.

To experience the Python interface, you need to move to the `output/` directory and use the following command to install the Whl installer.

```bash
pip install mindspore-lite-{version}-{python}-{os}-{arch}.whl
```

After installation, you can use the following command to check whether the installation is successful: if no error is reported, the installation is successful.

```bash
python -c "import mindspore_lite"
```

After installation, you can use the following command to check if the built-in AKG in MindSpore Lite is installed successfully: if no error is reported, the installation is successful.

```bash
python -c "import mindspore_lite.akg"
```

After successful installation, you can use the `pip show mindspore_lite` command to see where the Python modules for MindSpore Lite are installed.

## Directory Structure

```text
mindspore-lite-{version}-linux-{arch}
├── runtime
│   ├── include
│   ├── lib
│   │   ├── libascend_kernel_plugin.so # Ascend Kernel Plugin Dynamic Library
│   │   ├── libdvpp_utils.so           # DVPP Image Preprocessing Tools Dynamic Library
│   │   ├── libminddata-lite.a         # Image Processing Static Library
│   │   ├── libminddata-lite.so        # Image Processing Dynamic Library
│   │   ├── libmindspore-core.so       # MindSpore Core Dynamic Library
│   │   ├── libmindspore-glog.so.0     # glog Dynamic Library
│   │   ├── libmindspore-lite-jni.so   # jni dynamic library of MindSpore Lite inference framework
│   │   ├── libmindspore-lite.so       # MindSpore Lite Inference Framework Dynamic Library
│   │   ├── libmsplugin-ge-litert.so   # GE LiteRT Plugin Dynamic Library
│   │   └── mindspore-lite-java.jar    # MindSpore Lite Inference framework jar package
│   └── third_party
│       ├── glog
│       ├── libjpeg-turbo
│       └── securec
└── tools
    ├── akg
    |    └── akg-{version}-{python}-linux-{arch}.whl # AKG Python whl package
    ├── benchmark              # Benchmarking Tools
    │   └── benchmark          # Benchmarking tool executable file
    └── converter              # Model converter
        ├── converter
        │   └── converter_lite # Converter executable file
        ├── include
        └── lib
            ├── libascend_pass_plugin.so
            ├── libmindspore_converter.so
            ├── libmindspore_core.so
            ├── libmindspore_glog.so.0
            ├── libmslite_shared_lib.so
            ├── libmslite_converter_plugin.so
            ├── libopencv_core.so.4.5
            ├── libopencv_imgcodecs.so.4.5
            └── libopencv_imgproc.so.4.5
```

