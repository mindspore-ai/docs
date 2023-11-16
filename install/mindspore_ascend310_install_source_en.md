# Installing MindSpore in Ascend 310 by Source Code Compilation

<!-- TOC -->

- [Installing MindSpore in Ascend 310 by Source Code Compilation](#installing-mindspore-in-ascend-310-by-source-code-compilation)
    - [Checking System Environment Information](#checking-system-environment-information)
    - [Downloading Source Code from the Code Repository](#downloading-source-code-from-the-code-repository)
    - [Building MindSpore](#building-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Verifying the Installation](#verifying-the-installation)
    - [Installing MindSpore Serving](#installing-mindspore-serving)

<!-- /TOC -->

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.2/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/install/mindspore_ascend310_install_source_en.md)

The following describes how to quickly install MindSpore by compiling the source code on Linux in the Ascend 310 environment, MindSpore in Ascend 310 only supports inference.

## Checking System Environment Information

- Ensure that the 64-bit Ubuntu 18.04, CentOS 7.6, or EulerOS 2.8 is installed.
- Ensure that right version [GCC 7.3.0](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) is installed.
- Ensure that [GMP 6.1.2](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz) is installed.
- Ensure that [Python 3.7.5](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) is installed.
- Ensure that [OpenSSL 1.1.1 or later](https://github.com/openssl/openssl.git) is installed.
    - After installation, set the environment variable `export OPENSSL_ROOT_DIR= "OpenSSL installation directory"`.
- Ensure that [CMake 3.18.3](https://cmake.org/download/) is installed.
    - After installation, add the path of CMake to the system environment variables.
- Ensure that [patch 2.5 or later](http://ftp.gnu.org/gnu/patch/) is installed.
    - After installation, add the patch path to the system environment variables.
- Ensure that [wheel 0.32.0 or later](https://pypi.org/project/wheel/) is installed.
- Ensure that the Ascend 310 AI Processor software packages ([Ascend Data Center Solution 21.0.1.SPC001](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-data-center-solution-pid-251167910/software/253384063?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252309113%7C251167910)) are installed.
    - For the installation of software package,  please refer to the [Product Document](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-data-center-solution-pid-251167910).
    - The software packages include Driver/Firmware and CANN.
        - [Driver and Firmware A300-3000 1.0.10.SPC012 ARM platform](https://support.huawei.com/enterprise/zh/ascend-computing/a300-3000-pid-250702915/software/253286767?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C250702915) or [Driver and Firmware A300-3010 1.0.10.SPC012 x86 platform](https://support.huawei.com/enterprise/zh/ascend-computing/a300-3010-pid-251560253/software/253286771?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251560253)
        - [CANN 5.0.1.SPC102](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/253348209?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373)
    - Ensure that you have permissions to access the installation path `/usr/local/Ascend` of the Ascend 310 AI Processor software package. If not, ask the user root to add you to a user group to which `/usr/local/Ascend` belongs. For details about the configuration, see the description document in the software package.
    - Install the .whl package provided with the Ascend 310 AI Processor software package. The .whl package is released with the software package. After the software package is upgraded, you need to reinstall the .whl package.

        ```bash
        pip install /usr/local/Ascend/ascend-toolkit/latest/atc/lib64/topi-{version}-py3-none-any.whl
        pip install /usr/local/Ascend/ascend-toolkit/latest/atc/lib64/te-{version}-py3-none-any.whl
        ```

- Ensure that the git tool is installed.
    If not, run the following command to download and install it:

    ```bash
    apt-get install git # ubuntu and so on
    yum install git     # centos and so on
    ```

## Downloading Source Code from the Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.2
```

## Building MindSpore

Run the following command in the root directory of the source code.

```bash
bash build.sh -e ascend -V 310
```

In the preceding information:

The default number of build threads is 8 in `build.sh`. If the compiler performance is poor, build errors may occur. You can add -j{Number of threads} to script to reduce the number of threads. For example, `bash build.sh -e ascend -V 310 -j4`.

## Installing MindSpore

```bash
chmod +x output/mindspore-ascend-{version}-cp37-cp37m-linux_{arch}.whl
pip install output/mindspore-ascend-{version}-cp37-cp37m-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

In the preceding information:

- When the network is connected, dependencies of the MindSpore installation package are automatically downloaded during the .whl package installation. For details about dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.2/requirements.txt). In other cases, install the dependencies by yourself.
- `{version}` specifies the MindSpore version number. For example, when installing MindSpore 1.1.0, set `{version}` to 1.1.0.
- `{arch}` specifies the system architecture. For example, if a Linux OS architecture is x86_64, set `{arch}` to `x86_64`. If the system architecture is ARM64, set `{arch}` to `aarch64`.

## Configuring Environment Variables

After MindSpore is installed, export runtime environment variables. In the following command, `/usr/local/Ascend` in `LOCAL_ASCEND=/usr/local/Ascend` indicates the installation path of the software package. Change it to the actual installation path.

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/acllib/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/atc/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# lib libraries that the mindspore depends on, modify "pip3" according to the actual situation
export LD_LIBRARY_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore/lib"}' | xargs realpath`:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/atc/ccec_compiler/bin/:${PATH}                       # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                       # Python library that TBE implementation depends on
```

## Verifying the Installation

Create a directory to store the sample code project, for example, `/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_single_op_sample`. You can obtain the code from the [official website](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/sample_resources/ascend310_single_op_sample.zip). A simple example of adding `[1, 2, 3, 4]` to `[2, 3, 4, 5]` is used and the code project directory structure is as follows:

```text

└─ascend310_single_op_sample
    ├── CMakeLists.txt                    // Build script
    ├── README.md                         // Usage description
    ├── main.cc                           // Main function
    └── tensor_add.mindir                 // MindIR model file
```

Go to the directory of the sample project and change the path based on the actual requirements.

```bash
cd /home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_single_op_sample
```

Build a project by referring to `README.md`, modify `pip3` according to the actual situation.

```bash
cmake . -DMINDSPORE_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`
make
```

After the build is successful, execute the case.

```bash
./tensor_add_sample
```

The following information is displayed:

```text
3
5
7
9
```

The preceding information indicates that MindSpore is successfully installed.

## Installing MindSpore Serving

If you want to quickly experience the MindSpore online inference service, you can install MindSpore Serving.

For details, see [MindSpore Serving](https://gitee.com/mindspore/serving/blob/r1.2/README.md).
