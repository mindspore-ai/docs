# Installing MindSpore in Ascend 310 by binary package

<!-- TOC -->

- [Installing MindSpore in Ascend 310 by binary package](#installing-mindspore-in-ascend-310-by-binary-package)
    - [Checking System Environment Information](#checking-system-environment-information)
    - [Installing MindSpore](#installing-mindspore)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Verifying the Installation](#verifying-the-installation)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend310_install_binary_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

The following describes how to quickly install MindSpore by pip on Linux in the Ascend 310 environment, MindSpore in Ascend 310 only supports inference.

## Checking System Environment Information

- Ensure that the 64-bit operating system is installed and the [glibc](https://www.gnu.org/software/libc/)>=2.17, where Ubuntu 18.04/CentOS 7.6/EulerOS 2.8 are verified.

- Ensure that right version [GCC 7.3.0](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) is installed.

- Ensure that [GMP 6.1.2](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz) is installed.

- Ensure that [CMake 3.18.3 or later](https://cmake.org/download/) is installed.
    - After installation, add the path of CMake to the system environment variables.

- Ensure that Python 3.7.5 or 3.9.0 is installed. If not installed, download and install Python from:
    - Python 3.7.5 (64-bit): [Python official website](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) or [HUAWEI CLOUD](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz).
    - Python 3.9.0 (64-bit): [Python official website](https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz) or [HUAWEI CLOUD](https://mirrors.huaweicloud.com/python/3.9.0/Python-3.9.0.tgz).
- Ensure that the Ascend 310 AI Processor software packages ([Ascend Data Center Solution 21.0.5]) are installed.
    - For the installation of software package,  please refer to the [Product Document](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-data-center-solution-pid-251167910).
    - The software packages include Driver and Firmware and CANN.
        - [Driver and Firmware A300-3000 1.0.13 ARM platform] and [Driver and Firmware A300-3010 1.0.13 x86 platform]
        - [CANN 5.0.T306](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/254156433)
    - Ensure that you have permissions to access the installation path `/usr/local/Ascend` of the Ascend 310 AI Processor software package. If not, ask the user root to add you to a user group to which `/usr/local/Ascend` belongs. For details about the configuration, see the description document in the software package.
    - Install the .whl package provided with the Ascend 310 AI Processor software package. The .whl package is released with the software package. After the software package is upgraded, you need to reinstall the .whl package.

        ```bash
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-{version}-py3-none-any.whl
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-{version}-py3-none-any.whl
        ```

## Installing MindSpore

It is recommended to refer to [Version List](https://www.mindspore.cn/versions/en) to perform SHA-256 integrity verification, and then execute the following command to install MindSpore after the verification is consistent.

```bash
wget https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/ascend/{arch}/mindspore_ascend-{version}-linux_{arch}.tar.gz --no-check-certificate
tar -zxf mindspore_ascend-{version}-linux_{arch}.tar.gz
```

In the preceding information:

- `{version}` specifies the MindSpore version number. For example, when installing MindSpore 1.5.0, set `{version}` to 1.5.0, when installing MindSpore 1.5.0-rc1, the first `{version}` which represents download path should be written as 1.5.0-rc1, and the second `{version}` which represents file name should be 1.5.0rc1.
- `{arch}` specifies the system architecture. For example, if a Linux OS architecture is x86_64, set `{arch}` to `x86_64`. If the system architecture is ARM64, set `{arch}` to `aarch64`.

## Configuring Environment Variables

After MindSpore is installed, export runtime environment variables. In the following command, `/usr/local/Ascend` in `LOCAL_ASCEND=/usr/local/Ascend` indicates the installation path of the software package. Change it to the actual installation path.

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}                 # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                       # Python library that TBE implementation depends on

# Set path to extracted MindSpore accordingly
export LD_LIBRARY_PATH={mindspore_path}:${LD_LIBRARY_PATH}
```

In the preceding information:

- `{mindspore_path}` specifies the absolute path to which MindSpore package is extracted.

## Verifying the Installation

Create a directory to store the sample code project, for example, `/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_single_op_sample`. You can obtain the code from the [official website](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/sample_resources/ascend310_single_op_sample.zip). A simple example of adding `[1, 2, 3, 4]` to `[2, 3, 4, 5]` is used and the code project directory structure is as follows:

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

Build a project by referring to `README.md`, modify`{mindspore_path}`which specifies the absolute path to MindSpore.

```bash
cmake . -DMINDSPORE_PATH={mindspore_path}
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

