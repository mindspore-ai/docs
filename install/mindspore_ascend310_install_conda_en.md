# Installing MindSpore Ascend 310 by Conda

<!-- TOC -->

- [Installing MindSpore Ascend 310 by Conda](#installing-mindspore-ascend-310-by-conda)
    - [Checking System Environment Information](#checking-system-environment-information)
    - [Creating and Accessing the Conda Virtual Environment](#creating-and-accessing-the-conda-virtual-environment)
    - [Installing MindSpore](#installing-mindspore)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Verifying the Installation](#verifying-the-installation)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend310_install_conda_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

[Conda](https://docs.conda.io/en/latest/) is an open-source, cross-platform, language-agnostic package manager and environment management system. It allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.

The following describes how to quickly install MindSpore by Conda on Linux in the Ascend 310 environment, MindSpore in Ascend 310 only supports inference.

## Checking System Environment Information

- Ensure that the 64-bit operating system is installed and the [glibc](https://www.gnu.org/software/libc/)>=2.17, where Ubuntu 18.04/CentOS 7.6/EulerOS 2.8 are verified.
- Ensure that right version [GCC 7.3.0](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) is installed.
- Ensure that [GMP 6.1.2](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz) is installed.

- Ensure that the Conda version is compatible with the current system.

    - If you prefer the complete capabilities provided by Conda, you can choose to download [Anaconda3](https://repo.anaconda.com/archive/).
    - If you want to save disk space or prefer custom Conda installation, you can choose to download [Miniconda3](https://repo.anaconda.com/miniconda/).

- Ensure that the Ascend 310 AI processor software package (Ascend Data Center Solution 21.0.3) are installed.

    - For the installation of software package,  please refer to the [CANN Software Installation Guide](https://support.huawei.com/enterprise/en/doc/EDOC1100219213).
    - The software packages include Driver and Firmware and CANN.
        - [A300-3000 1.0.12 ARM platform](https://support.huawei.com/enterprise/zh/ascend-computing/a300-3000-pid-250702915/software/253845265) or [A300-3010 1.0.12 x86 platform](https://support.huawei.com/enterprise/zh/ascend-computing/a300-3010-pid-251560253/software/253845285)
        - [CANN 5.0.3](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/252806307)

    - Ensure that you have permissions to access the installation path `/usr/local/Ascend` of the Ascend 310 AI Processor software package. If not, ask the user root to add you to a user group to which `/usr/local/Ascend` belongs. For details about the configuration, see the description document in the software package.

## Creating and Accessing the Conda Virtual Environment

Create a Conda virtual environment based on the Python version you want to use and go to the virtual environment.
If you want to use Python 3.7.5:

```bash
conda create -n mindspore_py37 -c conda-forge python=3.7.5
conda activate mindspore_py37
```

If you want to use Python 3.9.0:

```bash
conda create -n mindspore_py39 -c conda-forge python=3.9.0
conda activate mindspore_py39
```

Install the .whl package provided with the Ascend 310 AI Processor software package in the virtual environment. The .whl package is released with the software package. After the software package is upgraded, you need to reinstall the .whl package.

```bash
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-{version}-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-{version}-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-{version}-py3-none-any.whl
```

## Installing MindSpore

Execute the following command to install MindSpore.

```bash
conda install mindspore-ascend={version} -c mindspore -c conda-forge
```

In the preceding information:

- When the network is connected, dependencies of the mindspore installation package are automatically downloaded during the .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/master/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt).
- `{version}` specifies the MindSpore version number. For example, when installing MindSpore 1.5.0-rc1, set `{version}` to 1.5.0rc1.

## Configuring Environment Variables

After MindSpore is installed, export runtime environment variables. In the following command, `/usr/local/Ascend` in `LOCAL_ASCEND=/usr/local/Ascend` indicates the installation path of the software package. Change it to the actual installation path.

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# lib libraries that the mindspore depends on, modify "pip3" according to the actual situation
export LD_LIBRARY_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore/lib"}' | xargs realpath`:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}                 # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                       # Python library that TBE implementation depends on
```

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

## Version Update

Using the following command if you need to update the MindSpore version:

```bash
conda update mindspore-ascend -c mindspore -c conda-forge
```
