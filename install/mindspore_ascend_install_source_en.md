# Installing MindSpore in Ascend by Source Code

<!-- TOC -->

- [Installing MindSpore in Ascend by Source Code](#installing-mindspore-in-ascend-by-source-code)
    - [Installing dependencies](#installing-dependencies)
        - [Installing Python](#installing-python)
        - [Installing Ascend AI processor software package](#installing-ascend-ai-processor-software-package)
        - [Installing wheel setuptools PyYAML and Numpy](#installing-wheel-setuptools-pyyaml-and-numpy)
        - [Installing GCC](#installing-gcc)
        - [Installing git tclsh patch NUMA and Flex](#installing-git-tclsh-patch-numa-and-flex)
        - [Installing git-lfs](#installing-git-lfs)
        - [Installing CMake](#installing-cmake)
    - [Downloading the Source Code from the Code Repository](#downloading-the-source-code-from-the-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.5.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.5.0/install/mindspore_ascend_install_source_en.md)

This document describes how to install MindSpore by compiling source code on Linux in an Ascend environment.

## Installing dependencies

The following table lists the system environment and third-party dependencies required for building and installing MindSpore.

|Software|Version|Description|
|-|-|-|
|Ubuntu 18.04/CentOS 7.6/EulerOS 2.8/openEuler 20.03/KylinV10 SP1|-|OS for running MindSpore|
|[Python](#installing-python)|3.9-3.11|Python environment that MindSpore depends on|
|[Ascend AI processor software package](#installing-ascend-ai-processor-software-package)|-|Ascend platform AI computing library used by MindSpore|
|[wheel](#installing-wheel-setuptools-pyyaml-and-numpy)|0.32.0 or later|Python packaging tool used by MindSpore|
|[setuptools](#installing-wheel-setuptools-pyyaml-and-numpy)|44.0 or later|Python package management tool used by MindSpore|
|[PyYAML](#installing-wheel-setuptools-pyyaml-and-numpy)|6.0-6.0.2|PyYAML module that operator compliation in MindSpore depends on|
|[Numpy](#installing-wheel-setuptools-pyyaml-and-numpy)|1.19.3-1.26.4|Numpy module that Numpy-related functions in MindSpore depends on|
|[GCC](#installing-gcc)|7.3.0|C++ compiler for compiling MindSpore|
|[git](#installing-git-tclsh-patch-numa-and-flex)|-|Source code management tool used by MindSpore|
|[git-lfs](#installing-git-lfs)|-|Source code management tool used by MindSpore|
|[CMake](#installing-cmake)|3.22.2 or later|Build tool for MindSpore|
|[Flex](#installing-git-tclsh-patch-numa-and-flex)|2.5.35 or later|lexical analyzer used by MindSpore|
|[tclsh](#installing-git-tclsh-patch-numa-and-flex)|-|MindSpore SQLite build dependency|
|[patch](#installing-git-tclsh-patch-numa-and-flex)|2.5 or later|Source code patching tool used by MindSpore|
|[NUMA](#installing-git-tclsh-patch-numa-and-flex)|2.0.11 or later|Non-uniform memory access library used by MindSpore|

The following describes how to install the third-party dependencies.

### Installing Python

[Python](https://www.python.org/) can be installed by Conda.

Install Miniconda:

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-$(arch).sh
bash Miniconda3-py37_4.10.3-Linux-$(arch).sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

After the installation is complete, you can set up Tsinghua source acceleration download for Conda, and see [here](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/).

Create a virtual environment, taking Python 3.9.11 as an example:

```bash
conda create -n mindspore_py39 python=3.9.11 -y
conda activate mindspore_py39
```

Run the following command to check the Python version.

```bash
python --version
```

### Installing Ascend AI processor software package

Ascend software package provides two distributions, commercial edition and community edition:

- Commercial edition needs approval from Ascend to download, for detailed installation guide, please refer to [Ascend Training Solution 24.0.0](https://support.huawei.com/enterprise/zh/doc/EDOC1100441839).

- Community edition has no restrictions, the recommended version is `8.0.0.beta1` in [CANN community edition](https://www.hiascend.com/developer/download/community/result?module=cann), then choose relevant driver and firmware packages in [firmware and driver](https://www.hiascend.com/hardware/firmware-drivers/community). Please refer to the abovementioned commercial edition installation guide to choose which packages are to be installed and how to install them.

The default installation path of the installation package is `/usr/local/Ascend`. Ensure that the current user has the right to access the installation path `/usr/local/Ascend` of Ascend AI processor software package. If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.

Install the .whl packages provided in Ascend AI processor software package. If the .whl packages have been installed before, you should uninstall the .whl packages by running the following command.

```bash
pip uninstall te topi hccl -y
```

Run the following command to install the .whl packages in the default path. If the installation path is not the default path, you need to replace the path in the command with the installation path.

```bash
pip install sympy
pip install protobuf
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-*-py3-none-any.whl
```

### Installing wheel setuptools PyYAML and Numpy

After installing Python, use the following command to install it.

```bash
pip install wheel
pip install -U setuptools
pip install pyyaml
pip install "numpy>=1.19.3,<=1.26.4"
```

The Numpy version used in the runtime environment must be no less than the Numpy version in the compilation environment to ensure the normal use of Numpy related capabilities in the framework.

### Installing GCC

- On Ubuntu 18.04, run the following commands to install.

    ```bash
    sudo apt-get install gcc-7 -y
    ```

- On CentOS 7, run the following commands to install.

    ```bash
    sudo yum install centos-release-scl
    sudo yum install devtoolset-7
    ```

    After installation, run the following commands to switch to GCC 7.

    ```bash
    scl enable devtoolset-7 bash
    ```

- On EulerOS and openEuler, run the following commands to install.

    ```bash
    sudo yum install gcc -y
    ```

### Installing git tclsh patch NUMA and Flex

- On Ubuntu 18.04, run the following commands to install.

    ```bash
    sudo apt-get install git tcl patch libnuma-dev flex -y
    ```

- On CentOS 7, EulerOS and openEuler, run the following commands to install.

    ```bash
    sudo yum install git tcl patch numactl-devel flex -y
    ```

### Installing git-lfs

- On Ubuntu, run the following commands to install.

    ```bash
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs -y
    git lfs install
    ```

- On CentOS 7, run the following commands to install.

    ```bash
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
    sudo yum install git-lfs -y
    git lfs install
    ```

- On EulerOS and openEuler, run the following commands to install.

    Select the appropriate version to download according to the system architecture.

    ```bash
    # x64
    curl -OL https://github.com/git-lfs/git-lfs/releases/download/v3.1.2/git-lfs-linux-amd64-v3.1.2.tar.gz
    # arm64
    curl -OL https://github.com/git-lfs/git-lfs/releases/download/v3.1.2/git-lfs-linux-arm64-v3.1.2.tar.gz
    ```

    Unzip and install.

    ```bash
    mkdir git-lfs
    tar xf git-lfs-linux-*-v3.1.2.tar.gz -C git-lfs
    cd git-lfs
    sudo bash install.sh
    ```

### Installing CMake

- Install CMake with the following commands.

    Choose different download links based on the system architecture.

    ```bash
    # x86 run
    curl -O https://cmake.org/files/v3.22/cmake-3.22.2-linux-x86_64.sh
    # aarch64 run
    curl -O https://cmake.org/files/v3.22/cmake-3.22.2-linux-aarch64.sh
    ```

    Run the script to install CMake, which is installed in the `/usr/local` by default.

    ```bash
    sudo mkdir /usr/local/cmake-3.22.2
    sudo bash cmake-3.22.2-linux-*.sh --prefix=/usr/local/cmake-3.22.2 --exclude-subdir
    ```

    Finally, add CMake to the `PATH` environment variable. Run the following commands if it is installed in the default path, other installation paths need to be modified accordingly.

    ```bash
    echo -e "export PATH=/usr/local/cmake-3.22.2/bin:\$PATH" >> ~/.bashrc
    source ~/.bashrc
    ```

## Downloading the Source Code from the Code Repository

```bash
git clone -b v2.5.0 https://gitee.com/mindspore/mindspore.git
```

## Configuring Environment Variables

**If Ascend AI processor software is installed in a non-default path**, before compiling and after installing MindSpore, export Runtime-related environment variables. `/usr/local/Ascend` in the following command `LOCAL_ASCEND=/usr/local/Ascend` denotes the installation path of the software package, and you need to replace it as the actual installation path of the software package.

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# environment variables
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# set environmet variables using script provided by CANN, swap "ascend-toolkit" with "nnae" if you are using CANN-nnae package instead
source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_PATH=${LOCAL_ASCEND}/ascend-toolkit/
```

## Compiling MindSpore

Go to the root directory of mindspore, then run the build script.

```bash
cd mindspore
bash build.sh -e ascend -S on
```

Where:

- In the `build.sh` script, the default number of compilation threads is 8. If the compiler performance is poor, compilation errors may occur. You can add -j{Number of threads} in to script to reduce the number of threads, for example, `bash build.sh -e ascend -j4`.
- By default, the dependent source code is downloaded from gitHub. When -S is set to `on`, the source code is downloaded from the corresponding gitee image.
- For details about how to use `build.sh`, see the script header description.

## Installing MindSpore

```bash
pip install output/mindspore-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

When the network is connected, dependencies of MindSpore are automatically downloaded during the .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/v2.5.0/setup.py).) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/master/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/v2.5.0/requirements.txt).

## Installation Verification

i:

```bash
python -c "import mindspore;mindspore.set_device('Ascend');mindspore.run_check()"
```

The outputs should be the same as:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!
```

It means MindSpore has been installed successfully.

ii:

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_device("Ascend")
x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
```

The outputs should be the same as:

```text
[[[[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]]]
```

It means MindSpore has been installed successfully.

## Version Update

After successfully executing the compile script `build.sh` in the root path of the source code, find the whl package in path `output`, and use the following command to update your version.

When upgrading from MindSpore 1.x to MindSpore 2.x, you need to manually uninstall the old version first:

```bash
pip uninstall mindspore-ascend
```

Then install MindSpore 2.x:

```bash
pip install mindspore*.whl
```

When upgrading from MindSpore 2.x:

```bash
pip install --upgrade mindspore*.whl
```
