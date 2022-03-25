# Installing MindSpore Ascend 910 by Conda

<!-- TOC -->

- [Installing MindSpore Ascend 910 by Conda](#installing-mindspore-ascend-910-by-conda)
    - [Installing Environment Dependencies](#installing-environment-dependencies)
        - [Installing Ascend AI processor software package](#installing-ascend-ai-processor-software-package)
        - [Installing Conda](#installing-conda)
        - [Installing GCC](#installing-gcc)
        - [Installing gmp](#installing-gmp)
        - [Installing Open MPI (Optional)](#installing-open-mpi-optional)
    - [Creating and Accessing the Conda Virtual Environment](#creating-and-accessing-the-conda-virtual-environment)
    - [Installing MindSpore](#installing-mindspore)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend_install_conda_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

[Conda](https://docs.conda.io/en/latest/) is an open-source, cross-platform, language-agnostic package manager and environment management system. It allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.

This document describes how to quickly install MindSpore in a Linux system with an Ascend 910 environment by Conda.

- If you want to install MindSpore by Conda on a EulerOS 2.8 with Ascend AI processor software package installed, you may use [automatic installation script](https://gitee.com/mindspore/mindspore/raw/master/scripts/install/euleros-ascend-conda.sh) for one-click installation. The automatic installation script will install the dependencies required to compile MindSpore.

    Run the following command to obtain and run the automatic installation script:

    ```bash
    wget https://gitee.com/mindspore/mindspore/raw/master/scripts/install/euleros-ascend-conda.sh
    # install Python 3.7 and the latest MindSpore by default
    bash -i ./euleros-ascend-conda.sh
    # to specify Python and MindSpore version, e.g. Python 3.9 and MindSpore 1.5.0
    # PYTHON_VERSION=3.9 MINDSPORE_VERSION=1.5.0 bash -i ./euleros-ascend-conda.sh
    ```

    This script performs the following operations:

    - Install the dependencies required by MindSpore, such as GCC, gmp.
    - Install Conda and create a virtual environment for MindSpore.
    - Install MindSpore Ascend by Conda.
    - Install Open MPI if OPENMPI is set to `on`.

    After the script is executed, please refer to the instructions in [Configuring Environment Variables](#configuring-environment-variables) to set relevant environment variables.

    For more usage, see the script header description.

- If some dependencies, such as Python and GCC, have been installed in your system, you are advised to perform the following steps to manually install MindSpore.

## Installing Environment Dependencies

The following table lists the system environment and third-party dependencies required for installing MindSpore.

|software|version|description|
|-|-|-|
|Ubuntu 18.04/CentOS 7.6/EulerOS 2.8/OpenEuler 20.03/KylinV10 SP1|-|OS for running MindSpore|
|[Ascend AI processor software package](#installing-ascend-ai-processor-software-package)|-|Ascend platform AI computing library used by MindSpore|
|[Conda](#installing-conda)|Anaconda3 or Miniconda3|Python environment management tool|
|[GCC](#installing-gcc)|7.3.0|C++ compiler for compiling MindSpore|
|[gmp](#installing-gmp)|6.1.2|Multiple precision arithmetic library used by MindSpore|
|[Open MPI](#installing-open-mpi-optional)|4.0.3|high performance message passing library used by MindSpore (optional, required for single-node/multi-GPU and multi-node/multi-GPU training)|

The following describes how to install the third-party dependencies.

### Installing Ascend AI processor software package

For detailed installation guide, please refer to [Ascend Data Center Solution 21.0.4 Installation Guide](https://support.huawei.com/enterprise/zh/doc/EDOC1100235797?section=j003).

The default installation path of the installation package is `/usr/local/Ascend`. Ensure that the current user has the right to access the installation path `/usr/local/Ascend` of Ascend AI processor software package, If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.

Install the .whl packages provided in Ascend AI processor software package. The .whl packages are released with the software package. If the .whl packages have been installed before, you need to uninstall the packages by the following command.

```bash
pip uninstall te topi hccl -y
```

Run the following command to install the .whl packages if the Ascend AI package has been installed in default path. If the installation path is not the default path, you need to replace the path in the command with the installation path.

```bash
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-*-py3-none-any.whl
```

### Installing Conda

Run the following command to install Miniconda.

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

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

- On EulerOS and OpenEuler, run the following commands to install.

    ```bash
    sudo yum install gcc -y
    ```

### Installing gmp

- On Ubuntu 18.04, run the following commands to install.

    ```bash
    sudo apt-get install libgmp-dev -y
    ```

- On CentOS 7, EulerOS and OpenEuler, run the following commands to install.

    ```bash
    sudo yum install gmp-devel -y
    ```

### Installing Open MPI (optional)

Run the following command to compile and install [Open MPI](https://www.open-mpi.org/).

```bash
curl -O https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz
tar xzf openmpi-4.0.3.tar.gz
cd openmpi-4.0.3
./configure --prefix=/usr/local/openmpi-4.0.3
make
sudo make install
echo -e "export PATH=/usr/local/openmpi-4.0.3/bin:\$PATH" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=/usr/local/openmpi-4.0.3/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
cd -
```

## Creating and Accessing the Conda Virtual Environment

Create a Conda virtual environment based on the Python version you want to use and activate the virtual environment.

If you want to use Python 3.7.5:

```bash
conda create -n mindspore_py37 python=3.7.5 -y
conda activate mindspore_py37
```

If you want to use Python 3.9.0:

```bash
conda create -n mindspore_py39 python=3.9.0 -y
conda activate mindspore_py39
```

Install the .whl package provided with the Ascend AI Processor software package in the virtual environment. The .whl package is released with the software package. After the software package is upgraded, you need to reinstall the .whl package.

```bash
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-{version}-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-{version}-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-{version}-py3-none-any.whl
```

If the software package of the Ascend AI processor is upgraded, the matching .whl package also needs to be reinstalled. Uninstall the original installation package and reinstall the .whl package by referring to the preceding commands.

```bash
pip uninstall te topi hccl -y
```

## Installing MindSpore

Ensure that you are in the Conda virtual environment and run the following command to install the latest MindSpore. To install other versions, please refer to [Version List](https://www.mindspore.cn/versions) and specify the version after `mindspore-ascend=`.

```bash
conda install mindspore-ascend -c mindspore -c conda-forge
```

When the network is connected, dependencies of MindSpore are automatically downloaded during the .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/master/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt).

## Configuring Environment Variables

**If Ascend AI processor software is installed in a non-default path**, after MindSpore is installed, export runtime-related environment variables. `/usr/local/Ascend` in the following command `LOCAL_ASCEND=/usr/local/Ascend` denotes the installation path of the software package, please replace it as your actual installation path.

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
## TBE operator implementation tool path
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
## OPP path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp
## TBE operator compilation tool path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}
## Python library that TBE implementation depends on
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}
```

## Installation Verification

i:

```bash
python -c "import mindspore;mindspore.run_check()"
```

The output should be like:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

It means MindSpore has been installed successfully.

ii:

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="Ascend")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
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

Using the following command if you need to update the MindSpore version:

```bash
conda update mindspore-ascend -c mindspore -c conda-forge
```
