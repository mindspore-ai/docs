# Installing MindSpore in Ascend 910 by Source Code

<!-- TOC -->

- [Installing MindSpore in Ascend 910 by Source Code](#installing-mindspore-in-ascend-910-by-source-code)
    - [Environment Preparation (automatic, recommended)](#environment-preparation-automatic-recommended)
    - [Environment Preparation (manual)](#environment-preparation-manual)
        - [Installing Python](#installing-python)
        - [Installing Ascend AI processor software package](#installing-ascend-ai-processor-software-package)
        - [Installing wheel and setuptools](#installing-wheel-and-setuptools)
        - [Installing GCC](#installing-gcc)
        - [Installing git, gmp, tclsh, patch, NUMA and Flex](#installing-git-gmp-tclsh-patch-numa-and-flex)
        - [Installing git-lfs](#installing-git-lfs)
        - [Installing CMake](#installing-cmake)
        - [Installing Open MPI (Optional)](#installing-open-mpi-optional)
    - [Downloading the Source Code from the Code Repository](#downloading-the-source-code-from-the-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.7/install/mindspore_ascend_install_source_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.7/resource/_static/logo_source_en.png"></a>

This document describes how to quickly install MindSpore in a Linux system with an Ascend 910 environment by source code compilation.

- If you want to configure an environment that can compile MindSpore on an EulerOS 2.8 with Ascend AI processor software package installed, you may use [automatic installation script](https://gitee.com/mindspore/mindspore/raw/r1.7/scripts/install/euleros-ascend-source.sh) for one-click configuration, see [Environment Preparation (automatic, recommended)](#environment-preparation-automatic-recommended) section. The automatic installation script will install the dependencies required to compile MindSpore.

- If some dependencies, such as Python and GCC, have been installed in your system, it is recommended to install manually by referring to the installation steps in the [Environment Preparation (manual)](#environment-preparation-manual) section.

## Environment Preparation (automatic, recommended)

Before running the automatic installation script, you need to make sure that the Ascend AI processor software package is correctly installed on your system. If it is not installed, please refer to the section [Installing Ascend AI processor software package](#installing-ascend-ai-processor-software-package) to install it.

Run the following command to obtain and run the automatic installation script. The environment configured by the automatic installation script only supports compiling MindSpore>=1.6.0.

```bash
wget https://gitee.com/mindspore/mindspore/raw/r1.7/scripts/install/euleros-ascend-source.sh
# install Python 3.7 by default
# the default value of LOCAL_ASCEND is /usr/local/Ascend
bash -i ./euleros-ascend-source.sh
# to specify the Python 3.9 installation and install the optional dependencies Open MPI
# and set LOCAL_ASCEND to /home/xxx/Ascend, use the following manners
# LOCAL_ASCEND=/home/xxx/Ascend PYTHON_VERSION=3.9 OPENMPI=on bash -i ./euleros-ascend-source.sh
```

This script performs the following operations:

- Install the compilation dependencies required by MindSpore, such as GCC, CMake, etc.
- Install Python3 and pip3 and set them as default.
- Install Open MPI if OPENMPI is set to `on`.

After the automatic installation script is executed, you need to reopen the terminal window to make the environment variables take effect.

The automatic installation script creates a virtual environment named `mindspore_pyXX` for MindSpore. Where `XX` is the Python version, such as Python 3.7, the virtual environment name is `mindspore_py37`. Run the following command to show all virtual environments.

```bash
conda env list
```

To activate the virtual environment, take Python 3.7 as an example, execute the following command.

```bash
conda activate mindspore_py37
```

Now you can jump to the [Downloading the Source Code from the Code Repository](#downloading-the-source-code-from-the-code-repository) section to downloading and compiling MindSpore.

For more usage, see the script header description.

## Environment Preparation (manual)

The following table lists the system environment and third-party dependencies required for building and installing MindSpore.

|software|version|description|
|-|-|-|
|Ubuntu 18.04/CentOS 7.6/EulerOS 2.8/OpenEuler 20.03/KylinV10 SP1|-|OS for running MindSpore|
|[Python](#installing-python)|3.7-3.9|Python environment that MindSpore depends on|
|[Ascend AI processor software package](#installing-ascend-ai-processor-software-package)|-|Ascend platform AI computing library used by MindSpore|
|[wheel](#installing-wheel-and-setuptools)|0.32.0 or later|Python packaging tool used by MindSpore|
|[setuptools](#installing-wheel-and-setuptools)|44.0 or later|Python package management tool used by MindSpore|
|[GCC](#installing-gcc)|7.3.0|C++ compiler for compiling MindSpore|
|[git](#installing-git-gmp-tclsh-patch-numa-and-flex)|-|Source code management tool used by MindSpore|
|[git-lfs](#installing-git-lfs)|-|Source code management tool used by MindSpore|
|[CMake](#installing-cmake)|3.18.3 or later|Build tool for MindSpore|
|[gmp](#installing-git-gmp-tclsh-patch-numa-and-flex)|6.1.2|Multiple precision arithmetic library used by MindSpore|
|[Flex](#installing-git-gmp-tclsh-patch-numa-and-flex)|2.5.35 or later|lexical analyzer used by MindSpore|
|[tclsh](#installing-git-gmp-tclsh-patch-numa-and-flex)|-|MindSpore SQLite build dependency|
|[patch](#installing-git-gmp-tclsh-patch-numa-and-flex)|2.5 or later|Source code patching tool used by MindSpore|
|[NUMA](#installing-git-gmp-tclsh-patch-numa-and-flex)|2.0.11 or later|Non-uniform memory access library used by MindSpore|
|[Open MPI](#installing-open-mpi-optional)|4.0.3|High performance message passing library used by MindSpore (optional, required for single-node/multi-GPU and multi-node/multi-GPU training)|

The following describes how to install the third-party dependencies.

### Installing Python

[Python](https://www.python.org/) can be installed by Conda.

Install Miniconda:

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

After the installation is complete, you can set up Tsinghua source acceleration download for Conda, and see [here](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/).

Create a virtual environment, taking Python 3.7.5 as an example:

```bash
conda create -n mindspore_py37 python=3.7.5 -y
conda activate mindspore_py37
```

Run the following command to check the Python version.

```bash
python --version
```

### Installing Ascend AI processor software package

For detailed installation guide, please refer to [Ascend Data Center Solution 21.1.0 Installation Guide].

The default installation path of the installation package is `/usr/local/Ascend`. Ensure that the current user has the right to access the installation path `/usr/local/Ascend` of Ascend AI processor software package, If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.

Install the .whl packages provided in Ascend AI processor software package. The .whl packages are released with the software package. If the .whl packages have been installed before, you need to uninstall the packages by the following command.

```bash
pip uninstall te topi hccl -y
```

Run the following command to install the .whl packages in the default path. If the installation path is not the default path, you need to replace the path in the command with the installation path.

```bash
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-*-py3-none-any.whl
```

### Installing wheel and setuptools

After installing Python, use the following command to install it.

```bash
pip install wheel
pip install -U setuptools
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

### Installing git, gmp, tclsh, patch, NUMA and Flex

- On Ubuntu 18.04, run the following commands to install.

    ```bash
    sudo apt-get install git libgmp-dev tcl patch libnuma-dev flex -y
    ```

- On CentOS 7, EulerOS and OpenEuler, run the following commands to install.

    ```bash
    sudo yum install git gmp-devel tcl patch numactl-devel flex -y
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

- On EulerOS and OpenEuler, run the following commands to install.

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

- On Ubuntu 18.04, run the following commands to install [CMake](https://cmake.org/).

    ```bash
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
    sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
    sudo apt-get install cmake -y
    ```

- Other Linux systems can be installed with the following commands.

    Choose different download links based on the system architecture.

    ```bash
    # x86 run
    curl -O https://cmake.org/files/v3.19/cmake-3.19.8-Linux-x86_64.sh
    # aarch64 run
    curl -O https://cmake.org/files/v3.19/cmake-3.19.8-Linux-aarch64.sh
    ```

    Run the script to install CMake, which is installed in the `/usr/local` by default.

    ```bash
    sudo mkdir /usr/local/cmake-3.19.8
    sudo bash cmake-3.19.8-Linux-*.sh --prefix=/usr/local/cmake-3.19.8 --exclude-subdir
    ```

    Finally, add CMake to the `PATH` environment variable. Run the following commands if it is installed in the default path, other installation paths need to be modified accordingly.

    ```bash
    echo -e "export PATH=/usr/local/cmake-3.19.8/bin:\$PATH" >> ~/.bashrc
    source ~/.bashrc
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

## Downloading the Source Code from the Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.7
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
pip install output/mindspore_ascend-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

When the network is connected, dependencies of MindSpore are automatically downloaded during the .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/r1.7/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/r1.7/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.7/requirements.txt).

## Configuring Environment Variables

**If Ascend AI processor software is installed in a non-default path**, after MindSpore is installed, export Runtime-related environment variables. `/usr/local/Ascend` in the following command `LOCAL_ASCEND=/usr/local/Ascend` denotes the installation path of the software package, and you need to replace it as the actual installation path of the software package.

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
## AICPU path
export ASCEND_AICPU_PATH=${ASCEND_OPP_PATH}/..
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

The outputs should be the same as:

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

Use the following command if you need to update the MindSpore version.

- Update Online directly

    ```bash
    pip install --upgrade mindspore-ascend
    ```

- Update after source code compilation

    After successfully executing the compile script `build.sh` in the root path of the source code, find the whl package in path `output`, and use the following command to update your version.

    ```bash
    pip install --upgrade mindspore_ascend-*.whl
    ```
