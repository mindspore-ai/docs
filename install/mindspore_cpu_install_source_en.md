# Installing MindSpore in CPU by Source Code

<!-- TOC -->

- [Installing MindSpore in CPU by Source Code](#installing-mindspore-in-cpu-by-source-code)
    - [Environment Preparation](#environment-preparation)
        - [Installing Python](#installing-python)
        - [Installing wheel and setuptools](#installing-wheel-and-setuptools)
        - [Installing GCC, git, gmp, tclsh, patch and NUMA](#installing-gcc-git-gmp-tclsh-patch-and-numa)
        - [Installing CMake](#installing-cmake)
        - [Installing LLVM (Optional)](#installing-llvm-optional)
    - [Downloading the Source Code from the Code Repository](#downloading-the-source-code-from-the-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_source_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

This document describes how to quickly install MindSpore by source code in a Linux system in the CPU environment. The following takes Ubuntu 18.04 as an example to describe how to install MindSpore.

- If you need to configure an environment for building MindSpore on the Ubuntu 18.04 that never installed MindSpore and its dependencies, you may use the [automatic installation script](https://gitee.com/mindspore/mindspore/raw/master/scripts/install/ubuntu-cpu-source.sh) for one-click configuration. The script installs the dependencies required for building MindSpore.

    The root permission is required because the script will modify the source list and install dependencies via APT. Run the following command to obtain and run the automatic installation script:

    ```bash
    wget https://gitee.com/mindspore/mindspore/raw/master/scripts/install/ubuntu-cpu-source.sh
    # install Python 3.7 by default
    sudo bash ./ubuntu-cpu-source.sh
    # to specify Python version
    # sudo PYTHON_VERSION=3.9 bash ./ubuntu-cpu-source.sh
    ```

    This script performs the following operations:

    - Change the source list to HUAWEI CLOUD source.
    - Install the compilation dependencies required by MindSpore, such as GCC, CMake, etc.
    - Install Python3 and pip3 via APT and set them as default.

    For more usage, see the script header description.

- If some dependencies, such as Python and GCC, have been installed in your system, you are advised to perform the following steps to manually install MindSpore.

## Environment Preparation

The following table lists the system environment and third-party dependencies required for building and installing MindSpore.

|software|version|description|
|-|-|-|
|Ubuntu|18.04|OS for compiling and running MindSpore|
|[Python](#installing-python)|3.7.5 or 3.9.0|Python environment that MindSpore depends on|
|[wheel](#installing-wheel-and-setuptools)|0.32.0 or later|Python packaging tool used by MindSpore|
|[setuptools](#installing-wheel-and-setuptools)|44.0 or later|Python package management tool used by MindSpore|
|[GCC](#installing-gcc-git-gmp-tclsh-patch-and-numa)|7.3.0|C++ compiler for compiling MindSpore|
|[git](#installing-gcc-git-gmp-tclsh-patch-and-numa)|-|Source code management tools used by MindSpore|
|[CMake](#installing-cmake)|3.18.3 or later|Build tools for MindSpore|
|[gmp](#installing-gcc-git-gmp-tclsh-patch-and-numa)|6.1.2|Multiple precision arithmetic library used by MindSpore|
|[tclsh](#installing-gcc-git-gmp-tclsh-patch-and-numa)|-|MindSpore SQLite build dependency|
|[patch](#installing-gcc-git-gmp-tclsh-patch-and-numa)|2.5 or later|Source code patching tool used by MindSpore|
|[NUMA](#installing-gcc-git-gmp-tclsh-patch-and-numa)|2.0.11 or later|Non-uniform memory access library used by MindSpore|
|[LLVM](#installing-llvm-optional)|12.0.1|Compiler framework used by MindSpore (optional, mandatory for graph kernel fusion)|

The following describes how to install the third-party dependencies.

### Installing Python

[Python](https://www.python.org/) can be installed in multiple ways.

- Install Python with Conda.

    Install Miniconda:

    ```bash
    cd /tmp
    curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
    bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b
    cd -
    . ~/miniconda3/etc/profile.d/conda.sh
    conda init bash
    ```

    Create a Python 3.7.5 environment:

    ```bash
    conda create -n mindspore_py37 python=3.7.5 -y
    conda activate mindspore_py37
    ```

    Or create a Python 3.9.0 environment:

    ```bash
    conda create -n mindspore_py39 python=3.9.0 -y
    conda activate mindspore_py39
    ```

- Or install Python via APT with the following command.

    ```bash
    sudo apt-get update
    sudo apt-get install software-properties-common -y
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get install python3.7 python3.7-dev python3.7-distutils python3-pip -y
    # set new installed Python as default
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 100
    # install pip
    python -m pip install pip -i https://repo.huaweicloud.com/repository/pypi/simple
    sudo update-alternatives --install /usr/bin/pip pip ~/.local/bin/pip3.7 100
    pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
    ```

    To install Python 3.9, just replace `3.7` with `3.9` in the command.

Run the following command to check the Python version.

```bash
python --version
```

### Installing wheel and setuptools

After installing Python, run the following command to install them.

```bash
pip install wheel
pip install -U setuptools
```

### Installing GCC, git, gmp, tclsh, patch and NUMA

Run the following commands to install GCC, git, gmp, tclsh, patch and NUMA.

```bash
sudo apt-get install gcc-7 git libgmp-dev tcl patch libnuma-dev -y
```

### Installing CMake

Run the following command to install [CMake](https://cmake.org/).

```bash
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt-get install cmake -y
```

### Installing LLVM (optional)

Run the following command to install [LLVM](https://llvm.org/).

```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-12 main"
sudo apt-get update
sudo apt-get install llvm-12-dev -y
```

## Downloading the Source Code from the Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git
```

## Compiling MindSpore

Go to the root directory of mindspore, then run the build script.

```bash
cd mindspore
bash build.sh -e cpu -j4 -S on
```

Where:

- If the compiler performance is good, add `-j{Number of threads}` to increase the number of threads. For example, `bash build.sh -e cpu -j12`.
- By default, the dependent source code is downloaded from GitHub. When `-S` is set to `on`, the source code is downloaded from the corresponding Gitee image.
- For details about how to use `build.sh`, see the script header description.

## Installing MindSpore

```bash
pip install output/mindspore-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

When the network is connected, dependencies of MindSpore are automatically downloaded during the .whl package installation. For details about dependencies, see required_package in the [setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py). In other cases, install the dependencies by yourself. When running a model, you need to install additional dependencies based on the requirements.txt file specified by different models in the [ModelZoo](https://gitee.com/mindspore/models/tree/master/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt).

## Installation Verification

```bash
python -c "import mindspore;mindspore.run_check()"
```

The output should be like:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

It means MindSpore has been installed successfully.

## Version Update

Using the following command if you need to update the MindSpore version:

- Update online

    ```bash
    pip install --upgrade mindspore
    ```

- Update after source code compilation

    After successfully executing the compile script `build.sh` in the root path of the source code, find the whl package in path `output`, use the following command to update your version.

    ```bash
    pip install --upgrade mindspore-*.whl
    ```
