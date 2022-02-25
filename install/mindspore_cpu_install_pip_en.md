# Installing MindSpore in CPU by pip

<!-- TOC -->

- [Installing MindSpore in CPU by pip](#installing-mindspore-in-cpu-by-pip)
    - [Environment Preparation](#environment-preparation)
    - [Downloading and Installing MindSpore](#downloading-and-installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_pip_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

This document describes how to quickly install MindSpore by pip in a Linux system with a CPU environment. The following takes Ubuntu 18.04 as an example to illustrate the steps to install MindSpore.

- If you want to install MindSpore by pip on a fresh Ubuntu 18.04, you may use [automatic installation script](https://gitee.com/mindspore/mindspore/raw/master/scripts/install/ubuntu-cpu-pip.sh) for one-click installation. The automatic installation script will install MindSpore and its dependencies.

    The automatic installation script needs to replace the source list and install dependencies via APT, so it needs root privileges to execute. Use the following commands to get the automatic installation script and execute.

    ```bash
    wget https://gitee.com/mindspore/mindspore/raw/master/scripts/install/ubuntu-cpu-pip.sh
    # install Python 3.7 and MindSpore 1.6.0 by default
    sudo bash ./ubuntu-cpu-pip.sh
    # to specify Python and MindSpore version, e.g. Python 3.9 and MindSpore 1.5.0
    # sudo PYTHON_VERSION=3.9 MINDSPORE_VERSION=1.5.0 bash ./ubuntu-cpu-pip.sh
    ```

    The script will:

    - Set the source list to huaweicloud source.
    - Install the compilation dependencies required by MindSpore, such as GCC, gmp.
    - Install Python3 and pip3 via APT and set them as default.
    - Install MindSpore in CPU by pip.

    For more usage, please refer to the description at the head of the script.

- If your system has already installed some dependencies, such as Python, GCC, etc., it is recommended to install manually by referring to the following installation steps.

## Environment Preparation

The following table lists the system environment and third-party dependencies required to install MindSpore.

|software|version|description|
|-|-|-|
|Ubuntu|18.04|operating system for compiling and running MindSpore|
|[Python](#install-python)|3.7.5 or 3.9.0|Python interface for MindSpore|
|[GCC](#install-gccgmp)|7.3.0|C++ compiler for compiling MindSpore|
|[gmp](#install-gccgmp)|6.1.2|multiple precision arithmetic library used by MindSpore|

The following are installation methods of third-party dependencies.

### Install Python

[Python](https://www.python.org/) can be installed in several ways.

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

You can check the Python version with the following command.

```bash
python --version
```

### Install GCC/gmp

You may install GCC and gmp by the following commands.

```bash
sudo apt-get install gcc-7 libgmp-dev -y
```

## Downloading and Installing MindSpore

It is recommended to refer to [Version List](https://www.mindspore.cn/versions/en) to perform SHA-256 integrity verification, and then run the following command to install MindSpore 1.6.0 according to the system architecture and Python version.

If you need to install other versions of MindSpore, please refer to [Version List](https://www.mindspore.cn/versions) to obtain the corresponding WHL package for installation.

```bash
# x86_64/Python3.7
# SHA-256: b4fe66629150c47397722057c32c806cd3eece5e158a93c62cac0bc03b464e3f
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.0/MindSpore/cpu/x86_64/mindspore-1.6.0-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# x86_64/Python3.9
# SHA-256: 87151059854e7388c6b5d6d56a7b75087f171efdcd84896c61d0e9088fcebcc0
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.0/MindSpore/cpu/x86_64/mindspore-1.6.0-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64/Python3.7
# SHA-256: 5dc6b9abe668d53960773d6e8ac4dc6f7016c15cce5c9480045c74a3e456e40f
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.0/MindSpore/cpu/aarch64/mindspore-1.6.0-cp37-cp37m-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64/Python3.9
# SHA-256: 4a8f49587b30b6a0413edba85ebbae07600fc84f8a1fa48c42a2359402bfc852
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.0/MindSpore/cpu/aarch64/mindspore-1.6.0-cp39-cp39-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/master/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt).

## Installation Verification

```bash
python -c "import mindspore;mindspore.run_check()"
```

The outputs should be the same as:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

It means MindSpore has been installed successfully.

## Version Update

Using the following command if you need to update the MindSpore version:

```bash
pip install --upgrade mindspore=={version}
```

Of which,

- When updating to a release candidate (rc) version, `{version}` should be specified, e.g. 1.5.0rc1; When updating to a standard release, `=={version}` could be removed.
