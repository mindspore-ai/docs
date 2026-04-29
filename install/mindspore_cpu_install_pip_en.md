# Installing MindSpore in CPU by pip

<!-- TOC -->

- [Installing MindSpore in CPU by pip](#installing-mindspore-in-cpu-by-pip)
    - [Installing MindSpore and dependencies](#installing-mindspore-and-dependencies)
        - [Installing Python](#installing-python)
        - [Installing GCC](#installing-gcc)
        - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/install/mindspore_cpu_install_pip_en.md)

This document describes how to install MindSpore by pip on Linux in a CPU environment. The following takes Ubuntu 18.04 as an example to describe how to install MindSpore.

## Installing MindSpore and dependencies

The following table lists the system environment and third-party dependencies required to install MindSpore.

|Software|Version|Description|
|-|-|-|
|Ubuntu|18.04|OS for running MindSpore|
|[Python](#installing-python)|3.9-3.12|Python environment that MindSpore depends|
|[GCC](#installing-gcc)|7.3.0-9.4.0|C++ compiler for compiling MindSpore|

The following describes how to install the third-party dependencies.

### Installing Python

[Python](https://www.python.org/) can be installed by Conda.

Install Miniconda:

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_26.1.1-1-Linux-$(arch).sh
bash Miniconda3-py310_26.1.1-1-Linux-$(arch).sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

After the installation is complete, you can set up Tsinghua source acceleration download for Conda, and see [here](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/).

Create a virtual environment, taking Python 3.10.20 as an example:

```bash
conda create -n mindspore_py310 python=3.10.20 -y
conda activate mindspore_py310
```

Run the following command to check the Python version.

```bash
python --version
```

### Installing GCC

The following takes GCC 7 as example for how to install GCC on common operating systems:

- On Ubuntu, run the following commands to install.

    ```bash
    sudo apt-get install gcc-7 g++-7 -y
    ```

- On CentOS, run the following commands to install.

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
    sudo yum install gcc g++ -y
    ```

### Installing MindSpore

First, refer to [Version List](https://www.mindspore.cn/versions) to select the version of MindSpore you want to install, and perform SHA-256 integrity check. Taking version 2.9.0 as an example, execute the following commands.

```bash
export MS_VERSION=2.9.0
```

Then run the following command to install MindSpore.

```bash
pip install mindspore==${MS_VERSION} -i https://repo.mindspore.cn/pypi/simple --trusted-host repo.mindspore.cn --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple/
```

When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://atomgit.com/mindspore/mindspore/blob/master/setup.py)). In other cases, you need to install dependency by yourself.

## Installation Verification

Execute the following command:

```bash
python -c "import mindspore;mindspore.set_device(device_target='CPU');mindspore.run_check()"
```

The outputs should be the same as:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
```

It means MindSpore has been installed successfully.

## Version Update

Use the following command if you need to update the MindSpore version:

```bash
pip install --upgrade mindspore=={version}
```

Of which,

- When updating to a release candidate (RC) version, set `{version}` to the RC version number, for example, 2.0.0.rc1. When updating to a stable release, you can remove `=={version}`.
