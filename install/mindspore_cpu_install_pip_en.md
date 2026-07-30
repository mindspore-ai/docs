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
|[Python](#installing-python)|3.10-3.12|Python environment that MindSpore depends|
|[GCC](#installing-gcc)|9.5.0-11.3.0 (preferred version 9.5.0)|C++ compiler for compiling MindSpore|

The following describes how to install the third-party dependencies.

### Installing Python

Please refer to the [Python official website](https://www.python.org/) to install Python by yourself. The version must be 3.10-3.12.

After the installation is complete, you can check the Python version with the following command.

```bash
python --version
```

### Installing GCC

The following takes GCC 9 as example for how to install GCC on common operating systems:

- On Ubuntu, run the following commands to install.

    ```bash
    sudo apt-get install gcc-9 g++-9 -y
    ```

- On CentOS, run the following commands to install.

    ```bash
    sudo yum install centos-release-scl
    sudo yum install -y devtoolset-9-gcc devtoolset-9-gcc-c++ devtoolset-9-binutils
    ```

    After installation, run the following commands to switch to GCC 9.

    ```bash
    scl enable devtoolset-9 bash
    ```

- On EulerOS and openEuler, run the following commands to install.

    ```bash
    sudo yum install gcc g++ -y
    ```

### Installing MindSpore

First, refer to [Version List](https://www.mindspore.cn/versions) to select the version of MindSpore you want to install, and perform SHA-256 integrity check. Taking version 2.10.0 as an example, execute the following commands.

```bash
export MS_VERSION=2.10.0
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
