# Installing MindSpore CPU Nightly by pip

<!-- TOC -->

- [Installing MindSpore CPU Nightly by pip](#installing-mindspore-cpu-nightly-by-pip)
    - [Installing MindSpore and dependencies](#installing-mindspore-and-dependencies)
        - [Installing Python](#installing-python)
        - [Installing GCC](#installing-gcc)
    - [Downloading and Installing MindSpore](#downloading-and-installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/install/mindspore_cpu_install_nightly_en.md)

MindSpore Nightly is a preview version which includes latest features and bugfixes, not fully supported and tested. Install MindSpore Nightly version if you wish to try out the latest features or bug fixes can use this version.

This document describes how to install MindSpore Nightly by pip on Linux in a CPU environment.

In the process of confirming the system environment information, if you need to know how to install third-party dependent software, you can refer to the practice provided by the community - [Source code compilation and installation on Ubuntu (CPU) MindSpore](https://www.mindspore.cn/news/newschildren?id=365) in the third-party dependent software installation related section, hereby thank the community members [damon0626]( https://gitee.com/damon0626) sharing.

## Installing MindSpore and dependencies

The following table lists the system environment and third-party dependencies required to install MindSpore.

| software                       | version     | description                                             |
| ------------------------------ | ----------- | ------------------------------------------------------- |
| Ubuntu                         | 18.04       | OS for running MindSpore                                |
| [Python](#installing-python)   | 3.9-3.11     | Python environment that MindSpore depends               |
| [GCC](#installing-gcc) | 7.3.0-9.4.0 | C++ compiler for compiling MindSpore                    |

The following describes how to install the third-party dependencies.

### Installing Python

[Python](https://www.python.org/) can be installed in multiple ways.

- Install Python with Conda.

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

- Or install Python via APT with the following command.

  ```bash
  sudo apt-get update
  sudo apt-get install software-properties-common -y
  sudo add-apt-repository ppa:deadsnakes/ppa -y
  sudo apt-get install python3.9 python3.9-dev python3.9-distutils python3-pip -y
  # set new installed Python as default
  sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.9 100
  # install pip
  python -m pip install pip -i https://repo.huaweicloud.com/repository/pypi/simple
  sudo update-alternatives --install /usr/bin/pip pip ~/.local/bin/pip3.9 100
  pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
  ```

  To install other Python versions, just change `3.9` in the command.

Run the following command to check the Python version.

```bash
python --version
```

### Installing GCC

Run the following commands to install GCC.

```bash
sudo apt-get install gcc-7 -y
```

To install a later version of GCC, run the following command to install GCC 8.

```bash
sudo apt-get install gcc-8 -y
```

Or install GCC 9.

```bash
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-9 -y
```

## Downloading and Installing MindSpore

Execute the following command to install MindSpore:

```bash
pip install mindspore-dev -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependencies are automatically downloaded during .whl package installation. (For details about the dependencies, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/setup.py)). In other cases, you need to install dependencies by yourself.
- pip will be installing the latest version of MindSpore Nightly automatically. If you wish to specify the version to be installed, please refer to the instruction below regarding to version update, and specify version manually.

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

Using the following command if you need to update the MindSpore version:

```bash
pip install --upgrade mindspore-dev=={version}
```

Of which,

- When updating to a release candidate (RC) version, set `{version}` to the RC version number, for example, 2.0.0.rc1. When updating to a stable release, you can remove `=={version}`.
