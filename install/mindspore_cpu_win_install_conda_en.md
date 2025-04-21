# Installing MindSpore CPU by Conda-Windows

<!-- TOC -->

- [Installing MindSpore CPU by Conda-Windows](#installing-mindspore-cpu-by-conda-windows)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Creating and Accessing the Conda Virtual Environment](#creating-and-accessing-the-conda-virtual-environment)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/install/mindspore_cpu_win_install_conda_en.md)

[Conda](https://docs.conda.io/en/latest/) is an open-source, cross-platform, language-agnostic package manager and environment management system. It allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.

The following describes how to install MindSpore by Conda on Windows in the CPU environment.

## System Environment Information Confirmation

- Ensure that Windows 10 is installed with the x86 architecture 64-bit operating system.
- Ensure that the Conda version that is compatible with the current system is installed.

    - If you prefer the complete capabilities provided by Conda, you can choose to download [Anaconda3](https://repo.anaconda.com/archive/).
    - If you want to save disk space or prefer customizing Conda installation package, you can choose to download [Miniconda3](https://repo.anaconda.com/miniconda/).

## Creating and Accessing the Conda Virtual Environment

To run Anaconda on Windows, you should open an Anaconda prompt via `Start | Anaconda3 | Anaconda Promt`.

Create a Conda virtual environment based on the Python version you want to use and go to the virtual environment.
If you want to use Python 3.9.11:

```bash
conda create -c conda-forge -n mindspore_py39 -c conda-forge python=3.9.11
conda activate mindspore_py39
```

## Installing MindSpore

Ensure that you are in the Conda virtual environment and run the following command to install the latest MindSpore. To install other versions, please refer to the specified version number of [Version List](https://www.mindspore.cn/versions) after `conda install mindspore=`.

```bash
conda install mindspore -c mindspore -c conda-forge
```

When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/setup.py)). In other cases, you need to install dependency by yourself.

## Installation Verification

Execute the following command:

```bash
python -c "import mindspore;mindspore.set_device(device_target='CPU');mindspore.run_check()"
```

The outputs should be the same as:

```text
MindSpore version:  __version__
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
```

It means MindSpore has been installed successfully.

## Version Update

When upgrading from MindSpore 2.4.x and below to MindSpore 2.5.x and above, you need to manually uninstall the old version first:

```bash
conda remove mindspore-cpu
```

Then install new package:

```bash
conda install mindspore -c mindspore -c conda-forge
```

When upgrading from MindSpore 2.5.x:

```bash
conda update mindspore -c mindspore -c conda-forge
```
