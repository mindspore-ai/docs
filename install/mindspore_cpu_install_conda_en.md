# Installing MindSpore CPU by Conda

<!-- TOC -->

- [Installing MindSpore CPU by Conda](#installing-mindspore-cpu-by-conda)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Creating and Accessing the Conda Virtual Environment](#creating-and-accessing-the-conda-virtual-environment)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.6/install/mindspore_cpu_install_conda_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

[Conda](https://docs.conda.io/en/latest/) is an open-source, cross-platform, language-agnostic package manager and environment management system. It allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.

This document describes how to quickly install MindSpore by Conda in a Linux system with a CPU environment.

For details about how to install third-party dependency software when confirming the system environment information, see the third-party dependency software installation section in the [Installing MindSpore Using Source Code Build on Ubuntu (CPU)](https://www.mindspore.cn/news/newschildren?id=365) provided by the community. Thank you to the community member [damon0626](https://gitee.com/damon0626) for sharing.

## System Environment Information Confirmation

- Ensure that a 64-bit operating system is installed and the [glibc](https://www.gnu.org/software/libc/) version is 2.17 or later, where Ubuntu 18.04 is verified.
- Ensure that [GCC 7.3.0](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) is installed.
- Ensure that [gmp 6.1.2](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz) is installed.
- Ensure that the Conda version is compatible with the current system.
    - If you prefer the complete capabilities provided by Conda, you can choose to download [Anaconda3](https://repo.anaconda.com/archive/).
    - If you want to save disk space or prefer custom Conda installation, you can choose to download [Miniconda3](https://repo.anaconda.com/miniconda/).

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

## Installing MindSpore

Ensure that you are in the Conda virtual environment and run the following command to install MindSpore:

```bash
conda install mindspore-cpu={version} -c mindspore -c conda-forge
```

In the preceding information:

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/master/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt).
- `{version}` denotes the version of MindSpore. For example, when you are installing MindSpore 1.5.0-rc1, `{version}` should be 1.5.0rc1.

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
conda update mindspore-cpu -c mindspore -c conda-forge
```
