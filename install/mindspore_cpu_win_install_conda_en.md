# Installing MindSpore CPU by Conda-Windows

<!-- TOC -->

- [Installing MindSpore CPU by Conda-Windows](#installing-mindspore-cpu-by-conda-windows)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Creating and Accessing the Conda Virtual Environment](#creating-and-accessing-the-conda-virtual-environment)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.8/install/mindspore_cpu_win_install_conda_en.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source_en.png"></a>

[Conda](https://docs.conda.io/en/latest/) is an open-source, cross-platform, language-agnostic package manager and environment management system. It allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.

The following describes how to quickly install MindSpore by Conda on Windows in the CPU environment.

For details about how to install third-party dependency software when confirming the system environment information, see the third-party dependency software installation section in the [Installing MindSpore Using Source Code Build on Windows (CPU)](https://www.mindspore.cn/news/newschildren?id=364) provided by the community. Thanks to the community member [lvmingfu](https://gitee.com/lvmingfu) for sharing.

## System Environment Information Confirmation

- Ensure that Windows 10 is installed with the x86 architecture 64-bit operating system.
- Ensure that the Conda version that is compatible with the current system is installed.

    - If you prefer the complete capabilities provided by Conda, you can choose to download [Anaconda3](https://repo.anaconda.com/archive/).
    - If you want to save disk space or prefer customizing Conda installation package, you can choose to download [Miniconda3](https://repo.anaconda.com/miniconda/).

## Creating and Accessing the Conda Virtual Environment

Create a Conda virtual environment based on the Python version you want to use and go to the virtual environment.
If you want to use Python 3.7.5:

```bash
conda create -c conda-forge -n mindspore_py37 -c conda-forge python=3.7.5
activate mindspore_py37
```

If you want to use Python 3.8.0:

```bash
conda create -c conda-forge -n mindspore_py37 -c conda-forge python=3.8.0
activate mindspore_py38
```

If you want to use Python 3.9.0:

```bash
conda create -c conda-forge -n mindspore_py39 -c conda-forge python=3.9.0
activate mindspore_py39
```

## Installing MindSpore

Ensure that you are in the Conda virtual environment and run the following command to install the latest MindSpore. To install other versions, please refer to the specified version number of [Version List](https://www.mindspore.cn/versions) after `mindspore-cpu=`.

```bash
conda install mindspore-cpu -c mindspore -c conda-forge
```

When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/r1.8/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/r1.8/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.8/requirements.txt).

## Installation Verification

```bash
python -c "import mindspore;mindspore.run_check()"
```

The outputs should be the same as:

```text
MindSpore version:  __version__
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

It means MindSpore has been installed successfully.

## Version Update

Use the following command if you need to update the MindSpore version:

```bash
conda update mindspore-cpu -c mindspore -c conda-forge
```
