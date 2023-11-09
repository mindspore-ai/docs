# Installing MindSpore CPU by Conda-macOS

<!-- TOC -->

- [Installing MindSpore CPU by Conda-macOS](#installing-mindspore-cpu-by-conda-macos)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Creating and Accessing the Conda Virtual Environment](#creating-and-accessing-the-conda-virtual-environment)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/install/mindspore_cpu_install_conda_en.md)

[Conda](https://docs.conda.io/en/latest/) is an open-source, cross-platform, language-agnostic package manager and environment management system. It allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.

This document describes how to install MindSpore by Conda in a macOS system.

## System Environment Information Confirmation

- According to the system and chip situation in the table below, determine the appropriate Python and Conda versions, and for the macOS version and chip information, click on the Apple logo in the upper left corner of the desktop - > `About this mac`:

    |Chip|Architecture|macOS Version|Supported Python Version|Supported Conda Version|
    |-|-|-|-|-|
    |M1|ARM|11.3|Python 3.8-3.9|Mambaforge or Miniforge|
    |Intel|x86_64|10.15/11.3|Python 3.7-3.9|Anaconda or Miniconda|

> Note: [Python 3.8.10](https://www.python.org/downloads/release/python-3810/) or Python 3.8.5 via Conda are the oldest Python releases that supports macOS for ARM

- Ensure that the Conda version is compatible with the current system and chip.

    - If you prefer the complete capabilities provided by Conda, you may download [Anaconda3](https://repo.anaconda.com/archive/) or [Mambaforge](https://github.com/conda-forge/miniforge).
    - If you want to save disk space or prefer customizing Conda installation package, you may download [Miniconda3](https://repo.anaconda.com/miniconda/) or [Miniforge](https://github.com/conda-forge/miniforge).

## Creating and Accessing the Conda Virtual Environment

Create a Conda virtual environment based on the Python version you want to use and go to the virtual environment.

- If you want to use Python 3.7.5 (for 64-bit macOS 10.15):

  ```bash
  conda create -c conda-forge -n mindspore_py37 -c conda-forge python=3.7.5
  conda activate mindspore_py37
  ```

- If you want to use Python 3.8.5 (for 64-bit macOS 10.15 and 11.3):

  ```bash
  conda create -c conda-forge -n mindspore_py38 -c conda-forge python=3.8.5
  conda activate mindspore_py38
  ```

- If you want to use Python 3.9.0 (for 64-bit macOS 10.15 and 11.3):

  ```bash
  conda create -c conda-forge -n mindspore_py39 -c conda-forge python=3.9.0
  conda activate mindspore_py39
  ```

## Installing MindSpore

Ensure that you are in the Conda virtual environment and run the following command to install the latest MindSpore. To install other versions, please refer to the specified version number of [Version List](https://www.mindspore.cn/versions) after `mindspore=`.

```bash
conda install mindspore -c mindspore -c conda-forge
```

When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/r2.3/setup.py).) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/master/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r2.3/requirements.txt).

## Installation Verification

```bash
python -c "import mindspore;mindspore.set_context(device_target='CPU');mindspore.run_check()"
```

The outputs should be the same as:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
```

It means MindSpore has been installed successfully.

## Version Update

When upgrading from MindSpore 1.x to MindSpore 2.x, you need to manually uninstall the old version first:

```bash
conda remove mindspore-cpu
```

Then install MindSpore 2.x:

```bash
conda install mindspore -c mindspore -c conda-forge
```

When upgrading from MindSpore 2.x:

```bash
conda update mindspore -c mindspore -c conda-forge
```
