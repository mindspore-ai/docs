# Installing MindSpore CPU Nightly by pip-macOS

<!-- TOC -->

- [Installing MindSpore CPU Nightly by pip-macOS](#installing-mindspore-cpu-nightly-by-pip-macos)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Creating and Accessing the Conda Virtual Environment](#creating-and-accessing-the-conda-virtual-environment)
    - [Downloading and Installing MindSpore](#downloading-and-installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/install/mindspore_cpu_mac_install_pip_en.md)

MindSpore Nightly is a preview version which includes latest features and bugfixes, not fully supported and tested. Install MindSpore Nightly version if you wish to try out the latest features or bug fixes can use this version.

[Conda](https://docs.conda.io/en/latest/) is an open-source, cross-platform, language-agnostic package manager and environment management system. It allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.

This document describes how to install MindSpore Nightly by pip in a macOS system with Conda installed.

## System Environment Information Confirmation

- According to the system and chip situation in the table below, determine the appropriate Python and Conda versions, and for the macOS version and chip information, click on the Apple logo in the upper left corner of the desktop - > `About this mac`:

    |Chip|Architecture|macOS Version|Supported Python Version|Supported Conda Version|
    |-|-|-|-|-|
    |M1|ARM|11.3|Python 3.9-3.11|Mambaforge or Miniforge|
    |Intel|x86_64|10.15/11.3|Python 3.9-3.11|Anaconda or Miniconda|

- Ensure that the Conda version is compatible with the current system and chip.

    - If you prefer the complete capabilities provided by Conda, you may download [Anaconda3](https://repo.anaconda.com/archive/) or [Mambaforge](https://github.com/conda-forge/miniforge).
    - If you want to save disk space or prefer customizing Conda installation package, you may download [Miniconda3](https://repo.anaconda.com/miniconda/) or [Miniforge](https://github.com/conda-forge/miniforge).

## Creating and Accessing the Conda Virtual Environment

Create a Conda virtual environment based on the Python version you want to use and go to the virtual environment.

- If you want to use Python 3.9.11 (for 64-bit macOS 10.15 and 11.3):

  ```bash
  conda create -c conda-forge -n mindspore_py39 -c conda-forge python=3.9.11
  conda activate mindspore_py39
  ```

## Downloading and Installing MindSpore

Execute the following command to install MindSpore:

```bash
# install prerequisites
conda install scipy -c conda-forge

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

Use the following command if you need to update the MindSpore version:

```bash
pip install --upgrade mindspore-dev=={version}
```

Of which,

- When updating to a release candidate (RC) version, set `{version}` to the RC version number, for example, 2.0.0.rc1. When updating to a stable release, you can remove `=={version}`.
