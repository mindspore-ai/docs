# Installing MindSpore in CPU by Source Code-macOS

<!-- TOC -->

- [Installing MindSpore in CPU by Source Code-macOS](#installing-mindspore-in-cpu-by-source-code-macos)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Creating and Accessing the Conda Virtual Environment](#creating-and-accessing-the-conda-virtual-environment)
    - [Downloading Source Code from Code Repository](#downloading-source-code-from-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/install/mindspore_cpu_mac_install_source_en.md)

[Conda](https://docs.conda.io/en/latest/) is an open-source, cross-platform, language-agnostic package manager and environment management system. It allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.

This document describes how to install MindSpore by compiling source code on macOS with Conda installed.

## System Environment Information Confirmation

- According to the system and chip situation in the table below, determine the appropriate Python and Conda versions, and for the macOS version and chip information, click on the Apple logo in the upper left corner of the desktop - > `About this mac`:

    |Chip|Architecture|macOS Version|Supported Python Version|Supported Conda Version|
    |-|-|-|-|-|
    |M1|ARM|11.3|Python 3.9-3.11|Mambaforge or Miniforge|
    |Intel|x86_64|10.15/11.3|Python 3.9-3.11|Anaconda or Miniconda|

- Ensure that the Conda version is compatible with the current system and chip.

    - If you prefer the complete capabilities provided by Conda, you may download [Anaconda3](https://repo.anaconda.com/archive/) or [Mambaforge](https://github.com/conda-forge/miniforge).
    - If you want to save disk space or prefer customizing Conda installation package, you may download [Miniconda3](https://repo.anaconda.com/miniconda/) or [Miniforge](https://github.com/conda-forge/miniforge).

- Ensure that [Xcode](https://xcodereleases.com/) (>=12.4 and <= 13.0) is installed, where 12.4(X86) and 13.0(m1) are verified.

- Ensure that `Command Line Tools for Xcode` is installed. If not, use `sudo xcode-select --install` command to install Command Line Tools.

- Ensure that [CMake 3.22.2 and the later version](https://cmake.org/download/) is installed. Use `brew install cmake` if it's not installed.

- Ensure that [patch 2.5](https://ftp.gnu.org/gnu/patch/) is installed. Use `brew install patch` if it's not installed.

- Ensure that [wheel 0.32.0 or later](https://pypi.org/project/wheel/) is installed. Use `conda install wheel` if it's not installed.

- Ensure that [PyYAML](https://pypi.org/project/pyyaml/) (>=6.0 and <= 6.0.2) is installed. Use `conda install pyyaml` if it's not installed.

- Ensure that [autoconf](https://ftp.gnu.org/gnu/autoconf/) is installed. Use `brew install autoconf` if it's not installed.

- Ensure that [Numpy](https://pypi.org/project/numpy/) (>=1.19.3 and <= 1.26.4) is installed. Use `conda install numpy` if it's not installed.

## Creating and Accessing the Conda Virtual Environment

Create a Conda virtual environment based on the Python version you want to use and go to the virtual environment.

- If you want to use Python 3.9.11 (for 64-bit macOS 10.15 and 11.3):

  ```bash
  conda create -c conda-forge -n mindspore_py39 -c conda-forge python=3.9.11
  conda activate mindspore_py39
  ```

## Downloading Source Code from Code Repository

```bash
git clone -b v2.6.0 https://gitee.com/mindspore/mindspore.git
```

## Compiling MindSpore

Run the following command in the root directory of the source code to compile MindSpore:

```bash
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
bash build.sh -e cpu -S on -j4   # -j is a thread configuration when compiled, and if cpu performance is better, compile using multithreading, with parameters usually twice the number of CPU cores
```

## Installing MindSpore

```bash
# install prerequisites
conda install scipy -c conda-forge

pip install output/mindspore-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

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

After successfully executing the compile script `build.sh` in the root path of the source code, find the whl package in path `output`, and use the following command to update your version.

```bash
pip install --upgrade mindspore*.whl
```
