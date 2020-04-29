# MindSpore Installation Guide

This document describes how to quickly install MindSpore on an Ascend AI processor environment.

<!-- TOC -->

- [MindSpore Installation Guide](#mindspore-installation-guide)
    - [Environment Requirements](#environment-requirements)
        - [Hardware Requirements](#hardware-requirements)
        - [System Requirements and Software Dependencies](#system-requirements-and-software-dependencies)
        - [(Optional) Installing Conda](#optional-installing-conda)
        - [Configuring software package Dependencies](#configuring-software-package-dependencies)
    - [Installation Guide](#installation-guide)
        - [Installing Using Executable Files](#installing-using-executable-files)
        - [Installing Using the Source Code](#installing-using-the-source-code)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Installation Verification](#installation-verification)
- [Installing MindInsight](#installing-mindinsight)
- [Installing MindArmour](#installing-mindarmour)

<!-- /TOC -->

## Environment Requirements

### Hardware Requirements

- Ascend 910 AI processor

  > - Reserve at least 32 GB memory for each card.

### System Requirements and Software Dependencies

| Version | Operating System | Executable File Installation Dependencies | Source Code Compilation and Installation Dependencies |
| ---- | :--- | :--- | :--- |
| MindSpore master | - Ubuntu 16.04 or later x86_64 <br> - EulerOS 2.8 arrch64 <br> - EulerOS 2.5 x86_64 | - [Python](https://www.python.org/downloads/) 3.7.5 <br> - Ascend 910 AI processor software package(Version:Atlas T 1.1.T107) <br> - For details about other dependency items, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r0.2/requirements.txt). | **Compilation dependencies:**<br> - [Python](https://www.python.org/downloads/) 3.7.5 <br> - Ascend 910 AI processor software package(Version:Atlas T 1.1.T107) <br> - [wheel](https://pypi.org/project/wheel/) >= 0.32.0 <br> - [GCC](https://gcc.gnu.org/releases.html) 7.3.0 <br> - [CMake](https://cmake.org/download/) >= 3.14.1 <br> - [patch](http://ftp.gnu.org/gnu/patch/) >= 2.5 <br> **Installation dependencies:**<br> same as the executable file installation dependencies. |

- Confirm that the current user has the right to access the installation path `/usr/local/Ascend `of Ascend 910 AI processor software package(Version:Atlas T 1.1.T107). If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located. For the specific configuration, please refer to the software package instruction document.
- When Ubuntu version is 18.04, GCC 7.3.0 can be installed by using apt command.
- When the network is connected, dependency items in the requirements.txt file are automatically downloaded during .whl package installation. In other cases, you need to manually install dependency items.

### (Optional) Installing Conda

1. Download the Conda installation package from the following path:

   - [X86 Anaconda](https://www.anaconda.com/distribution/) or [X86 Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   - [ARM Anaconda](https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh)

2. Create and activate the Python environment.

    ```bash
    conda create -n {your_env_name} python=3.7.5
    conda activate {your_env_name}
    ```

> Conda is a powerful Python environment management tool. It is recommended that a beginner read related information on the Internet first.

### Configuring software package Dependencies

 - Install the .whl package provided in Ascend 910 AI processor software package(Version:Atlas T 1.1.T107). The .whl package is released with the software package. After software package is upgraded, reinstall the .whl package.

    ```bash
    pip install /usr/local/Ascend/fwkacllib/lib64/topi-{version}-py3-none-any.whl
    pip install /usr/local/Ascend/fwkacllib/lib64/te-{version}-py3-none-any.whl
    pip install /usr/local/Ascend/fwkacllib/lib64/hccl-{version}-py3-none-any.whl
    ```

## Installation Guide

### Installing Using Executable Files

- Download the .whl package from the [MindSpore website](https://www.mindspore.cn/versions/en). It is recommended to perform SHA-256 integrity verification first and run the following command to install MindSpore:

    ```bash
    pip install mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl
    ```

### Installing Using the Source Code

The compilation and installation must be performed on the Ascend 910 AI processor environment.

1. Download the source code from the code repository.

    ```bash
    git clone https://gitee.com/mindspore/mindspore.git -b r0.2
    ```

2. Run the following command in the root directory of the source code to compile MindSpore:

    ```bash
    bash build.sh -e d -z
    ```
    > - Before running the preceding command, ensure that the paths where the executable files cmake and patch store have been added to the environment variable PATH.
    > - In the build.sh script, the git clone command will be executed to obtain the code in the third-party dependency database. Ensure that the network settings of Git are correct.
    > - In the build.sh script, the default number of compilation threads is 8. If the compiler performance is poor, compilation errors may occur. You can add -j{Number of threads} in to script to reduce the number of threads. For example, `bash build.sh -e d -z -j4`.

3. Run the following command to install MindSpore:

    ```bash
    chmod +x build/package/mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl
    pip install build/package/mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl
    ```

## Configuring Environment Variables

- After MindSpore is installed, export runtime-related environment variables.

    ```bash
    # control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
    export GLOG_v=2
    # Conda environmental options
    LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package
    # lib libraries that the run package depends on
    export LD_LIBRARY_PATH=${LOCAL_ASCEND}/add-ons/:${LOCAL_ASCEND}/fwkacllib/lib64:${LD_LIBRARY_PATH}
    # Environment variables that must be configured
    export TBE_IMPL_PATH=${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe # TBE operator implementation tool path
    export PATH=${LOCAL_ASCEND}/fwkacllib/ccec_compiler/bin/:${PATH} # TBE operator compilation tool path
    export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH} # Python library that TBE implementation depends on
    ```

## Installation Verification

- After configuring the environment variables, execute the following Python script：

    ```bash
    import numpy as np
    from mindspore import Tensor
    from mindspore.ops import functional as F
    import mindspore.context as context

    context.set_context(device_target="Ascend")
    x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
    y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
    print(F.tensor_add(x, y))
    ```

- The outputs should be same as:

    ```
    [[[ 2.  2.  2.  2.],
      [ 2.  2.  2.  2.],
      [ 2.  2.  2.  2.]],

     [[ 2.  2.  2.  2.],
      [ 2.  2.  2.  2.],
      [ 2.  2.  2.  2.]],

     [[ 2.  2.  2.  2.],
      [ 2.  2.  2.  2.],
      [ 2.  2.  2.  2.]]]
    ```

# Installing MindInsight

If you need to analyze information such as model scalars, graphs, and model traceback, you can install MindInsight.

## Environment Requirements

### System Requirements and Software Dependencies

| Version | Operating System | Executable File Installation Dependencies | Source Code Compilation and Installation Dependencies |
| ---- | :--- | :--- | :--- |
| MindInsight master | - Ubuntu 16.04 or later x86_64 <br> - EulerOS 2.8 arrch64 <br> - EulerOS 2.5 x86_64 <br> | - [Python](https://www.python.org/downloads/) 3.7.5 <br> - MindSpore master <br> - For details about other dependency items, see [requirements.txt](https://gitee.com/mindspore/mindinsight/blob/master/requirements.txt). | **Compilation dependencies:**<br> - [Python](https://www.python.org/downloads/) 3.7.5 <br> - [CMake](https://cmake.org/download/) >= 3.14.1 <br> - [GCC](https://gcc.gnu.org/releases.html) 7.3.0 <br> - [node.js](https://nodejs.org/en/download/) >= 10.19.0 <br> - [wheel](https://pypi.org/project/wheel/) >= 0.32.0 <br> - [pybind11](https://pypi.org/project/pybind11/) >= 2.4.3 <br> **Installation dependencies:**<br> same as the executable file installation dependencies. |

- When the network is connected, dependency items in the requirements.txt file are automatically downloaded during .whl package installation. In other cases, you need to manually install dependency items.

## Installation Guide

### Installing Using Executable Files

1. Download the .whl package from the [MindSpore website](https://www.mindspore.cn/versions/en). It is recommended to perform SHA-256 integrity verification first  and run the following command to install MindInsight:

    ```bash
    pip install mindinsight-{version}-cp37-cp37m-linux_{arch}.whl
    ```

2. Run the following command. If `web address: http://127.0.0.1:8080` is displayed, the installation is successful.

    ```bash
    mindinsight start
    ```

### Installing Using the Source Code

1. Download the source code from the code repository.

    ```bash
    git clone https://gitee.com/mindspore/mindinsight.git -b r0.2
    ```
    > You are **not** supposed to obtain the source code from the zip package downloaded from the repository homepage.

2. Install MindInsight by using either of the following installation methods:

   (1) Access the root directory of the source code and run the following installation command:

      ```bash
      cd mindinsight
      pip install -r requirements.txt
      python setup.py install
      ```

   (2) Create a .whl package to install MindInsight.

      Access the root directory of the source code.
      First run the MindInsight compilation script under the build directory of the source code.
      Then run the command to install the .whl package generated into the output directory of the source code.

      ```bash
      cd mindinsight
      bash build/build.sh
      pip install output/mindinsight-{version}-cp37-cp37m-linux_{arch}.whl
      ```

3. Run the following command. If `web address: http://127.0.0.1:8080` is displayed, the installation is successful.

    ```bash
    mindinsight start
    ```

# Installing MindArmour

If you need to conduct AI model security research or enhance the security of the model in you applications, you can install MindArmour.

## Environment Requirements

### System Requirements and Software Dependencies

| Version | Operating System | Executable File Installation Dependencies | Source Code Compilation and Installation Dependencies |
| ---- | :--- | :--- | :--- |
| MindArmour master | - Ubuntu 16.04 or later x86_64 <br> - EulerOS 2.8 arrch64 <br> - EulerOS 2.5 x86_64 <br> | - [Python](https://www.python.org/downloads/) 3.7.5 <br> - MindSpore master <br> - For details about other dependency items, see [setup.py](https://gitee.com/mindspore/mindarmour/blob/master/setup.py). | Same as the executable file installation dependencies. |

- When the network is connected, dependency items in the setup.py file are automatically downloaded during .whl package installation. In other cases, you need to manually install dependency items.

## Installation Guide

### Installing Using Executable Files

1. Download the .whl package from the [MindSpore website](https://www.mindspore.cn/versions/en). It is recommended to perform SHA-256 integrity verification first  and run the following command to install MindArmour:

   ```bash
   pip install mindarmour-{version}-cp37-cp37m-linux_{arch}.whl
   ```

2. Run the following command. If no loading error message such as `No module named 'mindarmour'` is displayed, the installation is successful.

   ```bash
   python -c 'import mindarmour'
   ```

### Installing Using the Source Code

1. Download the source code from the code repository.

   ```bash
   git clone https://gitee.com/mindspore/mindarmour.git -b r0.2
   ```

2. Run the following command in the root directory of the source code to compile and install MindArmour:

   ```bash
   cd mindarmour
   python setup.py install
   ```


3. Run the following command. If no loading error message such as `No module named 'mindarmour'` is displayed, the installation is successful.

   ```bash
   python -c 'import mindarmour'
   ```
