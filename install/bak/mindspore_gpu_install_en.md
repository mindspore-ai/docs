# MindSpore Installation Guide

This document describes how to quickly install MindSpore in a NVIDIA GPU environment.

<!-- TOC -->

- [MindSpore Installation Guide](#mindspore-installation-guide)
    - [Environment Requirements](#environment-requirements)
        - [Hardware Requirements](#hardware-requirements)
        - [System Requirements and Software Dependencies](#system-requirements-and-software-dependencies)
        - [(Optional) Installing Conda](#optional-installing-conda)
    - [Installation Guide](#installation-guide)
        - [Installing Using Executable Files](#installing-using-executable-files)
        - [Installing Using the Source Code](#installing-using-the-source-code)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)
- [Installing MindInsight](#installing-mindinsight)
- [Installing MindArmour](#installing-mindarmour)
- [Installing MindSpore Hub](#installing-mindspore-hub)

<!-- /TOC -->

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/install/mindspore_gpu_install_en.md)

This document describes how to quickly install MindSpore in a NVIDIA GPU environment.

## Environment Requirements

### Hardware Requirements

- Nvidia GPU

### System Requirements and Software Dependencies

| Version | Operating System | Executable File Installation Dependencies | Source Code Compilation and Installation Dependencies |
| ---- | :--- | :--- | :--- |
| MindSpore 1.0.1 | Ubuntu 18.04 x86_64 | - [Python](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) 3.7.5 <br> - [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) <br> - [CuDNN](https://developer.nvidia.com/rdp/cudnn-archive) 7.6 <br> - [OpenMPI](https://www.open-mpi.org/faq/?category=building#easy-build) 3.1.5 (optional, required for single-node/multi-GPU and multi-node/multi-GPU training) <br> - [NCCL](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#debian) 2.7.6-1 (optional, required for single-node/multi-GPU and multi-node/multi-GPU training) <br> - [gmp](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz) 6.1.2 <br> - For details about other dependency items, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.0/requirements.txt). | **Compilation dependencies:**<br> - [Python](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) 3.7.5 <br> - [wheel](https://pypi.org/project/wheel/) >= 0.32.0 <br> - [CMake](https://cmake.org/download/) >= 3.14.1 <br> - [GCC](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) 7.3.0 <br> - [patch](http://ftp.gnu.org/gnu/patch/) >= 2.5 <br> - [Autoconf](https://www.gnu.org/software/autoconf) >= 2.69 <br> - [Libtool](https://www.gnu.org/software/libtool) >= 2.4.6-29.fc30 <br> - [Automake](https://www.gnu.org/software/automake) >= 1.15.1 <br> - [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) <br> - [CuDNN](https://developer.nvidia.com/rdp/cudnn-archive) 7.6  <br> - [gmp](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz) 6.1.2 <br> - [Flex](https://github.com/westes/flex/) >= 2.5.35 <br> **Installation dependencies:**<br> same as the executable file installation dependencies. |

- If Python has already installed, using `python --version` to check whether the version match, make sure your Python was added in environment variable `PATH`.
- Add pip to the environment variable to ensure that Python related toolkits can be installed directly through pip. You can get pip installer here `https://pypi.org/project/pip/` if pip is not install.
- When the network is connected, dependency items in the `requirements.txt` file are automatically downloaded during `.whl` package installation. In other cases, you need to manually install dependency items.
- MindSpore reduces dependency on Autoconf, Libtool, Automake versions for the convenience of users, default versions of these tools built in their systems are now supported.
- **If CUDA is installed in a non-default path**, after installing CUDA, environment variable `PATH`(e.g. `export PATH=/usr/local/cuda-${version}/bin:$PATH`) and `LD_LIBRARY_PATH`(e.g. `export LD_LIBRARY_PATH=/usr/local/cuda-${version}/lib64:$LD_LIBRARY_PATH`) need to be set. Please refer to [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions) for detailed post installation actions.

### (Optional) Installing Conda

1. Download the Conda installation package from the following path:

   - [X86 Anaconda](https://www.anaconda.com/distribution/) or [X86 Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Create and activate the Python environment.

    ```bash
    conda create -n {your_env_name} python=3.7.5
    conda activate {your_env_name}
    ```

> Conda is a powerful Python environment management tool. Beginners are adviced to check related information on the Internet.

## Installation Guide

### Installing Using Executable Files

- Download the .whl package from the [MindSpore website](https://www.mindspore.cn/versions/en). It is recommended to perform SHA-256 integrity verification first and run the following command to install MindSpore:

    ```bash
    pip install mindspore_gpu-{version}-cp37-cp37m-linux_{arch}.whl -i https://mirrors.huaweicloud.com/repository/pypi/simple
    ```

### Installing Using the Source Code

1. Download the source code from the code repository.

    ```bash
    git clone https://gitee.com/mindspore/mindspore.git -b r1.0
    ```

2. Run the following command in the root directory of the source code to compile MindSpore:

    ```bash
    bash build.sh -e gpu
    ```
    >
    > - Before running the preceding command, ensure that the paths where the executable files `cmake` and `patch` store have been added to the environment variable PATH.
    > - Before running the preceding command, ensure that [OpenSSL](https://github.com/openssl/openssl) is installed and set system variable `export OPENSSL_ROOT_DIR="path/to/openssl/install/directory"`.
    > - In the `build.sh` script, the `git clone` command will be executed to obtain the code in the third-party dependency database. Ensure that the network settings of Git are correct.
    > - In the `build.sh` script, the default number of compilation threads is 8. If the compiler performance is poor, compilation errors may occur. You can add -j{Number of threads} in to script to reduce the number of threads. For example, `bash build.sh -e gpu -j4`.

3. Run the following command to install MindSpore:

    ```bash
    chmod +x build/package/mindspore_gpu-{version}-cp37-cp37m-linux_{arch}.whl
    pip install build/package/mindspore_gpu-{version}-cp37-cp37m-linux_{arch}.whl -i https://mirrors.huaweicloud.com/repository/pypi/simple
    ```

## Installation Verification

- After Installation, execute the following Python script：

    ```bash
    import numpy as np
    from mindspore import Tensor
    import mindspore.ops as ops
    import mindspore.context as context

    context.set_context(device_target="GPU")
    x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
    y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
    print(ops.tensor_add(x, y))
    ```

- The outputs should be the same as:

    ```bash
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

## Version Update

Using the following command if you need update MindSpore version.

- Update Online

    ```bash
    pip install --upgrade mindspore-gpu
    ```

- Update after source code compilation

    After successfully executing the compile script `build.sh` in the root path of the source code, find the whl package in path `build/package`, use the following command to update your version.

    ```bash
    pip install --upgrade build/package/mindspore_gpu-{version}-cp37-cp37m-linux_{arch}.whl
    ```

# Installing MindInsight

If you need to analyze information such as model scalars, graphs, and model traceback, you can install MindInsight.

## Environment Requirements

### System Requirements and Software Dependencies

| Version | Operating System | Executable File Installation Dependencies | Source Code Compilation and Installation Dependencies |
| ---- | :--- | :--- | :--- |
| MindInsight 1.0.1 | - Ubuntu 18.04 x86_64 | - [Python](https://www.python.org/downloads/) 3.7.5 <br> - MindSpore 1.0.1 <br> - For details about other dependency items, see [requirements.txt](https://gitee.com/mindspore/mindinsight/blob/r1.0/requirements.txt). | **Compilation dependencies:**<br> - [Python](https://www.python.org/downloads/) 3.7.5 <br> - [CMake](https://cmake.org/download/) >= 3.14.1 <br> - [GCC](https://gcc.gnu.org/releases.html) 7.3.0 <br> - [node.js](https://nodejs.org/en/download/) >= 10.19.0 <br> - [wheel](https://pypi.org/project/wheel/) >= 0.32.0 <br> - [pybind11](https://pypi.org/project/pybind11/) >= 2.4.3 <br> **Installation dependencies:**<br> same as the executable file installation dependencies. |

- When the network is connected, dependency items in the `requirements.txt` file are automatically downloaded during .whl package installation. In other cases, you need to manually install dependency items.

## Installation Guide

### Installing Using Executable Files

1. Download the .whl package from the [MindSpore website](https://www.mindspore.cn/versions/en). It is recommended to perform SHA-256 integrity verification first  and run the following command to install MindInsight:

    ```bash
    pip install mindinsight-{version}-cp37-cp37m-linux_{arch}.whl -i https://mirrors.huaweicloud.com/repository/pypi/simple
    ```

2. Run the following command. If `web address: http://127.0.0.1:8080` is displayed, the installation is successful.

    ```bash
    mindinsight start
    ```

### Installing Using the Source Code

1. Download the source code from the code repository.

    ```bash
    git clone https://gitee.com/mindspore/mindinsight.git -b r1.0
    ```

    > You are **not** supposed to obtain the source code from the zip package downloaded from the repository homepage.

2. Install MindInsight by using either of the following installation methods:

   (1) Access the root directory of the source code and run the following installation command:

      ```bash
      cd mindinsight
      pip install -r requirements.txt -i https://mirrors.huaweicloud.com/repository/pypi/simple
      python setup.py install
      ```

   (2) Create a .whl package to install MindInsight.

      Access the root directory of the source code.
      First run the MindInsight compilation script under the `build` directory of the source code.
      Then run the command to install the .whl package generated in the `output` directory of the source code.

      ```bash
      cd mindinsight
      bash build/build.sh
      pip install output/mindinsight-{version}-cp37-cp37m-linux_{arch}.whl -i https://mirrors.huaweicloud.com/repository/pypi/simple
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
| MindArmour 1.0.1 | Ubuntu 18.04 x86_64 | - [Python](https://www.python.org/downloads/) 3.7.5 <br> - MindSpore 1.0.1 <br> - For details about other dependency items, see [setup.py](https://gitee.com/mindspore/mindarmour/blob/r1.0/setup.py). | Same as the executable file installation dependencies. |

- When the network is connected, dependency items in the `setup.py` file are automatically downloaded during .whl package installation. In other cases, you need to manually install dependency items.

## Installation Guide

### Installing Using Executable Files

1. Download the .whl package from the [MindSpore website](https://www.mindspore.cn/versions/en). It is recommended to perform SHA-256 integrity verification first and run the following command to install MindArmour:

   ```bash
   pip install mindarmour-{version}-cp37-cp37m-linux_{arch}.whl -i https://mirrors.huaweicloud.com/repository/pypi/simple
   ```

2. Run the following command. If no loading error message such as `No module named 'mindarmour'` is displayed, the installation is successful.

   ```bash
   python -c 'import mindarmour'
   ```

### Installing Using the Source Code

1. Download the source code from the code repository.

   ```bash
   git clone https://gitee.com/mindspore/mindarmour.git -b r1.0
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

# Installing MindSpore Hub

If you need to access and experience MindSpore pre-trained models quickly, you can install MindSpore Hub.

For details about install steps, see [MindSpore Hub](https://gitee.com/mindspore/hub/blob/master/README.md).
