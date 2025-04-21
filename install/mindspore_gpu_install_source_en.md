# Installing MindSpore in GPU by Source Code

<!-- TOC -->

- [Installing MindSpore in GPU by Source Code](#installing-mindspore-in-gpu-by-source-code)
    - [Installing dependencies](#installing-dependencies)
        - [Installing CUDA](#installing-cuda)
        - [Installing cuDNN](#installing-cudnn)
        - [Installing Python](#installing-python)
        - [Installing wheel setuptools PyYAML and Numpy](#installing-wheel-setuptools-pyyaml-and-numpy)
        - [Installing GCC git and other dependencies](#installing-gcc-git-and-other-dependencies)
        - [Installing CMake](#installing-cmake)
        - [Installing LLVM-optional](#installing-llvm-optional)
        - [Installing TensorRT-optional](#installing-tensorrt-optional)
    - [Downloading the Source Code from the Code Repository](#downloading-the-source-code-from-the-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/install/mindspore_gpu_install_source_en.md)

This document describes how to install MindSpore by compiling source code on Linux in a GPU environment. The following takes Ubuntu 18.04 as an example to describe how to install MindSpore.

## Installing dependencies

The following table lists the system environment and third-party dependencies required to compile and install MindSpore GPU.

|Software|Version|Description|
|-|-|-|
|Ubuntu|18.04|OS for compiling and running MindSpore|
|[CUDA](#installing-cuda)|10.1 or 11.1 or 11.6|parallel computing architecture for MindSpore GPU|
|[cuDNN](#installing-cudnn)|7.6.x or 8.0.x or 8.5.x|deep neural network acceleration library used by MindSpore GPU|
|[Python](#installing-python)|3.9-3.11|Python environment that MindSpore depends on|
|[wheel](#installing-wheel-setuptools-pyyaml-and-numpy)|0.32.0 or later|Python packaging tool used by MindSpore|
|[setuptools](#installing-wheel-setuptools-pyyaml-and-numpy)|44.0 or later|Python package management tool used by MindSpore|
|[PyYAML](#installing-wheel-setuptools-pyyaml-and-numpy)|6.0-6.0.2|PyYAML module that operator compliation in MindSpore depends on|
|[Numpy](#installing-wheel-setuptools-pyyaml-and-numpy)|1.19.3-1.26.4|Numpy module that Numpy-related functions in MindSpore depends on|
|[GCC](#installing-gcc-git-and-other-dependencies)|7.3.0-9.4.0|C++ compiler for compiling MindSpore|
|[git](#installing-gcc-git-and-other-dependencies)|-|source code management tools used by MindSpore|
|[CMake](#installing-cmake)|3.22.2 or later|Compilation tool that builds MindSpore|
|[Autoconf](#installing-gcc-git-and-other-dependencies)|2.69 or later|Compilation tool that builds MindSpore|
|[Libtool](#installing-gcc-git-and-other-dependencies)|2.4.6-29.fc30 or later|Compilation tool that builds MindSpore|
|[Automake](#installing-gcc-git-and-other-dependencies)|1.15.1 or later|Compilation tool that builds MindSpore|
|[Flex](#installing-gcc-git-and-other-dependencies)|2.5.35 or later|lexical analyzer used by MindSpore|
|[tclsh](#installing-gcc-git-and-other-dependencies)|-|sqlite compilation dependencies for MindSpore|
|[patch](#installing-gcc-git-and-other-dependencies)|2.5 or later|source code patching tool used by MindSpore|
|[NUMA](#installing-gcc-git-and-other-dependencies)|2.0.11 or later|non-uniform memory access library used by MindSpore|
|[LLVM](#installing-llvm-optional)|12.0.1|compiler framework used by MindSpore (optional, required for graph kernel fusion and sparse computing)|
|[TensorRT](#installing-tensorrt-optional)|7.2.2 or 8.4|high performance deep learning inference SDK used by MindSpore(optional, required for serving inference)|

The following describes how to install the third-party dependencies.

### Installing CUDA

MindSpore GPU supports CUDA 10.1, CUDA 11.1 and CUDA 11.6. NVIDIA officially shows a variety of installation methods. For details, please refer to [CUDA download page](https://developer.nvidia.com/cuda-toolkit-archive) and [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
The following only shows instructions for installing by runfile on Linux systems.

Before installing CUDA, you need to run the following commands to install related dependencies.

```bash
sudo apt-get install linux-headers-$(uname -r) gcc-7
```

The minimum required GPU driver version of CUDA 10.1 is 418.39. The minimum required GPU driver version of CUDA 11.1 is 450.80.02. The minimum required GPU driver version of CUDA 11.6 is 510.39.01. You may run `nvidia-smi` command to confirm the GPU driver version. If the driver version does not meet the requirements, you should choose to install the driver during the CUDA installation. After installing the driver, you need to reboot your system.

Run the following command to install CUDA 11.6 (recommended).

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
sudo sh cuda_11.6.0_510.39.01_linux.run
echo -e "export PATH=/usr/local/cuda-11.6/bin:\$PATH" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

Or install CUDA 11.1 with the following command.

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
sudo sh cuda_11.1.1_455.32.00_linux.run
echo -e "export PATH=/usr/local/cuda-11.1/bin:\$PATH" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

Or install CUDA 10.1 with the following command.

```bash
wget https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
sudo sh cuda_10.1.243_418.87.00_linux.run
echo -e "export PATH=/usr/local/cuda-10.1/bin:\$PATH" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

When the default path `/usr/local/cuda` has an installation package, the LD_LIBRARY_PATH environment variable does not work. The reason is that MindSpore uses DT_RPATH to support startup without environment variables, reducing user settings. DT_RPATH has a higher priority than the LD_LIBRARY_PATH environment variable.

### Installing cuDNN

After completing the installation of CUDA, Log in and download the corresponding cuDNN installation package from [cuDNN page](https://developer.nvidia.com/cudnn). If CUDA 10.1 was previously installed, download cuDNN v7.6.x for CUDA 10.1. If CUDA 11.1 was previously installed, download cuDNN v8.0.x for CUDA 11.1. If CUDA 11.6 was previously installed, download cuDNN v8.5.x for CUDA 11.6. Note that download the tgz compressed file. Assuming that the downloaded cuDNN package file is named `cudnn.tgz` and the installed CUDA version is 11.6, execute the following command to install cuDNN.

```bash
tar -zxvf cudnn.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda-11.6/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.6/lib64
sudo chmod a+r /usr/local/cuda-11.6/include/cudnn*.h /usr/local/cuda-11.6/lib64/libcudnn*
```

If a different version of CUDA have been installed or the CUDA installation path is different, just replace `/usr/local/cuda-11.6` in the above command with the currently installed CUDA path.

### Installing Python

[Python](https://www.python.org/) can be installed in several ways.

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

### Installing wheel setuptools PyYAML and Numpy

After installing Python, run the following command to install them.

```bash
pip install wheel
pip install -U setuptools
pip install pyyaml
pip install "numpy>=1.19.3,<=1.26.4"
```

The Numpy version used in the runtime environment must be no less than the Numpy version in the compilation environment to ensure the normal use of Numpy related capabilities in the framework.

### Installing GCC git and other dependencies

Run the following commands to install GCC, git, Autoconf, Libtool, Automake, Flex, tclsh, patch and NUMA.

```bash
sudo apt-get install gcc-7 git automake autoconf libtool tcl patch libnuma-dev flex -y
```

To install a later version of GCC, run the following command to install GCC 8.

```bash
sudo apt-get install gcc-8 -y
```

Or install GCC 9 (Note that GCC 9 is not compatible with CUDA 10.1).

```bash
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-9 -y
```

### Installing CMake

Run the following commands to install [CMake](https://cmake.org/).

```bash
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt-get install cmake -y
```

### Installing LLVM-optional

Run the following commands to install [LLVM](https://llvm.org/).

```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-12 main"
sudo apt-get update
sudo apt-get install llvm-12-dev -y
```

### Installing TensorRT-optional

After completing the installation of CUDA and cuDNN, download TensorRT 8.4 for CUDA 11.6 from [TensorRT download page](https://developer.nvidia.com/nvidia-tensorrt-8x-download), and note to download installation package in TAR format. Suppose the downloaded file is named `TensorRT-8.4.1.5.Ubuntu-18.04.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz`. Install TensorRT with the following command.

```bash
tar xzf TensorRT-8.4.1.5.Ubuntu-18.04.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz
cd TensorRT-8.4.1.5
echo -e "export TENSORRT_HOME=$PWD" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=\$TENSORRT_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
cd -
```

## Downloading the Source Code from the Code Repository

```bash
git clone -b v2.6.0rc1 https://gitee.com/mindspore/mindspore.git
```

## Compiling MindSpore

Before compiling, please make sure that installation path of nvcc has been added to `PATH` and `LD_LIBRARY_PATH` environment variabels. If you have not done so, please follow the example below, based on CUDA11 installed in default location:

```bash
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.6
```

If a different version of CUDA have been installed or the CUDA installation path is different, replace `/usr/local/cuda-11.6` in the above command with the currently installed CUDA path.

Go to the root directory of MindSpore, then run the build script.

```bash
cd mindspore
bash build.sh -e gpu -S on
```

Where:

- In the `build.sh` script, the default number of compilation threads is 8. If the compiler performance is poor, compilation errors may occur. You can add `-j{Number of threads}` in to script to reduce the number of threads. For example, `bash build.sh -e ascend -j4`.
- By default, the dependent source code is downloaded from github. You may set -S option to `on` to download from the corresponding Gitee image.
- For more usage of `build.sh`, please refer to the description at the head of the script.

## Installing MindSpore

```bash
pip install output/mindspore-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

When the network is connected, dependencies of MindSpore are automatically downloaded during the .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/setup.py)). In other cases, you need to install dependencies by yourself.

## Installation Verification

**Method 1:**

Execute the following command:

```bash
python -c "import mindspore;mindspore.set_device(device_target='GPU');mindspore.run_check()"
```

The output should be like:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed on platform [GPU] successfully!
```

It means MindSpore has been installed successfully.

**Method 2:**

Execute the following command:

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_device(device_target="GPU")
x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
```

The outputs should be the same as:

```text
[[[[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]]]
```

It means MindSpore has been installed successfully.

## Version Update

After successfully executing the compile script `build.sh` in the root path of the source code, find the whl package in path `output`, and use the following command to update your version.

When upgrading from MindSpore 1.x to MindSpore 2.x, you need to manually uninstall the old version first:

```bash
pip uninstall mindspore-gpu
```

Then install MindSpore 2.x:

```bash
pip install mindspore*.whl
```

When upgrading from MindSpore 2.x:

```bash
pip install --upgrade mindspore*.whl
```
