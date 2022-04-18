# Installing MindSpore in GPU by Source Code

<!-- TOC -->

- [Installing MindSpore in GPU by Source Code](#installing-mindspore-in-gpu-by-source-code)
    - [Environment Preparation-automatic recommended](#environment-preparation-automatic-recommended)
    - [Environment Preparation-manual](#environment-preparation-manual)
        - [Installing CUDA](#installing-cuda)
        - [Installing cuDNN](#installing-cudnn)
        - [Installing Python](#installing-python)
        - [Installing wheel and setuptools](#installing-wheel-and-setuptools)
        - [Installing GCC git and other dependencies](#installing-gcc-git-and-other-dependencies)
        - [Installing CMake](#installing-cmake)
        - [Installing Open MPI-optional](#installing-open-mpi-optional)
        - [Installing LLVM-optional](#installing-llvm-optional)
        - [Installing TensorRT-optional](#installing-tensorrt-optional)
    - [Downloading the Source Code from the Code Repository](#downloading-the-source-code-from-the-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.7/install/mindspore_gpu_install_source_en.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

This document describes how to quickly install MindSpore by source code in a Linux system with a GPU environment. The following takes Ubuntu 18.04 as an example to describe how to install MindSpore.

- If you want to configure an environment that can compile MindSpore on a fresh Ubuntu 18.04 with a GPU environment, you may use [automatic installation script](https://gitee.com/mindspore/mindspore/raw/r1.7/scripts/install/ubuntu-gpu-source.sh) for one-click configuration, see [Environment Preparation-automatic, recommended](#environment-preparation-automatic-recommended) section. The automatic installation script will install the dependencies required to compile MindSpore.

- If some dependencies, such as CUDA, Python and GCC, have been installed in your system, it is recommended to install manually by referring to the installation steps in the [Environment Preparation-manual](#environment-preparation-manual) section.

## Environment Preparation-automatic recommended

Before using the automatic installation script, you need to make sure that the NVIDIA GPU driver is correctly installed on the system. The minimum required GPU driver version of CUDA 10.1 is 418.39. The minimum required GPU driver version of CUDA 11.1 is 450.80.02. Execute the following command to check the driver version.

```bash
nvidia-smi
```

If the GPU driver is not installed, run the following command to install it.

```bash
sudo apt-get update
sudo apt-get install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
```

After the installation is complete, please reboot your system.

The root permission is required because the automatic installation script needs to change the software source configuration and install dependencies via APT. Run the following command to obtain and run the automatic installation script. The environment configured by the automatic installation script only supports compiling MindSpore>=1.6.0.

```bash
wget https://gitee.com/mindspore/mindspore/raw/r1.7/scripts/install/ubuntu-gpu-source.sh
# install Python 3.7 and CUDA 11.1 by default
bash -i ./ubuntu-gpu-source.sh
# to specify Python 3.9 and CUDA 10.1, and the installation optionally relying on Open MPI, use the following manners
# PYTHON_VERSION=3.9 CUDA_VERSION=10.1 OPENMPI=on bash -i ./ubuntu-gpu-source.sh
```

This script performs the following operations:

- Change the software source configuration to a HUAWEI CLOUD source
- Install the compilation dependencies required by MindSpore, such as GCC, CMake, etc.
- Install Python3 and pip3 via APT and set them as default.
- Download and install CUDA and cuDNN.
- Install Open MPI if OPENMPI is set to `on`.
- Install LLVM if LLVM is set to `on`.

After the automatic installation script is executed, you need to reopen the terminal window to make the environment variables take effect, and then you can jump to the [Downloading the Source Code from the Code Repository](#downloading-the-source-code-from-the-code-repository) section to downloading and compiling MindSpore.

For more usage, see the script header description.

## Environment Preparation-manual

The following table lists the system environment and third-party dependencies required to compile and install MindSpore GPU.

|software|version|description|
|-|-|-|
|Ubuntu|18.04|OS for compiling and running MindSpore|
|[CUDA](#installing-cuda)|10.1 or 11.1|parallel computing architecture for MindSpore GPU|
|[cuDNN](#installing-cudnn)|7.6.x or 8.0.x|deep neural network acceleration library used by MindSpore GPU|
|[Python](#installing-python)|3.7-3.9|Python environment that MindSpore depends on|
|[wheel](#installing-wheel-and-setuptools)|0.32.0 or later|Python packaging tool used by MindSpore|
|[setuptools](#installing-wheel-and-setuptools)|44.0 or later|Python package management tool used by MindSpore|
|[GCC](#installing-gcc-git-and-other-dependencies)|7.3.0~9.4.0|C++ compiler for compiling MindSpore|
|[git](#installing-gcc-git-and-other-dependencies)|-|source code management tools used by MindSpore|
|[CMake](#installing-cmake)|3.18.3 or later|Compilation tool that builds MindSpore|
|[Autoconf](#installing-gcc-git-and-other-dependencies)|2.69 or later|Compilation tool that builds MindSpore|
|[Libtool](#installing-gcc-git-and-other-dependencies)|2.4.6-29.fc30 or later|Compilation tool that builds MindSpore|
|[Automake](#installing-gcc-git-and-other-dependencies)|1.15.1 or later|Compilation tool that builds MindSpore|
|[gmp](#installing-gcc-git-and-other-dependencies)|6.1.2|multiple precision arithmetic library used by MindSpore|
|[Flex](#installing-gcc-git-and-other-dependencies)|2.5.35 or later|lexical analyzer used by MindSpore|
|[tclsh](#installing-gcc-git-and-other-dependencies)|-|sqlite compilation dependencies for MindSpore|
|[patch](#installing-gcc-git-and-other-dependencies)|2.5 or later|source code patching tool used by MindSpore|
|[NUMA](#installing-gcc-git-and-other-dependencies)|2.0.11 or later|non-uniform memory access library used by MindSpore|
|[Open MPI](#installing-open-mpi-optional)|4.0.3|high performance message passing library used by MindSpore (optional, required for single-node/multi-GPU and multi-node/multi-GPU training)|
|[LLVM](#installing-llvm-optional)|12.0.1|compiler framework used by MindSpore (optional, required for graph kernel fusion and sparse computing)|
|[TensorRT](#installing-tensorrt-optional)|7.2.2|high performance deep learning inference SDK used by MindSpore(optional, required for serving inference)|

The following describes how to install the third-party dependencies.

### Installing CUDA

MindSpore GPU supports CUDA 10.1 and CUDA 11.1. NVIDIA officially shows a variety of installation methods. For details, please refer to [CUDA download page](https://developer.nvidia.com/cuda-toolkit-archive) and [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
The following only shows instructions for installing by runfile on Linux systems.

Before installing CUDA, you need to run the following commands to install related dependencies.

```bash
sudo apt-get install linux-headers-$(uname -r) gcc-7
```

The minimum required GPU driver version of CUDA 10.1 is 418.39. The minimum required GPU driver version of CUDA 11.1 is 450.80.02. You may run `nvidia-smi` command to confirm the GPU driver version. If the driver version does not meet the requirements, you should choose to install the driver during the CUDA installation. After installing the driver, you need to reboot your system.

Run the following command to install CUDA 11.1 (recommended).

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

### Installing cuDNN

After completing the installation of CUDA, Log in and download the corresponding cuDNN installation package from [cuDNN page](https://developer.nvidia.com/zh-cn/cudnn). If CUDA 10.1 was previously installed, download cuDNN v7.6.x for CUDA 10.1. If CUDA 11.1 was previously installed, download cuDNN v8.0.x for CUDA 11.1. Note that download the tgz compressed file. Assuming that the downloaded cuDNN package file is named `cudnn.tgz` and the installed CUDA version is 11.1, execute the following command to install cuDNN.

```bash
tar -zxvf cudnn.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda-11.1/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.1/lib64
sudo chmod a+r /usr/local/cuda-11.1/include/cudnn.h /usr/local/cuda-11.1/lib64/libcudnn*
```

If a different version of CUDA have been installed or the CUDA installation path is different, just replace `/usr/local/cuda-11.1` in the above command with the currently installed CUDA path.

### Installing Python

[Python](https://www.python.org/) can be installed in several ways.

- Install Python with Conda.

    Install Miniconda:

    ```bash
    cd /tmp
    curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
    bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b
    cd -
    . ~/miniconda3/etc/profile.d/conda.sh
    conda init bash
    ```

    After the installation is complete, you can set up Tsinghua source acceleration download for Conda, and see [here](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/).

    Create a virtual environment, taking Python 3.7.5 as an example:

    ```bash
    conda create -n mindspore_py37 python=3.7.5 -y
    conda activate mindspore_py37
    ```

- Or install Python via APT with the following command.

    ```bash
    sudo apt-get update
    sudo apt-get install software-properties-common -y
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get install python3.7 python3.7-dev python3.7-distutils python3-pip -y
    # set new installed Python as default
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 100
    # install pip
    python -m pip install pip -i https://repo.huaweicloud.com/repository/pypi/simple
    sudo update-alternatives --install /usr/bin/pip pip ~/.local/bin/pip3.7 100
    pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
    ```

    To install other Python versions, just change `3.7` in the command.

Run the following command to check the Python version.

```bash
python --version
```

### Installing wheel and setuptools

After installing Python, run the following command to install them.

```bash
pip install wheel
pip install -U setuptools
```

### Installing GCC git and other dependencies

Run the following commands to install GCC, git, Autoconf, Libtool, Automake, gmp, Flex, tclsh, patch and NUMA.

```bash
sudo apt-get install gcc-7 git automake autoconf libtool libgmp-dev tcl patch libnuma-dev flex -y
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

### Installing Open MPI-optional

Run the following commands to install [Open MPI](https://www.open-mpi.org/).

```bash
curl -O https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz
tar xzf openmpi-4.0.3.tar.gz
cd openmpi-4.0.3
./configure --prefix=/usr/local/openmpi-4.0.3
make
sudo make install
echo -e "export PATH=/usr/local/openmpi-4.0.3/bin:\$PATH" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=/usr/local/openmpi-4.0.3/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
cd -
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

After completing the installation of CUDA and cuDNN, download TensorRT 7.2.2 for CUDA 11.1 from [TensorRT download page](https://developer.nvidia.com/nvidia-tensorrt-7x-download), and note to download installation package in TAR format. Suppose the downloaded file is named `TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz`. Install TensorRT with the following command.

```bash
tar xzf TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz
cd TensorRT-7.2.2.3
echo -e "export TENSORRT_HOME=$PWD" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=\$TENSORRT_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
cd -
```

## Downloading the Source Code from the Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.7
```

## Compiling MindSpore

Before compiling, please make sure that installation path of nvcc has been added to `PATH` and `LD_LIBRARY_PATH` environment variabels. If you have not done so, please follow the example below, based on CUDA11 installed in default location:

```bash
export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
```

If a different version of CUDA have been installed or the CUDA installation path is different, replace `/usr/local/cuda-11.1` in the above command with the currently installed CUDA path.

Go to the root directory of MindSpore, then run the build script.

```bash
cd mindspore
bash build.sh -e gpu -S on
```

Where:

- In the `build.sh` script, the default number of compilation threads is 8. If the compiler performance is poor, compilation errors may occur. You can add -j{Number of threads} in to script to reduce the number of threads. For example, `bash build.sh -e ascend -j4`.
- By default, the dependent source code is downloaded from github. You may set -S option to `on` to download from the corresponding gitee mirror.
- For more usage of `build.sh`, please refer to the description at the head of the script.

## Installing MindSpore

```bash
pip install output/mindspore_gpu-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

When the network is connected, dependencies of MindSpore are automatically downloaded during the .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/r1.7/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/r1.7/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.7/requirements.txt).

## Installation Verification

i:

```bash
python -c "import mindspore;mindspore.run_check()"
```

The output should be like:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

It means MindSpore has been installed successfully.

ii:

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="GPU")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
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

Using the following command if you need to update the MindSpore version.

- Update online directly

    ```bash
    pip install --upgrade mindspore-gpu
    ```

     Note: MindSpore with CUDA11 is selected by default when upgrading version 1.3.0 and above. If you still want to use MindSpore with CUDA10, please select the corresponding wheel installation package.

- Update after source code compilation

     After successfully executing the compile script `build.sh` in the source code root directory, find the .whl package generated by compilation in directory `output`, and use the following command to update your version.

    ```bash
    pip install --upgrade mindspore_gpu-*.whl
    ```
