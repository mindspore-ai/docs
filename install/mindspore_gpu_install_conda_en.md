# Installing MindSpore GPU by Conda

<!-- TOC -->

- [Installing MindSpore GPU by Conda](#installing-mindspore-gpu-by-conda)
    - [Automatic Installation](#automatic-installation)
    - [Manual Installation](#manual-installation)
        - [Installing CUDA](#installing-cuda)
        - [Installing cuDNN](#installing-cudnn)
        - [Installing Conda](#installing-conda)
        - [Installing GCC and gmp](#installing-gcc-and-gmp)
        - [Installing Open MPI-optional](#installing-open-mpi-optional)
        - [Installing TensorRT-optional](#installing-tensorrt-optional)
        - [Creating and Accessing the Conda Virtual Environment](#creating-and-accessing-the-conda-virtual-environment)
        - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.7/install/mindspore_gpu_install_conda_en.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

[Conda](https://docs.conda.io/en/latest/) is an open-source, cross-platform, language-agnostic package manager and environment management system. It allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.

This document describes how to quickly install MindSpore by Conda in a Linux system with a GPU environment. The following takes Ubuntu 18.04 as an example to describe how to install MindSpore.

- If you want to install MindSpore by Conda on a fresh Ubuntu 18.04 with a GPU environment, you may use [automatic installation script](https://gitee.com/mindspore/mindspore/raw/r1.7/scripts/install/ubuntu-gpu-conda.sh) for one-click installation, see [Automatic Installation](#automatic-installation) section. The automatic installation script will install MindSpore and its dependencies.

- If some dependencies, such as CUDA, Conda and GCC, have been installed in your system, it is recommended to install manually by referring to the installation steps in the [Manual Installation](#manual-installation) section.

## Automatic Installation

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

The automatic installation script needs to replace the source list and install dependencies via APT, it will apply for root privileges during execution. Use the following commands to get the automatic installation script and execute. The automatic installation script only supports the installation of MindSpore>=1.6.0.

```bash
wget https://gitee.com/mindspore/mindspore/raw/r1.7/scripts/install/ubuntu-gpu-conda.sh
# install Python 3.7, CUDA 11.1 and the latest MindSpore by default
bash -i ./ubuntu-gpu-conda.sh
# to specify Python, CUDA and MindSpore version, taking Python 3.9, CUDA 10.1 and MindSpore 1.6.0 as examples, use the following manners
# PYTHON_VERSION=3.9 CUDA_VERSION=10.1 MINDSPORE_VERSION=1.6.0 bash -i ./ubuntu-gpu-conda.sh
```

This script performs the following operations:

- Change the software source configuration to a HUAWEI CLOUD source.
- Install the dependencies required by MindSpore, such as GCC, gmp.
- Download and install CUDA and cuDNN.
- Install Conda and create a virtual environment for MindSpore.
- Install MindSpore GPU by Conda.
- Install Open MPI if OPENMPI is set to `on`.

After the automatic installation script is executed, you need to reopen the terminal window to make the environment variables take effect. The automatic installation script creates a virtual environment named `mindspore_pyXX` for MindSpore. Where `XX` is the Python version, such as Python 3.7, the virtual environment name is `mindspore_py37`. Run the following command to show all virtual environments.

```bash
conda env list
```

To activate the virtual environment, take Python 3.7 as an example, execute the following command.

```bash
conda activate mindspore_py37
```

For more usage, see the script header description.

## Manual Installation

The following table lists the system environment and third-party dependencies required to install MindSpore.

|software|version|description|
|-|-|-|
|Ubuntu|18.04|OS for running MindSpore|
|[CUDA](#installing-cuda)|10.1 or 11.1|parallel computing architecture for MindSpore GPU|
|[cuDNN](#installing-cudnn)|7.6.x or 8.0.x|deep neural network acceleration library used by MindSpore GPU|
|[Conda](#installing-conda)|Anaconda3 or Miniconda3|Python environment management tool|
|[GCC](#installing-gcc-and-gmp)|7.3.0~9.4.0|C++ compiler for compiling MindSpore|
|[gmp](#installing-gcc-and-gmp)|6.1.2|multiple precision arithmetic library used by MindSpore|
|[Open MPI](#installing-open-mpi-optional)|4.0.3|high performance message passing library used by MindSpore (optional, required for single-node/multi-GPU and multi-node/multi-GPU training)|
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

If a different version of CUDA have been installed or the CUDA installation path is different, just replace `/usr/local/cuda-11.1` in the above command with the CUDA path currently installed.

### Installing Conda

Run the following command to install Miniconda.

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

After the installation is complete, you can set up Tsinghua source acceleration download for Conda, and see [here](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/).

### Installing GCC and gmp

Run the following commands to install GCC and gmp.

```bash
sudo apt-get install gcc-7 libgmp-dev -y
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

### Installing Open MPI-optional

You may compile and install [Open MPI](https://www.open-mpi.org/) by the following command.

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

### Installing TensorRT-optional

After completing the installation of CUDA and cuDNN, download TensorRT 7.2.2 for CUDA 11.1 from [TensorRT download page](https://developer.nvidia.com/nvidia-tensorrt-7x-download), and note to download installation package in TAR format. Suppose the downloaded file is named `TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz`, install TensorRT with the following command.

```bash
tar xzf TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz
cd TensorRT-7.2.2.3
echo -e "export TENSORRT_HOME=$PWD" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=\$TENSORRT_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
cd -
```

### Creating and Accessing the Conda Virtual Environment

Create a Conda virtual environment based on the Python version you want to use and go to the virtual environment.

If you want to use Python 3.7.5:

```bash
conda create -c conda-forge -n mindspore_py37 python=3.7.5 -y
conda activate mindspore_py37
```

If you wish to use another version of Python, just change the Python version in the above command. Python 3.7, Python 3.8 and Python 3.9 are currently supported.

### Installing MindSpore

Ensure that you are in the Conda virtual environment and run the following command to install the latest MindSpore. To install other versions, please refer to the specified version number of [Version List](https://www.mindspore.cn/versions) after `mindspore-ascend=`.

For CUDA 10.1:

```bash
conda install mindspore-gpu cudatoolkit=10.1 -c mindspore -c conda-forge
```

For CUDA 11.1:

```bash
conda install mindspore-gpu cudatoolkit=11.1 -c mindspore -c conda-forge
```

When the network is connected, dependency items are automatically downloaded during MindSpore installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/r1.7/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/r1.7/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.7/requirements.txt).

## Installation Verification

Before running MindSpore GPU version, please make sure that installation path of nvcc has been added to `PATH` and `LD_LIBRARY_PATH` environment variabels. If you have not done so, please follow the example below, based on CUDA11 installed in default location:

```bash
export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
```

If a different version of CUDA have been installed or the CUDA installation path is different, replace `/usr/local/cuda-11.1` in the above command with the currently installed CUDA path.

i:

```bash
python -c "import mindspore;mindspore.run_check()"
```

The outputs should be the same as:

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
