# Installing MindSpore GPU by Conda

<!-- TOC -->

- [Installing MindSpore GPU by Conda](#installing-mindspore-gpu-by-conda)
    - [Installing MindSpore and dependencies](#installing-mindspore-and-dependencies)
        - [Installing CUDA](#installing-cuda)
        - [Installing cuDNN](#installing-cudnn)
        - [Installing Conda](#installing-conda)
        - [Installing GCC](#installing-gcc)
        - [Installing TensorRT-optional](#installing-tensorrt-optional)
        - [Creating and Accessing the Conda Virtual Environment](#creating-and-accessing-the-conda-virtual-environment)
        - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/install/mindspore_gpu_install_conda_en.md)

[Conda](https://docs.conda.io/en/latest/) is an open-source, cross-platform, language-agnostic package manager and environment management system. It allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.

This document describes how to install MindSpore by Conda on Linux in a GPU environment. The following takes Ubuntu 18.04 as an example to describe how to install MindSpore.

## Installing MindSpore and dependencies

The following table lists the system environment and third-party dependencies required to install MindSpore.

|Software|Version|Description|
|-|-|-|
|Ubuntu|18.04|OS for running MindSpore|
|[CUDA](#installing-cuda)|10.1 or 11.1 or 11.6|parallel computing architecture for MindSpore GPU|
|[cuDNN](#installing-cudnn)|7.6.x or 8.0.x or 8.5.x|deep neural network acceleration library used by MindSpore GPU|
|[Conda](#installing-conda)|Anaconda3 or Miniconda3|Python environment management tool|
|[GCC](#installing-gcc)|7.3.0-9.4.0|C++ compiler for compiling MindSpore|
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

If a different version of CUDA have been installed or the CUDA installation path is different, just replace `/usr/local/cuda-11.6` in the above command with the CUDA path currently installed.

### Installing Conda

Run the following command to install Miniconda.

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-$(arch).sh
bash Miniconda3-py37_4.10.3-Linux-$(arch).sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

After the installation is complete, you can set up Tsinghua source acceleration download for Conda, and see [here](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/).

### Installing GCC

Run the following commands to install GCC.

```bash
sudo apt-get install gcc-7 -y
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

### Installing TensorRT-optional

After completing the installation of CUDA and cuDNN, download TensorRT 8.4 for CUDA 11.6 from [TensorRT download page](https://developer.nvidia.com/nvidia-tensorrt-8x-download), and note to download installation package in TAR format. Suppose the downloaded file is named `TensorRT-8.4.1.5.Ubuntu-18.04.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz`, install TensorRT with the following command.

```bash
tar xzf TensorRT-8.4.1.5.Ubuntu-18.04.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz
cd TensorRT-8.4.1.5
echo -e "export TENSORRT_HOME=$PWD" >> ~/.bashrc
echo -e "export LD_LIBRARY_PATH=\$TENSORRT_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
cd -
```

### Creating and Accessing the Conda Virtual Environment

Create a Conda virtual environment based on the Python version you want to use and go to the virtual environment.

If you want to use Python 3.9.11, execute the following command:

```bash
conda create -c conda-forge -n mindspore_py39 python=3.9.11 -y
conda activate mindspore_py39
```

If you wish to use another version of Python, just change the Python version in the above command. Python 3.9, Python 3.10 and Python 3.11 are currently supported.

### Installing MindSpore

Ensure that you are in the Conda virtual environment and run the following command to install the latest MindSpore. To install other versions, please refer to the specified version number of [Version List](https://www.mindspore.cn/versions) after `conda install mindspore=`.

For CUDA 10.1:

```bash
conda install mindspore -c mindspore -c conda-forge
```

For CUDA 11.1:

```bash
conda install mindspore -c mindspore -c conda-forge
```

For CUDA 11.6:

```bash
conda install mindspore -c mindspore -c conda-forge
```

When the network is connected, dependency items are automatically downloaded during MindSpore installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/br_base/setup.py)). In other cases, you need to install dependency by yourself.

## Installation Verification

Before running MindSpore GPU version, please make sure that installation path of nvcc has been added to `PATH` and `LD_LIBRARY_PATH` environment variabels. If you have not done so, please follow the example below, based on CUDA11 installed in default location:

```bash
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
```

If a different version of CUDA have been installed or the CUDA installation path is different, replace `/usr/local/cuda-11.6` in the above command with the currently installed CUDA path.

**Method 1:**

Execute the following command:

```bash
python -c "import mindspore;mindspore.set_device(device_target='GPU');mindspore.run_check()"
```

The outputs should be the same as:

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

When upgrading from MindSpore 1.x to MindSpore 2.x, you need to manually uninstall the old version first:

```bash
conda remove mindspore-gpu
```

Then install MindSpore 2.x:

```bash
conda install mindspore -c mindspore -c conda-forge
```

When upgrading from MindSpore 2.x:

```bash
conda update mindspore -c mindspore -c conda-forge
```
