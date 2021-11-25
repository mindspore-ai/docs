# Installing MindSpore GPU by Conda

<!-- TOC -->

- [Installing MindSpore GPU by Conda](#installing-mindspore-gpu-by-conda)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Creating and Accessing the Conda Virtual Environment](#creating-and-accessing-the-conda-virtual-environment)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/install/mindspore_gpu_install_conda_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

[Conda](https://docs.conda.io/en/latest/) is an open-source, cross-platform, language-agnostic package manager and environment management system. It allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.

This document describes how to quickly install MindSpore by Conda in a Linux system with a GPU environment.

For details about how to install third-party dependency software when confirming the system environment information, see the third-party dependency software installation section in the [Installing MindSpore GPU Using Source Code Build on Linux](https://www.mindspore.cn/news/newschildren?id=401) provided by the community. Thank you to the community member [zhangyi](https://gitee.com/zhang_yi2020) for sharing.

## System Environment Information Confirmation

- Ensure that the 64-bit operating system is installed and the [glibc](https://www.gnu.org/software/libc/)>=2.17, where Ubuntu 18.04 is verified.
- Ensure that [GCC 7.3.0](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) is installed.
- Ensure that [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) with [cuDNN 7.6.X](https://developer.nvidia.com/rdp/cudnn-archive) or [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.0-download-archive) with [cuDNN 8.0.X](https://developer.nvidia.com/rdp/cudnn-archive) is installed.
    - If CUDA is installed in a non-default path, after installing CUDA, environment variable `PATH`(e.g. `export PATH=/usr/local/cuda-${version}/bin:$PATH`) and `LD_LIBRARY_PATH`(e.g. `export LD_LIBRARY_PATH=/usr/local/cuda-${version}/lib64:$LD_LIBRARY_PATH`) need to be set. Please refer to [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions) for detailed post installation actions.
- Ensure that [OpenMPI 4.0.3](https://www.open-mpi.org/faq/?category=building#easy-build) is installed. (optional, required for single-node/multi-GPU and multi-node/multi-GPU training)
- Ensure that [OpenSSL 1.1.1 or later](https://github.com/openssl/openssl.git) is installed.
    - Ensure that [OpenSSL](https://github.com/openssl/openssl) is installed and set system variable `export OPENSSL_ROOT_DIR="OpenSSL installation directory"`.
- Ensure that [NCCL 2.7.6](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#debian) for CUDA 10.1 or [NCCL 2.7.8](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#debian) for CUDA 11.1 is installed. (optional, required for single-node/multi-GPU and multi-node/multi-GPU training)
- Ensure that [TensorRT-7.2.2](https://developer.nvidia.com/nvidia-tensorrt-download) is installed. (optional, required for Serving inference).
- Ensure that [gmp 6.1.2](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz) is installed.
- Ensure that the Conda version is compatible with the current system.
    - If you prefer the complete capabilities provided by Conda, you can choose to download [Anaconda3](https://repo.anaconda.com/archive/).
    - If you want to save disk space or prefer custom Conda installation, you can choose to download [Miniconda3](https://repo.anaconda.com/miniconda/).

## Creating and Accessing the Conda Virtual Environment

Create a Conda virtual environment based on the Python version you want to use and go to the virtual environment.
If you want to use Python 3.7.5:

```bash
conda create -n mindspore_py37 -c conda-forge python=3.7.5
conda activate mindspore_py37
```

If you want to use Python 3.9.0:

```bash
conda create -n mindspore_py39 -c conda-forge python=3.9.0
conda activate mindspore_py39
```

## Installing MindSpore

Execute the following command to install MindSpore.

For CUDA 10.1:

```bash
conda install mindspore-gpu={version} cudatoolkit=10.1 -c mindspore -c conda-forge
```

For CUDA 11.1:

```bash
conda install mindspore-gpu={version} cudatoolkit=11.1 -c mindspore -c conda-forge
```

In the preceding information:

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/r1.5/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/r1.5/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.5/requirements.txt).
- `{version}` denotes the version of MindSpore. For example, when you are installing MindSpore 1.5.0-rc1, `{version}` should be 1.5.0rc1.

## Installation Verification

i:

```bash
python -c "import mindspore;mindspore.run_check()"
```

The outputs should be the same as:

```text
mindspore version: __version__
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

- The outputs should be the same as:

```text
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

It means MindSpore has been installed successfully.
