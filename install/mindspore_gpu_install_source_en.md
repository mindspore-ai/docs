# Installing MindSpore in GPU by Source Code

<!-- TOC -->

- [Installing MindSpore in GPU by Source Code](#installing-mindspore-in-gpu-by-source-code)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Downloading Source Code from Code Repository](#downloading-source-code-from-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.6/install/mindspore_gpu_install_source_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

This document describes how to quickly install MindSpore by source code in a Linux system with a GPU environment.

## System Environment Information Confirmation

- Ensure that the 64-bit operating system is installed, where Ubuntu 18.04 is verified.

- Ensure that [GCC 7.3.0](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) is installed.

- Ensure that [gmp 6.1.2](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz) is installed.

- Ensure that [llvm 12.0.1](https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-12.0.1.tar.gz) is installed. (optional, required for graph kernel fusion for CPU backend)

- Ensure that Python 3.7.5 or 3.9.0 is installed. If not installed, download and install Python from:

    - Python 3.7.5 (64-bit): [Python official website](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) or [HUAWEI CLOUD](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz).
    - Python 3.9.0 (64-bit): [Python official website](https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz) or [HUAWEI CLOUD](https://mirrors.huaweicloud.com/python/3.9.0/Python-3.9.0.tgz).

- Ensure that [CMake 3.18.3 or later](https://cmake.org/download/) is installed.
    - After installing, add the path of `cmake` to the environment variable PATH.

- Ensure that [patch 2.5 or later](http://ftp.gnu.org/gnu/patch/) is installed.
    - After installing, add the path of `patch` to the environment variable PATH.

- Ensure that [Autoconf 2.69 or later](https://www.gnu.org/software/autoconf) is installed. (Default versions of these tools built in their systems are supported.)

- Ensure that [Libtool 2.4.6-29.fc30 or later](https://www.gnu.org/software/libtool) is installed. (Default versions of these tools built in their systems are supported.)

- Ensure that [Automake 1.15.1 or later](https://www.gnu.org/software/automake) is installed.(Default versions of these tools built in their systems are supported.)

- Ensure that [Flex 2.5.35 or later](https://github.com/westes/flex/) is installed.

- Ensure that [wheel 0.32.0 or later](https://pypi.org/project/wheel/) is installed.

- Ensure that [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) with [cuDNN 7.6.X](https://developer.nvidia.com/rdp/cudnn-archive) or [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.0-download-archive) with [cuDNN 8.0.X](https://developer.nvidia.com/rdp/cudnn-archive) is installed as default configuration.
    - If CUDA is installed in a non-default path, after installing CUDA, environment variable `PATH`(e.g. `export PATH=/usr/local/cuda-${version}/bin:$PATH`) and `LD_LIBRARY_PATH`(e.g. `export LD_LIBRARY_PATH=/usr/local/cuda-${version}/lib64:$LD_LIBRARY_PATH`) need to be set. Please refer to [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions) for detailed post installation actions.

- Ensure that [OpenMPI 4.0.3](https://www.open-mpi.org/faq/?category=building#easy-build) is installed. (optional, required for single-node/multi-GPU and multi-node/multi-GPU training)

- Ensure that [OpenSSL 1.1.1 or later](https://github.com/openssl/openssl.git) is installed.
    - Ensure that [OpenSSL](https://github.com/openssl/openssl) is installed and set system variable `export OPENSSL_ROOT_DIR="OpenSSL installation directory"`.

- Ensure that [NCCL 2.7.6](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#debian) for CUDA 10.1 or [NCCL 2.7.8](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#debian) for CUDA 11.1 is installed. (optional, required for single-node/multi-GPU and multi-node/multi-GPU training)

- Ensure that [TensorRT-7.2.2](https://developer.nvidia.com/nvidia-tensorrt-download) is installed. (optional，required for Serving inference)

- Ensure that [NUMA 2.0.11 or later](https://github.com/numactl/numactl) is installed.
     If not, use the following command to install it:

    ```bash
    apt-get install libnuma-dev
    ```

- Ensure that the git is installed, otherwise execute the following command to install git:

    ```bash
    apt-get install git # for linux distributions using apt, e.g. ubuntu
    yum install git     # for linux distributions using yum, e.g. centos
    ```

## Downloading Source Code from Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git
```

## Compiling MindSpore

Run the following command in the root directory of the source code to compile MindSpore:

```bash
bash build.sh -e gpu
```

Of which,

- In the `build.sh` script, the default number of compilation threads is 8. If the compiler performance is poor, compilation errors may occur. You can add -j{Number of threads} in to script to reduce the number of threads. For example, `bash build.sh -e ascend -j4`.

## Installing MindSpore

```bash
pip install output/mindspore_gpu-{version}-{python_version}-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/r1.6/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/r1.6/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.6/requirements.txt).
- `{version}` specifies the MindSpore version number. For example, when installing MindSpore 1.5.0-rc1, set `{version}` to 1.5.0rc1.
- `{python_version}` spcecifies the python version for which MindSpore is built. If you wish to use Python3.7.5,`{python_version}` should be `cp37-cp37m`. If Python3.9.0 is used, it should be `cp39-cp39`.

## Installation Verification

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

- The outputs should be the same as:

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

- Update online

    ```bash
    pip install --upgrade mindspore-gpu
    ```

     Note: MindSpore with CUDA11 is selected by default when upgrading version 1.3.0 and above. If you still want to use MindSpore with CUDA10, please select the corresponding wheel installation package.

- Update after source code compilation

     After successfully executing the compile script `build.sh` in the root path of the source code, find the whl package in path `output`, use the following command to update your version.

    ```bash
    pip install --upgrade mindspore_gpu-{version}-{python_version}-linux_{arch}.whl
    ```
