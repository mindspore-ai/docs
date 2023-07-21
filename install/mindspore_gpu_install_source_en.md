# Installing MindSpore in GPU by Source Code

<!-- TOC -->

- [Installing MindSpore in GPU by Source Code](#installing-mindspore-in-gpu-by-source-code)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Downloading Source Code from Code Repository](#downloading-source-code-from-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Installing MindInsight](#installing-mindinsight)
    - [Installing MindArmour](#installing-mindarmour)
    - [Installing MindSpore Hub](#installing-mindspore-hub)

<!-- /TOC -->

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.1/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/install/mindspore_gpu_install_source_en.md)

This document describes how to quickly install MindSpore by source code in a Linux system with a GPU environment.

## System Environment Information Confirmation

- Confirm that Ubuntu 18.04 is installed with the 64-bit operating system.
- Confirm that [GCC 7.3.0](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) is installed.
- Confirm that [gmp 6.1.2](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz) is installed.
- Confirm that [Python 3.7.5](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) is installed.
- Confirm that [CMake 3.18.3 or later](https://cmake.org/download/) is installed.
    - After installing, add the path of `cmake` to the environment variable PATH.
- Confirm that [patch 2.5 or later](http://ftp.gnu.org/gnu/patch/) is installed.
    - After installing, add the path of `patch` to the environment variable PATH.
- Confirm that [Autoconf 2.69 or later](https://www.gnu.org/software/autoconf) is installed. (Default versions of these tools built in their systems are supported.)
- Confirm that [Libtool 2.4.6-29.fc30 or later](https://www.gnu.org/software/libtool) is installed. (Default versions of these tools built in their systems are supported.)
- Confirm that [Automake 1.15.1 or later](https://www.gnu.org/software/automake) is installed.(Default versions of these tools built in their systems are supported.)
- Confirm that [cuDNN 7.6 or later](https://developer.nvidia.com/rdp/cudnn-archive) is installed.
- Confirm that [Flex 2.5.35 or later](https://github.com/westes/flex/) is installed.
- Confirm that [wheel 0.32.0 or later](https://pypi.org/project/wheel/) is installed.
- Confirm that [OpenSSL 1.1.1 or later](https://github.com/openssl/openssl.git) is installed.
    - ensure that [OpenSSL](https://github.com/openssl/openssl) is installed and set system variable `export OPENSSL_ROOT_DIR="OpenSSL installation directory"`.
- Confirm that [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) is installed as default configuration.
    - If CUDA is installed in a non-default path, after installing CUDA, environment variable `PATH`(e.g. `export PATH=/usr/local/cuda-${version}/bin:$PATH`) and `LD_LIBRARY_PATH`(e.g. `export LD_LIBRARY_PATH=/usr/local/cuda-${version}/lib64:$LD_LIBRARY_PATH`) need to be set. Please refer to [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions) for detailed post installation actions.
- Confirm that [OpenMPI 4.0.3](https://www.open-mpi.org/faq/?category=building#easy-build) is installed. (optional, required for single-node/multi-GPU and multi-node/multi-GPU training)
- Confirm that [NCCL 2.7.6-1](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#debian) is installed. (optional, required for single-node/multi-GPU and multi-node/multi-GPU training)
- Confirm that [NUMA 2.0.11 or later](https://github.com/numactl/numactl) is installed.
     If not, use the following command to install it:

    ```bash
    apt-get install libnuma-dev
    ```

- Confirm that the git tool is installed.  
     If not, use the following command to install it:

    ```bash
    apt-get install git
    ```

## Downloading Source Code from Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.1
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
chmod +x build/package/mindspore_gpu-{version}-cp37-cp37m-linux_x86_64.whl
pip install build/package/mindspore_gpu-{version}-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.1/requirements.txt)). In other cases, you need to manually install dependency items.
- `{version}` denotes the version of MindSpore. For example, when you are installing MindSpore 1.1.0, `{version}` should be 1.1.0.

## Installation Verification

```python
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

## Version Update

Using the following command if you need to update the MindSpore version.

- Update online

    ```bash
    pip install --upgrade mindspore-gpu
    ```

- Update after source code compilation

     After successfully executing the compile script `build.sh` in the root path of the source code, find the whl package in path `build/package`, use the following command to update your version.

    ```bash
    pip install --upgrade mindspore_gpu-{version}-cp37-cp37m-linux_{arch}.whl
    ```

## Installing MindInsight

If you need to analyze information such as model scalars, graphs, computation graphs and model traceback, you can install MindInsight.

For more details, please refer to [MindInsight](https://gitee.com/mindspore/mindinsight/blob/r1.1/README.md).

## Installing MindArmour

If you need to conduct AI model security research or enhance the security of the model in you applications, you can install MindArmour.

For more details, please refer to [MindArmour](https://gitee.com/mindspore/mindarmour/blob/r1.1/README.md).

## Installing MindSpore Hub

If you need to access and experience MindSpore pre-trained models quickly, you can install MindSpore Hub.

For more details, please refer to [MindSpore Hub](https://gitee.com/mindspore/hub/blob/r1.1/README.md).
