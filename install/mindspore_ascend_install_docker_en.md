# Installing MindSpore in Ascend 910 by Docker

<!-- TOC -->

- [Installing MindSpore in Ascend 910 by Docker](#installing-mindspore-in-ascend-910-by-docker)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Obtaining MindSpore Image](#obtaining-mindspore-image)
    - [Running MindSpore Image](#running-mindspore-image)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.1/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/install/mindspore_ascend_install_docker_en.md)

[Docker](https://docs.docker.com/get-docker/) is an open source application container engine, developers can package their applications and dependencies into a lightweight, portable container. By using Docker, MindSpore can be rapidly deployed and separated from the system environment.

This document describes how to quickly install MindSpore in a Linux system with an Ascend 910 environment by Docker.

The Ascend 910 image of MindSpore is hosted on the [Ascend Hub](https://ascend.huawei.com/ascendhub/#/main).

The current support for containerized build options is as follows:

| Hardware   | Docker Image Hub                | Label                       | Note                                       |
| :----- | :------------------------ | :----------------------- | :--------------------------------------- |
| Ascend | `public-ascendhub/ascend-mindspore-arm` | `x.y.z` | The production environment of MindSpore released together with the Ascend Data Center Solution `x.y.z` version is pre-installed. |

> `x.y.z` corresponds to the version number of Atlas Data Center Solution, which can be obtained on the Ascend Hub page.

## System Environment Information Confirmation

- Confirm that Ubuntu 18.04/CentOS 8.2 is installed with the 64-bit operating system.
- Confirm that [Docker 18.03 or later](https://docs.docker.com/get-docker/) is installed.
- Confirm that the Ascend 910 AI processor software package ([Atlas Data Center Solution V100R020C20](https://support.huawei.com/enterprise/en/ascend-computing/atlas-data-center-solution-pid-251167910/software/251826872)) are installed.
    - Confirm that the current user has the right to access the installation path `/usr/local/Ascend`of Ascend 910 AI processor software package. If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located. For the specific configuration, please refer to the software package instruction document.
    - After installing basic driver and corresponding software packages, confirm that the toolbox utility package in the CANN software package is installed, namely Ascend-cann-toolbox-{version}.run. The toolbox provides Ascend Docker runtime tools supported by Ascend NPU containerization.

## Obtaining MindSpore Image

1. Log in to [Ascend Hub Image Center](https://ascend.huawei.com/ascendhub/#/home), register and activate an account, get login instructions and pull instructions.
2. After obtaining the download permission, enter the MindSpore image download page ([x86 version](https://ascend.huawei.com/ascendhub/#/detail?name=ascend-mindspore-x86), [arm version](https://ascend.huawei.com/ascendhub/#/detail?name=ascend-mindspore-arm)). Get login and download commands and execute:

    ```bash
    docker login -u {username} -p {password} {url}
    docker pull swr.cn-south-1.myhuaweicloud.com/public-ascendhub/ascend-mindspore-{arch}:{tag}
    ```

    of which,

    - `{username}` `{password}` `{url}` represents the user's login information and image server information, which are automatically generated after registering and activating the account, and can be obtained by copying the login command on the corresponding MindSpore image page.
    - `{arch}` denotes the system architecture. For example, the Linux system you are using is x86 architecture 64-bit, {arch} should be x86. If the system is ARM architecture 64-bit, then it should be arm.
    - `{tag}` corresponds to the version number of Atlas Data Center Solution, which can also be obtained by copying the download command on the MindSpore image download page.

## Running MindSpore Image

Execute the following command to start the Docker container instance:

```bash
docker run -it -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
               -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
               -v /var/log/npu/:/usr/slog \
               --device=/dev/davinci0 \
               --device=/dev/davinci1 \
               --device=/dev/davinci2 \
               --device=/dev/davinci3 \
               --device=/dev/davinci4 \
               --device=/dev/davinci5 \
               --device=/dev/davinci6 \
               --device=/dev/davinci7 \
               --device=/dev/davinci_manager \
               --device=/dev/devmm_svm \
               --device=/dev/hisi_hdc \
               swr.cn-south-1.myhuaweicloud.com/public-ascendhub/ascend-mindspore-{arch}:{tag} \
               /bin/bash
```

of which,

- `{arch}` denotes the system architecture. For example, the Linux system you are using is x86 architecture 64-bit, {arch} should be x86. If the system is ARM architecture 64-bit, then it should be arm.
- `{tag}` corresponds to the version number of Atlas Data Center Solution, which can be automatically obtained on the MindSpore image download page.

## Installation Verification

After entering the MindSpore container according to the above steps, to test whether the Docker container is working properly, please run the following Python code and check the output:

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="Ascend")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.tensor_add(x, y))
```

The outputs should be the same as:

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

It means MindSpore has been installed by docker successfully.

## Version Update

When you need to update the MindSpore version:

- update Ascend 910 AI processor software package according to MindSpore package version of which you wish to update.
- log in to [Ascend Hub Image Center](https://ascend.huawei.com/ascendhub/#/home) again to obtain the download command of the latest docker version and execute:

    ```bash
    docker pull swr.cn-south-1.myhuaweicloud.com/public-ascendhub/ascend-mindspore-{arch}:{tag}
    ```

    of which,

    - `{arch}` denotes the system architecture. For example, the Linux system you are using is x86 architecture 64-bit, {arch} should be x86. If the system is ARM architecture 64-bit, then it should be arm.
    - `{tag}` corresponds to the version number of Atlas Data Center Solution, which can also be obtained by copying the download command on the MindSpore image download page.
