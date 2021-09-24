# Installing MindSpore in Ascend 910 by Docker

<!-- TOC -->

- [Installing MindSpore in Ascend 910 by Docker](#installing-mindspore-in-ascend-910-by-docker)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Obtaining MindSpore Image](#obtaining-mindspore-image)
    - [Running MindSpore Image](#running-mindspore-image)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/install/mindspore_ascend_install_docker_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

[Docker](https://docs.docker.com/get-docker/) is an open source application container engine, developers can package their applications and dependencies into a lightweight, portable container. By using Docker, MindSpore can be rapidly deployed and separated from the system environment.

This document describes how to quickly install MindSpore in a Linux system with an Ascend 910 environment by Docker.

The Ascend 910 image of MindSpore is hosted on the [Ascend Hub](https://ascend.huawei.com/ascendhub/#/main).

The current support for containerized build options is as follows:

| Hardware   | Docker Image Hub                | Label                       | Note                                       |
| :----- | :------------------------ | :----------------------- | :--------------------------------------- |
| Ascend | `public-ascendhub/mindspore-modelzoo` | `x.y.z` | The production environment of MindSpore released together with the Ascend Data Center Solution `x.y.z` version is pre-installed. |

> `x.y.z` corresponds to the version number of Atlas Data Center Solution, which can be obtained on the Ascend Hub page.

## System Environment Information Confirmation

- Ensure that Ubuntu 18.04/CentOS 7.6 is installed with the 64-bit operating system.

- Ensure that [Docker 18.03 or later](https://docs.docker.com/get-docker/) is installed.

- Ensure that the Ascend 910 AI Processor software packages ([Ascend Data Center Solution 21.0.5]) are installed.

    - For the installation of software package,  please refer to the [Product Document](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-data-center-solution-pid-251167910).
    - The software packages include Driver and Firmware and CANN.
        - [Driver and Firmware A800-9000 1.0.13 ARM platform] and [Driver and Firmware A800-9010 1.0.13 x86 platform]
        - [CANN 5.0.T306](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/254156433)

    - Ensure that the current user has the right to access the installation path `/usr/local/Ascend`of Ascend 910 AI processor software package. If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located. For the specific configuration, please refer to the software package instruction document.
    - After installing basic driver and corresponding software packages, ensure that the toolbox utility package in the CANN software package is installed, namely Ascend-cann-toolbox-{version}.run. The toolbox provides Ascend Docker runtime tools supported by Ascend NPU containerization.

## Obtaining MindSpore Image

1. Log in to [Ascend Hub Image Center](https://ascend.huawei.com/ascendhub/#/home), register and activate an account, get login instructions and download instructions.
2. After obtaining the download permission, enter the [MindSpore image download page](https://ascendhub.huawei.com/#/detail/mindspore-modelzoo). Get login and download commands and execute:

    ```bash
    docker login -u {username} -p {password} {url}
    docker pull ascendhub.huawei.com/public-ascendhub/mindspore-modelzoo:{tag}
    ```

    of which,

    - `{username}` `{password}` `{url}` represents the user's login information and image server information, which are automatically generated after registering and activating the account, and can be obtained by copying the login command on the corresponding MindSpore image page.
    - `{tag}` corresponds to the version number of Atlas Data Center Solution, which can also be obtained by copying the download command on the MindSpore image download page.

## Running MindSpore Image

Execute the following command to start the Docker container instance:

```bash
docker run -it --ipc=host \
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
               -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
               -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
               -v /var/log/npu/:/usr/slog \
               ascendhub.huawei.com/public-ascendhub/mindspore-modelzoo:{tag} \
               /bin/bash
```

of which,

- `{tag}` corresponds to the version number of Atlas Data Center Solution, which can be automatically obtained on the MindSpore image download page.

If you want to use MindInsight, you need to set the `--network` parameter to "host" mode, for example:

```bash
docker run -it --ipc=host \
               --network host
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
               -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
               -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
               -v /var/log/npu/:/usr/slog \
               ascendhub.huawei.com/public-ascendhub/mindspore-modelzoo:{tag} \
               /bin/bash
```

## Installation Verification

After entering the MindSpore container according to the above steps, to test whether the Docker container is working properly, please run the following Python code and check the output:

i:

```bash
python -c "import mindspore;mindspore.run_check()"
```

The outputs should be the same as:

```text
mindspore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

It means MindSpore has been installed by docker successfully.

ii:

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="Ascend")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
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

If you need to verify the MindInsight installation:

Enter ```mindinsight start --port 8080```, if it prompts that the startup status is successful, it means MindInsight has been installed successfully.

## Version Update

When you need to update the MindSpore version:

- update Ascend 910 AI processor software package according to MindSpore package version of which you wish to update.
- log in to [Ascend Hub Image Center](https://ascend.huawei.com/ascendhub/#/home) again to obtain the download command of the latest docker version and execute:

    ```bash
    docker pull ascendhub.huawei.com/public-ascendhub/mindspore-modelzoo:{tag}
    ```

    of which,

    - `{tag}` corresponds to the version number of Atlas Data Center Solution, which can be automatically obtained on the MindSpore image download page.
