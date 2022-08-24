# Installing MindSpore in Ascend 910 by Docker

<!-- TOC -->

- [Installing MindSpore in Ascend 910 by Docker](#installing-mindspore-in-ascend-910-by-docker)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Installing Ascend AI processor software package](#installing-ascend-ai-processor-software-package)
    - [Obtaining MindSpore Image](#obtaining-mindspore-image)
    - [Running MindSpore Image](#running-mindspore-image)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.7/install/mindspore_ascend_install_docker_en.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

[Docker](https://docs.docker.com/get-docker/) is an open source application container engine, and developers can package their applications and dependencies into a lightweight, portable container. By using Docker, MindSpore can be rapidly deployed and separated from the system environment.

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

## Installing Ascend AI processor software package

Ascend software package provides two distributions, commercial edition and community edition:

- Commercial edition needs approval from Ascend to download, for detailed installation guide, please refer to [Ascend Data Center Solution 22.0.RC1 Installation Guide](https://support.huawei.com/enterprise/zh/doc/EDOC1100246310)。

- Community edition has no restrictions, choose `5.1.RC1.alpha005` in [CANN community edition](https://www.hiascend.com/software/cann/community-history), then choose relevant driver and firmware packages in [firmware and driver](https://www.hiascend.com/hardware/firmware-drivers?tag=community). Please refer to the abovementioned commercial edition installation guide to choose which packages are to be installed and how to install them.

- After installing basic driver and corresponding software packages, ensure that the toolbox utility package in the CANN software package is installed, namely Ascend-cann-toolbox-{version}.run. The toolbox provides Ascend Docker runtime tools supported by Ascend NPU containerization.

The default installation path of the installation package is `/usr/local/Ascend`. Ensure that the current user has the right to access the installation path `/usr/local/Ascend` of Ascend AI processor software package, If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.

## Obtaining MindSpore Image

1. Log in to [Ascend Hub Image Center](https://ascend.huawei.com/ascendhub/#/home), and register and activate an account. Get login instructions and download instructions.
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
docker run -it -u root --ipc=host \
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
               -v /usr/local/Ascend/latest/driver:/usr/local/Ascend/latest/driver \
               -v /usr/local/Ascend/latest/add-ons/:/usr/local/Ascend/latest/add-ons/ \
               -v /var/log/npu/:/usr/slog \
               ascendhub.huawei.com/public-ascendhub/mindspore-modelzoo:{tag} \
               /bin/bash
```

of which,

- `{tag}` corresponds to the version number of Atlas Data Center Solution, which can be automatically obtained on the MindSpore image download page.

If you want to use MindInsight, you need to set the `--network` parameter to "host" mode, for example:

```bash
docker run -it -u root --ipc=host \
               --network host \
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
               -v /usr/local/Ascend/latest/driver:/usr/local/Ascend/latest/driver \
               -v /usr/local/Ascend/latest/add-ons/:/usr/local/Ascend/latest/add-ons/ \
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
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

So far, it means MindSpore Ascend 910 has been installed by Docker successfully.

ii:

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(device_target="Ascend")
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

So far, it means MindSpore Ascend 910 has been installed by Docker successfully.

If you need to verify the MindInsight installation:

Enter ```mindinsight start --port 8080```, if it prompts that the startup status is successful, it means MindInsight has been installed successfully.

## Version Update

When you need to update the MindSpore version:

- update corresponding Ascend AI processor software package according to MindSpore package version of which you wish to update.
- log in to [Ascend Hub Image Center](https://ascend.huawei.com/ascendhub/#/home) again to obtain the download command of the latest Docker and execute:

    ```bash
    docker pull ascendhub.huawei.com/public-ascendhub/mindspore-modelzoo:{tag}
    ```

    of which,

    - `{tag}` corresponds to the version number of Atlas Data Center Solution, which can be automatically obtained on the MindSpore image download page.
