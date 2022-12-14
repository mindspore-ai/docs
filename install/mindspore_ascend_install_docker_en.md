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

<a href="https://gitee.com/mindspore/docs/blob/r1.10/install/mindspore_ascend_install_docker_en.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png"></a>

[Docker](https://docs.docker.com/get-docker/) is an open source application container engine, and developers can package their applications and dependencies into a lightweight, portable container. By using Docker, MindSpore can be rapidly deployed and separated from the system environment.

This document describes how to quickly install MindSpore in a Linux system with an Ascend 910 environment by Docker.

The Docker image of MindSpore is hosted on [Huawei SWR](https://support.huaweicloud.com/swr/index.html).

The current support for containerized build options is as follows:

| Hardware   | Docker Image Hub                | Label                       | Note                                       |
| :----- | :------------------------ | :----------------------- | :--------------------------------------- |
| Ascend | `mindspore/mindspore-ascend` | `x.y.z` | The production environment of MindSpore Ascend x.y.z together with the corresponding version of Ascend Data Center Solution. |

> `x.y.z` corresponds to the MindSpore version number. For example, when MindSpore version 1.9.0 is installed, `x.y.z` should be written as 1.9.0.

## System Environment Information Confirmation

- Ensure that Ubuntu 18.04/CentOS 7.6 is installed with the 64-bit ARM architecture operating system.

- Ensure that [Docker 18.03 or later](https://docs.docker.com/get-docker/) is installed.

## Installing Ascend AI processor software package

Ascend software package provides two distributions, commercial edition and community edition:

- Commercial edition needs approval from Ascend to download, for detailed installation guide, please refer to [Ascend Data Center Solution 22.0.RC3 Installation Guide](https://support.huawei.com/enterprise/zh/doc/EDOC1100280094).

- Community edition has no restrictions, choose `6.0.RC1.alpha005` in [CANN community edition](https://www.hiascend.com/software/cann/community-history), then choose relevant driver and firmware packages in [firmware and driver](https://www.hiascend.com/hardware/firmware-drivers?tag=community). Please refer to the abovementioned commercial edition installation guide to choose which packages are to be installed and how to install them.

The default installation path of the installation package is `/usr/local/Ascend`. Ensure that the current user has the right to access the installation path `/usr/local/Ascend` of Ascend AI processor software package, If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.

## Obtaining MindSpore Image

For the `Ascend` backend, you can directly use the following command to obtain the latest stable image:

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-ascend:{tag}
```

of which,

- `{tag}` corresponds to the label in the above table.

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
               -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
               -v /var/log/npu/:/usr/slog \
               swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-ascend:{tag} \
               /bin/bash
```

of which,

- `{tag}` corresponds to the label in the above table.

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
               -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
               -v /var/log/npu/:/usr/slog \
               swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-ascend:{tag} \
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
- directly use the following command to obtain the latest stable image:

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-ascend:{tag}
```

of which,

- `{tag}` corresponds to the label in the above table.
