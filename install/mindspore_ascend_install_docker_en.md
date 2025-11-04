# Installing MindSpore in Ascend by Docker

<!-- TOC -->

- [Installing MindSpore in Ascend by Docker](#installing-mindspore-in-ascend-by-docker)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Installing Ascend AI processor software package](#installing-ascend-ai-processor-software-package)
    - [Obtaining MindSpore Image](#obtaining-mindspore-image)
    - [Running MindSpore Image](#running-mindspore-image)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)
    - [Notes](#notes)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.7.1/install/mindspore_ascend_install_docker_en.md)

[Docker](https://docs.docker.com/get-docker/) is an open source application container engine, and supports packaging developers' applications and dependency packages into a lightweight, portable container. By using Docker, MindSpore can be rapidly deployed and separated from the system environment.

This document describes how to install MindSpore by Docker on Linux in an Ascend environment.

The Docker image of MindSpore is hosted on [Huawei SWR](https://support.huaweicloud.com/swr/index.html).

The current support for containerized build options is as follows:

| Hardware   | Docker Namespace   | Image Name             | Label                       | Note                                       |
| :--------- | :------------------------ | :------------------------ | :----------------------- | :--------------------------------------- |
| Atlas Training Series‌| `mindspore` | `mindspore-ascend-a1` | `x.y.z` | The production environment of MindSpore Ascend x.y.z together with the corresponding version of Ascend Data Center Solution. |
| Atlas A2 Training Series‌| `mindspore` | `mindspore-ascend-a2` | `x.y.z` | The production environment of MindSpore Ascend x.y.z together with the corresponding version of Ascend Data Center Solution. |

> `x.y.z` corresponds to the MindSpore version number. For example, when MindSpore version 2.7.1 is installed, `x.y.z` should be written as 2.7.1.

## System Environment Information Confirmation

The following table outlines the system requirements for deploying MindSpore using Docker.

|Software Name|Version|Function|
|-|-|-|
|Debian series OS / openEuler series OS|Debianseries: Debian, Ubuntu, veLinux / openEuler serires: openEuler, CentOS, Kylin, BCLinux, UOS V20, AntOS, CTyunOS, CULinux, Tlinux, MTOS|‌Recommended OS for MindSpore Container Deployment|
|[Ascend AI processor software package](#installing-ascend-ai-processor-software-package)|CANN 8.3.RC1, CANN 8.2.RC1, CANN 8.1.RC1|Ascend platform AI computing library used by MindSpore|
|Docker | Docker 18.03+ |Provides lightweight containerization environment for isolated deployment and cross-platform execution of MindSpore and its dependencies|

## Installing Ascend AI processor software package

To install Ascend software package community edition, the recommended version is `8.3.RC1` in [CANN community edition](https://www.hiascend.com/developer/download/community/result?module=cann), then choose relevant driver and firmware packages in [firmware and driver](https://www.hiascend.com/hardware/firmware-drivers/community). Please refer to [Installation guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_quick.html) to choose which packages are to be installed and how to install them.

The default installation path of the installation package is `/usr/local/Ascend`. Ensure that the current user has the right to access the installation path `/usr/local/Ascend` of Ascend AI processor software package. If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.

## Obtaining MindSpore Image

For the `Ascend` backend, you can directly use the following command to obtain the latest stable image:

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/{image_name}:{tag}
```

of which,

- `{image_name}` corresponds to the image name in the above table. For Atlas Training Series A1 products, download the `mindspore-ascend-a1` image; for Atlas A2 Training Series products, download the `mindspore-ascend-a2` image.
- `{tag}` corresponds to the label in the above table.

To install MindSpore 2.7.1 on Atlas Training Platform, use the following command:

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-ascend-a1:2.7.1
```

To install MindSpore 2.7.1 on Atlas A2 Training Platform, use the following command:

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-ascend-a2:2.7.1
```

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
               -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
               -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
               -v /etc/ascend_install.info:/etc/ascend_install.info \
               -v /var/log/npu/:/usr/slog \
               -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
               -v /etc/hccn.conf:/etc/hccn.conf \
               swr.cn-south-1.myhuaweicloud.com/mindspore/{image_name}:{tag} \
               /bin/bash
```

of which,

- `{image_name}` corresponds to the image name in the above table. For Atlas Training Series A1 products, use `mindspore-ascend-a1` image; for Atlas A2 Training Series products, use `mindspore-ascend-a2` image.
- `{tag}` corresponds to the label in the above table.
- Description for parameters are listed below:

|Parameter|Description|
|-|-|
|--device|Mapping devices to the container. <br> /dev/davinciX: NPU device, X represents device ID, e.g. davinci0. <br> /dev/davinci_manager: Management device for NPU. <br> /dev/hisi_hdc: Management device for HDC. <br> /dev/devmm_svm: Management device for device memory.|
|-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi|Mapping NPU management interface `npu-smi` to the container.|
|-v /usr/local/Ascend/driver:/usr/local/Ascend/driver|Mapping host directory "/usr/local/Ascend/driver" to the container.|
|-v /etc/ascend_install.info:/etc/ascend_install.info|Mapping installation log for CANN software packages to the container.|
|-v /var/log/npu/:/usr/slog|Mapping NPU log to the container.|
|-v /usr/bin/hccn_tool:/usr/bin/hccn_tool|Mapping NPU communication configuration tool `hccn_tool` to the container.|
|-v /etc/hccn.conf:/etc/hccn.conf|Mapping hccn configuration file to the container.|

## Installation Verification

After entering the MindSpore container according to the above steps, to test whether the Docker container is working properly, please execute the following Python code and check the output:

**Method 1:**

Execute the following command:

```bash
python -c "import mindspore;mindspore.set_device('Ascend');mindspore.run_check()"
```

The outputs should be the same as:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!
```

So far, it means MindSpore Ascend has been installed by Docker successfully.

**Method 2:**

Execute the following command:

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_device("Ascend")
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

So far, it means MindSpore Ascend has been installed by Docker successfully.

## Version Update

When you need to update the MindSpore version:

- update corresponding Ascend AI processor software package according to MindSpore package version of which you wish to update.
- directly use the following command to obtain the latest stable image:

    ```bash
    docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/{image_name}:{tag}
    ```

    of which,

    - `{tag}` corresponds to the label in the above table.

## Notes

- When deploying containers in non-root user mode, it is essential to verify that the target NPU device is not occupied by other unprivileged containers. After startup, execute the `npu-smi` info command to check device status. If the target NPU device is already allocated to another non-root container, the following error will occur, You can add `-u root  --privileged` when creating the container.

```text
    DrvMngGetConsoleLogLevel failed. (g_conLogLevel=3)
    dcmi model initialized failed, because the device is used. ret is -802
```