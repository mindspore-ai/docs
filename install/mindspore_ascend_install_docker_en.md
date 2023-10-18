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

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.1/install/mindspore_ascend_install_docker_en.md)

[Docker](https://docs.docker.com/get-docker/) is an open source application container engine, and developers can package their applications and dependencies into a lightweight, portable container. By using Docker, MindSpore can be rapidly deployed and separated from the system environment.

This document describes how to install MindSpore by Docker on Linux in an Ascend 910 environment.

The Docker image of MindSpore is hosted on [Huawei SWR](https://support.huaweicloud.com/swr/index.html).

The current support for containerized build options is as follows:

| Hardware   | Docker Image Hub                | Label                       | Note                                       |
| :----- | :------------------------ | :----------------------- | :--------------------------------------- |
| Ascend | `mindspore/mindspore-ascend` | `x.y.z` | The production environment of MindSpore Ascend x.y.z together with the corresponding version of Ascend Data Center Solution. |

> `x.y.z` corresponds to the MindSpore version number. For example, when MindSpore version 2.1.1 is installed, `x.y.z` should be written as 2.1.1.

## System Environment Information Confirmation

- Ensure that Ubuntu 18.04/CentOS 7.6 is installed with the 64-bit ARM architecture operating system.

- Ensure that [Docker 18.03 or later](https://docs.docker.com/get-docker/) is installed.

## Installing Ascend AI processor software package

Ascend software package provides two distributions, commercial edition and community edition:

- Commercial edition needs approval from Ascend to download, for detailed installation guide, please refer to [Ascend Training Solution 23.0.RC2 Installation Guide](https://support.huawei.com/enterprise/zh/doc/EDOC1100318640).

- Community edition has no restrictions, choose `6.3.RC2.alpha005` in [CANN community edition](https://www.hiascend.com/software/cann/community-history), then choose relevant driver and firmware packages in [firmware and driver](https://www.hiascend.com/hardware/firmware-drivers?tag=community). Please refer to the abovementioned commercial edition installation guide to choose which packages are to be installed and how to install them.

The default installation path of the installation package is `/usr/local/Ascend`. Ensure that the current user has the right to access the installation path `/usr/local/Ascend` of Ascend AI processor software package. If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.

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

If you want to use MindSpore Insight, you need to set the `--network` parameter to "host" mode, for example:

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
python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"
```

The outputs should be the same as:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!
```

So far, it means MindSpore Ascend 910 has been installed by Docker successfully.

> The Ascend 910B does not support this verification method.

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

If you need to verify the MindSpore Insight installation:

Enter ```mindinsight start --port 8080```, if it prompts that the startup status is successful, it means MindSpore Insight has been installed successfully.

> The Ascend 910B does not support this verification method.

The current version is not fully verified on Ascend 910B. If you want to use Ascend 910B to verify whether the installation is successful, it is recommended to perform the following steps, and the detailed reasons can be found in [FAQ](https://www.mindspore.cn/docs/en/r2.1/faq/installation.html#verifying-the-installation).

1. Configure the following environment variables:

    ```bash
    export MS_ENABLE_GE=1
    export MS_GE_TRAIN=1 # not setting by default. The training network needs to be set to 1
    ```

2. Execute the following sample:

    ```python
    import numpy as np
    import mindspore as ms
    from mindspore import ops, nn

    ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
        def construct(self, x, y):
            return ops.add(x, y)
    x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
    y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
    net = Net()
    print(net(x, y))
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

When you need to update the MindSpore version:

- update corresponding Ascend AI processor software package according to MindSpore package version of which you wish to update.
- directly use the following command to obtain the latest stable image:

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-ascend:{tag}
```

of which,

- `{tag}` corresponds to the label in the above table.
