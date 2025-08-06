# Installing MindSpore in CPU by Docker

<!-- TOC -->

- [Installing MindSpore in CPU by Docker](#installing-mindSpore-in-cpu-by-docker)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Obtaining MindSpore Image](#obtaining-mindspore-image)
    - [Running MindSpore Image](#running-mindspore-image)
    - [Installation Verification](#installation-verification)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/install/mindspore_cpu_install_docker_en.md)

[Docker](https://docs.docker.com/get-docker/) is an open source application container engine, and supports packaging developers' applications and dependency packages into a lightweight, portable container. By using Docker, MindSpore can be rapidly deployed and separated from the system environment.

This document describes how to install MindSpore by Docker on Linux in a CPU environment.

The Docker image of MindSpore is hosted on [Huawei SWR](https://support.huaweicloud.com/swr/index.html).

The current support for the containerization build option is as follows:

| Hardware   | Docker Image Hub                | Label                       | Note                                       |
| :----- | :------------------------ | :----------------------- | :--------------------------------------- |
| CPU    | `mindspore/mindspore-cpu` | `x.y.z`                  | A production environment with the MindSpore `x.y.z` CPU version pre-installed.       |
|        |                           | `devel`                  | Provide a development environment to build MindSpore from the source (`CPU` backend). For installation details, please refer to <https://www.mindspore.cn/install/en>. |
|        |                           | `runtime`                | Provide runtime environment, MindSpore binary package (`CPU` backend) is not installed.         |

> `x.y.z` corresponds to the MindSpore version number. For example, when MindSpore version 1.1.0 is installed, `x.y.z` should be written as 1.1.0.

## System Environment Information Confirmation

- Ensure that a 64-bit Linux operating system with the x86 architecture is installed, where Ubuntu 18.04 is verified.
- Ensure that [Docker 18.03 or later versioin](https://docs.docker.com/get-docker/) is installed.

## Obtaining MindSpore Image

For the `CPU` backend, you can directly use the following command to obtain the latest stable image:

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-cpu:{tag}
```

of which,

- `{tag}` corresponds to the label in the above table.

## Running MindSpore Image

Execute the following command to start the Docker container instance:

```bash
docker run -it swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-cpu:{tag} /bin/bash
```

of which,

- `{tag}` corresponds to the label in the above table.

## Installation Verification

- If you are installing the container of the specified version `x.y.z`.

    After entering the MindSpore container according to the above steps, to test whether the Docker is working properly, please run the following Python code and check the output:

    **Method 1:**

    Execute the following command:

    ```bash
    python -c "import mindspore;mindspore.set_device(device_target='CPU');mindspore.run_check()"
    ```

    - The outputs should be the same as:

    ```text
    MindSpore version: __version__
    The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
    ```

    So far, it means MindSpore CPU has been installed by Docker successfully.

    **Method 2:**

    Execute the following command:

    ```python
    import numpy as np
    import mindspore as ms
    import mindspore.ops as ops

    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device(device_target="CPU")

    x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
    y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
    print(ops.add(x, y))
    ```

    When the code is successfully run, the outputs should be the same as:

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

    So far, it means MindSpore CPU has been installed by Docker successfully.

- If you install a container with the label of `runtime`, you need to install MindSpore yourself.

    Go to [MindSpore Installation Guide Page](https://www.mindspore.cn/install/en), choose the CPU hardware platform, Linux-x86_64 operating system and pip installation method to get the installation guide. Refer to the installation guide after running the container and install the MindSpore CPU version by pip, and verify it.

- If you install a container with the label of `devel`, you need to compile and install MindSpore yourself.

    Go to [MindSpore Installation Guide Page](https://www.mindspore.cn/install/en), and choose the CPU hardware platform, Linux-x86_64 operating system and pip installation method to get the installation guide. After running the container, download the MindSpore code repository and refer to the installation guide, install the MindSpore CPU version through source code compilation, and verify it.

If you want to know more about the MindSpore Docker image building process, please check [docker repo](https://gitee.com/mindspore/mindspore/blob/br_base/scripts/docker/README.md#) for details.
