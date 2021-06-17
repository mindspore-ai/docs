# Installing MindSpore in GPU by Docker

<!-- TOC -->

- [Installing MindSpore in GPU by Docker](#installing-mindSpore-in-gpu-by-docker)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [nvidia-container-toolkit Installation](#nvidia-container-toolkit-installation)
    - [Obtaining MindSpore Image](#obtaining-mindspore-image)
    - [Running MindSpore Image](#running-mindspore-image)
    - [Installation Verification](#installation-verification)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.2/install/mindspore_gpu_install_docker_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.2/resource/_static/logo_source.png"></a>

[Docker](https://docs.docker.com/get-docker/) is an open source application container engine, developers can package their applications and dependencies into a lightweight, portable container. By using Docker, MindSpore can be rapidly deployed and separated from the system environment.

This document describes how to quickly install MindSpore by Docker in a Linux system with a GPU environment.

The Docker image of MindSpore is hosted on [Huawei SWR](https://support.huaweicloud.com/swr/index.html).

The current support for containerized build is as follows:

| Hardware   | Docker Image Hub                | Label                       | Note                                       |
| :----- | :------------------------ | :----------------------- | :--------------------------------------- |
| GPU    | `mindspore/mindspore-gpu` | `x.y.z`                  | A production environment with the MindSpore `x.y.z` GPU version pre-installed.       |
|        |                           | `devel`                  | Provide a development environment to build MindSpore from the source (`GPU CUDA10.1` backend). For installation details, please refer to <https://www.mindspore.cn/install/en>. |
|        |                           | `runtime`                | Provide runtime environment, MindSpore binary package (`GPU CUDA10.1` backend) is not installed. |

> **Note:** It is not recommended to install the whl package directly after building the GPU `devel` Docker image from the source. We strongly recommend that you transfer and install the `whl` package in the GPU `runtime` Docker image.
> `x.y.z` corresponds to the MindSpore version number. For example, when installing MindSpore version 1.1.0, `x.y.z` should be written as 1.1.0.

## System Environment Information Confirmation

- Confirm that Ubuntu 18.04 is installed with the 64-bit operating system.
- Confirm that [Docker 18.03 or later versioin](https://docs.docker.com/get-docker/) is installed.

## nvidia-container-toolkit Installation

For the `GPU` backend, please make sure that `nvidia-container-toolkit` has been installed in advance. The following is the installation guide for `nvidia-container-toolkit` for `Ubuntu` users:

```bash
# Acquire version of operating system version
DISTRIBUTION=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$DISTRIBUTION/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit nvidia-docker2
sudo systemctl restart docker
```

daemon.json is the configuration file of Docker. Edit the file daemon.json to configure the container runtime so that Docker can use nvidia-container-runtime:

```bash
$ vim /etc/docker/daemon.json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

Restart Docker:

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

## Obtaining MindSpore Image

For the `CPU` backend, you can directly use the following command to obtain the latest stable image:

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-gpu:{tag}
```

of which,

- `{tag}` corresponds to the label in the above table.

## Running MindSpore Image

Execute the following command to start the Docker container instance:

```bash
docker run -it -v /dev/shm:/dev/shm --runtime=nvidia --privileged=true swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-gpu:{tag} /bin/bash
```

of which,

- `-v /dev/shm:/dev/shm` mounts the directory where the NCCL shared memory segment is located into the container;
- `--runtime=nvidia` is used to specify the container runtime as `nvidia-container-runtime`;
- `--privileged=true` enables the container to expand;
- `{tag}` corresponds to the label in the above table.

If you want to use MindInsight, you need to set the --network parameter to "host" mode, for example:

```bash
docker run -it -v /dev/shm:/dev/shm --network host --runtime=nvidia --privileged=true swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-gpu:{tag} /bin/bash
```

## Installation Verification

- If you are installing the container of the specified version `x.y.z`.

    After entering the MindSpore container according to the above steps, to test whether the Docker container is working properly, please run the following Python code and check the output:

    ```python
    import numpy as np
    import mindspore.context as context
    import mindspore.ops as ops
    from mindspore import Tensor

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

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

- If you need to verify the MindInsight installation:

    1. If you install a container with the label of `1.2.0`, you need execute the command: ```export PATH=/usr/local/python-3.7.5/bin:$PATH```.

    2. Enter ```mindinsight start --port 8080```, if it prompts that the startup status is successful, it means MindInsight has been installed successfully.

- If you install a container with the label of `runtime`, you need to install MindSpore yourself.

    Go to [MindSpore Installation Guide Page](https://www.mindspore.cn/install/en), choose the GPU hardware platform, Ubuntu-x86 operating system and pip installation method to get the installation guide. Refer to the installation guide after running the container and install the MindSpore GPU version by pip, and verify it.

- If you install a container with the label of `devel`, you need to compile and install MindSpore yourself.

    Go to [MindSpore Installation Guide Page](https://www.mindspore.cn/install/en), choose the GPU hardware platform, Ubuntu-x86 operating system and pip installation method to get the installation guide. After running the container, download the MindSpore code repository and refer to the installation guide, install the MindSpore GPU version through source code compilation, and verify it.

If you want to know more about the MindSpore Docker image building process, please check [docker repo](https://gitee.com/mindspore/mindspore/blob/r1.2/docker/README.md) for details.
