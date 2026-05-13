# Installing MindSpore in Ascend by Docker

<!-- TOC -->

- [Installing MindSpore in Ascend by Docker](#installing-mindspore-in-ascend-by-docker)
    - [Supported Tags and Dockerfile Usage](#supported-tags-and-dockerfile-usage)
        - [Tag Specification](#tag-specification)
        - [Image Repository Address](#image-repository-address)
    - [Quick Start](#quick-start)
        - [Obtaining MindSpore Image](#obtaining-mindspore-image)
        - [Running MindSpore Container](#running-mindspore-container)
        - [Building Arguments](#building-arguments)
        - [Building MindSpore Image](#building-mindspore-image)
        - [Running local build MindSpore Container](#running-local-build-mindspore-container)
        - [Installation Verification](#installation-verification)
        - [Version Update](#version-update)
        - [Notes](#notes)
        - [How to Extend for Custom Development](#how-to-extend-for-custom-development)
    - [Supported Hardware](#supported-hardware)
    - [License](#license)

<!-- /TOC -->

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.9.0/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/r2.9.0/install/mindspore_ascend_install_docker_en.md)

[Docker](https://docs.docker.com/get-docker/) is an open source application container engine, and supports packaging developers' applications and dependency packages into a lightweight, portable container. By using Docker, MindSpore can be rapidly deployed and separated from the system environment.

This document describes how to install MindSpore by Docker on Linux in an Ascend environment.

## Supported Tags and Dockerfile Usage

### Tag Specification

Tags follow this format:

```text
<MindSpore Version>-<Hardware Info (Chip)>-<Operating System>-<Python Version>
```

| Field | Example Values | Description |
|-------|---------------|-------------|
| MindSpore Version | 2.9.0 | Corresponds to the version identifier in MindSpore official release tags |
| Hardware Info (Chip) | see below | Ascend chip model identifier |
| Operating System | ubuntu22.04 / openeuler24.03 | Operating system distribution and version used in the base image |
| Python Version | py3.11 | Major Python version built into the image |

> Tips: System architecture is automatically detected via Docker Manifest, no need to specify in the tag.

### Image Repository Address

MindSpore Ascend images are hosted on Huawei Cloud SWR image repository:

```text
swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore
```

**Full Image Example:**

```text
swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:2.9.0-910b-ubuntu22.04-py3.11
```

## Quick Start

### Obtaining MindSpore Image

For `Ascend` backend, you can directly use the following command to obtain the latest stable image:

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:<MindSpore Version>-<Hardware Info (Chip)>-<Operating System>-<Python Version>
```

To install MindSpore 2.9.0 on Atlas A2 Training Platform, with Ubuntu 22.04 operating system, use the following command:

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:2.9.0-910b-ubuntu22.04-py3.11
```

To install MindSpore 2.9.0 on Atlas A3 Training Platform, with OpenEuler 24.04 operating system, use the following command:

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:2.9.0-a3-openeuler24.03-py3.11
```

### Running MindSpore Container

```bash
docker run \
    --privileged \
    --name mindspore_container \
    --device /dev/davinci1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:2.9.0-a3-openeuler24.03-py3.11 bash
```

### Building Arguments

| Parameter | Description | Required | Source | Example Values |
|-----------|-------------|----------|--------|----------------|
| CANN_VERSION | Ascend CANN toolkit version | Yes | CANN image tag | 9.0.0 |
| CHIP_ARCH | Ascend chip architecture identifier | Yes | Tag specification | see below |
| OS_SYSTEM | Base image operating system and version | Yes | Tag specification | ubuntu22.04 / openeuler24.03 |
| PY_VERSION | Python version built into the base image | Yes | Tag specification | py3.11 |
| MINDSPORE_VERSION | MindSpore version number | Yes | [MindSpore repository releases](https://atomgit.com/mindspore/mindspore/releases) | 2.9.0 |
| PIP_INDEX_URL | pip installation source URL (default: Huawei Cloud mirror) | No | PyPI mirror source | https://mirrors.huaweicloud.com/repository/pypi/simple |

### Building MindSpore Image

```bash
docker build \
    --build-arg CANN_VERSION=9.0.0 \
    --build-arg CHIP_ARCH=910b \
    --build-arg OS_SYSTEM=ubuntu22.04 \
    --build-arg PY_VERSION=py3.11 \
    --build-arg MINDSPORE_VERSION=2.9.0 \
    --build-arg PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
    -t mindspore:2.9.0-910b-ubuntu22.04-py3.11 \
    -f Dockerfile .
```

### Running a locally built MindSpore Container

```bash
docker run \
    --privileged \
    --name mindspore_container \
    --device /dev/davinci1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it mindspore:{tag} bash
```

Where,

- `{tag}` corresponds to the label designated when building MindSpore image, for example, `mindspore:2.9.0-910b-ubuntu22.04-py3.11`.

### Installation Verification

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

### Version Update

When you need to update the MindSpore version:

- update corresponding Ascend AI processor software package according to MindSpore package version of which you wish to update.
- directly use the following command to obtain the latest stable image:

    ```bash
    docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:<MindSpore Version>-<Hardware Info (Chip)>-<Operating System>-<Python Version>
    ```

### Notes

- When deploying containers in non-root user mode, it is essential to verify that the target NPU device is not occupied by other unprivileged containers. After startup, execute the `npu-smi` info command to check device status. If the target NPU device is already allocated to another non-root container, the following error will occur, You can add `-u root  --privileged` when creating the container.

```text
    DrvMngGetConsoleLogLevel failed. (g_conLogLevel=3)
    dcmi model initialized failed, because the device is used. ret is -802
```

### How to Extend for Custom Development

```bash
# Use MindSpore image as base image, add user software
FROM swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore:2.9.0-910b-ubuntu22.04-py3.11

RUN apt update -y && \
    apt install -y gcc g++

# Install additional dependencies
RUN pip install pandas scikit-learn

# Copy user code
WORKDIR /workspace
COPY . /workspace

CMD ["python", "train.py"]
```

## Supported Hardware

| Chip Series | Product Examples | Architecture |
|-------------|-----------------|--------------|
| Atlas A2 | Atlas 800T A2, Atlas 900 A2 PoD | Auto-detected (ARM64/x86_64) |
| Atlas A3 | Atlas 800T A3 | Auto-detected (ARM64/x86_64) |

> Tips: Use the `docker manifest inspect` command to view the system architectures supported by an image.

## License

View the [license information](https://atomgit.com/mindspore/mindspore/blob/master/LICENSE) for MindSpore included in these images.

As with all container images, pre-installed software packages (Python, system libraries, etc.) may be subject to their respective licenses.
