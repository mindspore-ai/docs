<!-- __SG_DEPRECATED_BANNER__ -->

```{admonition} Deprecated
:class: warning

This page belongs to the **Static Graph (GRAPH_MODE) Implementation** section and has been marked as deprecated. New features are primarily being developed in the "r2.0.0 Dynamic Graph Implementation" section. Please refer to the dynamic graph documentation first.
```

# Practical Case: Creating a Docker Image for MindSpore Transformers

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/static_graph/example/docker-installation.md)

This case shares the practice of creating a Docker image for **MindSpore Transformers**. Developers can create their own images by referring to this case.

> The solution and software packages provided in this case come from the open-source community and are for reference only. To use the image created by this case for commercial purposes, such as deployment, in a production environment, you need to ensure the reliability and security of the image. MindSpore Transformers is not responsible for network security. Use the image in a trusted environment.

## Environment Preparations

Before building an image, you need to prepare the host environment, including the hardware, software, and network. This step ensures that the build process goes smoothly.

### System Requirements

- **Hardware requirements**: The NPU driver and firmware must be installed on the host. For details, see [Ascend Community > Installing the NPU Driver and Firmware](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/softwareinst/instg/instg_0005.html?Mode=DockerIns&InstallType=local&OS=Debian&Software=cannToolKit)

- **Software requirements**: Docker `26.1.4` or later

- **Network requirements**: Stable Internet connection; Access to Huawei Cloud (for downloading CANN, MindSpore, and more); Prolonged build time when the network is slow.

> Ensure that the host time and time zone are correct to avoid download problems.

## Tool Installation

Verify the installation of the tool.

```bash
docker --version
```

If no version information is displayed, install the tool according to the official guide.

- [Install Docker Engine](https://docs.docker.com/engine/install/)

## Base Image Selection

- In this case, Dockerfiles use `ubuntu:24.04` as the base image.
- **Multiphase building** is used.

  1. Phase 1: Install Python.
  2. Phase 2: Install CANN.
  3. Final phase: Install MindSpore and MindSpore Transformers and integrate the results.

This multiphase approach helps reduce the final image size and improve the build efficiency.

For details about the Dockerfile content, see [Community Issues](https://atomgit.com/mindspore/mindformers/issues/2231).

Save Dockerfiles to the local PC.

## Image Build

Build the **MindSpore Transformers** image as follows:

- Create a folder.

  ```shell
  # Create a directory for storing Dockerfiles and enter the directory.
  mkdir -p mindformers-Dockerfiles
  cd mindformers-Dockerfiles
  ```

- Save Dockerfiles to the following directory:

  ```text
  mindformers-Dockerfiles/
  └── Dockerfile
  ```

- Run the Docker build command.

  The general commands are as follows:

  ```bash
  # Build an image.
  docker build -f Dockerfile \
    --build-arg PYTHON_VERSION="Python version" \
    --build-arg CANN_TOOLKIT_URL="CANN toolkit download link" \
    --build-arg CANN_KERNELS_URL="CANN kernels download link" \
    --build-arg MS_WHL_URL="MindSpore wheel package download link" \
    --build-arg MINDFORMERS_GIT_REF="MindSpore Transformers code repository branch name" \
    -t "Image name:Tag" .
  ```

  The following is an example for MindSpore Transformers 1.6.0:

  ```bash
  # Build the image. The tag naming mode is for reference only. The version information is included for easy management.
  docker build -f Dockerfile \
    --build-arg PYTHON_VERSION="3.11.4" \
    --build-arg CANN_TOOLKIT_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.2.RC1/Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run" \
    --build-arg CANN_KERNELS_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.2.RC1/Ascend-cann-kernels-910b_8.2.RC1_linux-aarch64.run" \
    --build-arg MS_WHL_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0/MindSpore/unified/aarch64/mindspore-2.7.0-cp311-cp311-linux_aarch64.whl" \
    --build-arg MINDFORMERS_GIT_REF="r1.6.0" \
    -t "mindformers:r1.6.0_ms2.7.0_cann8.2.RC1_py3.11" .
  ```

### Parameters

| Parameter| Description| URL|
|------|------|----------|
| `PYTHON_VERSION` | Python version.| [Python website](https://www.python.org/downloads/)|
| `CANN_TOOLKIT_URL` | URL for downloading the CANN Toolkit package.| [Ascend community download page](https://www.hiascend.com/developer/download/community/result?module=cann)|
| `CANN_KERNELS_URL` | URL for downloading the CANN kernels package.| [Ascend community download page](https://www.hiascend.com/developer/download/community/result?module=cann)|
| `MS_WHL_URL` | URL for downloading the MindSpore wheel package.| [MindSpore PyPI](https://repo.mindspore.cn/pypi/simple/mindspore/) |
| `MINDFORMERS_GIT_REF` | MindFormers branch name. The corresponding branch is automatically checked out.| [MindFormers repository](https://atomgit.com/mindspore/mindformers)|

> The build process may take about 30 minutes, depending on the network speed and hardware performance.

## Verification of the Build

Check whether the image is built successfully.

```bash
# Search for a specific image.
docker images | grep mindformers
```

Example:

```text
REPOSITORY    TAG                                IMAGE ID       CREATED        SIZE
mindformers   r1.6.0_ms2.7.0_cann8.2.RC1_py3.11  67fa2e821694   19 hours ago   14GB
```

## Examples

### Starting a Development Container

```bash
docker run -itd \
  --hostname $(hostname -I | awk '{print $1}' | tr '.' '-') \
  --ipc=host \
  --network=host \
  --device=/dev/davinci0:rwm \
  --device=/dev/davinci1:rwm \
  --device=/dev/davinci2:rwm \
  --device=/dev/davinci3:rwm \
  --device=/dev/davinci4:rwm \
  --device=/dev/davinci5:rwm \
  --device=/dev/davinci6:rwm \
  --device=/dev/davinci7:rwm \
  --device=/dev/davinci_manager:rwm \
  --device=/dev/devmm_svm:rwm \
  --device=/dev/hisi_hdc:rwm \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /var/log/npu/:/usr/slog \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
  -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common \
  -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /etc/localtime:/etc/localtime \
  --name Container name. \
  Image name. \
  /bin/bash
```

## Security Risks

When using Docker containers to run MindSpore Transformers, be aware of the following security risks:

- **Running as the root user**: Containers run as the **root** user by default, which may bring security risks. You are advised to create non-privileged users in the production environment to run applications.

- **Lack of CPU and memory resource limits**: If resource limits are not set, containers may consume excessive system resources, affecting the host performance. You are advised to use the `--cpus` and `--memory` parameters to limit resource usage.

- **`rwm` permissions for devices**: Read, write, and mknod permissions are allocated to NPU devices. Although the permissions are required for functions to run, the permission scope should be carefully evaluated in security-sensitive environments.

> In the production environment, adjust the container configurations based on the actual security requirements to ensure system security.

## References

- [MindSpore website](https://www.mindspore.cn/en)
- [MindSpore Transformers repository](https://atomgit.com/mindspore/mindformers)
- [Docker documentation](https://docs.docker.com)
- [Ascend community](https://www.hiascend.com/developer)
- [MindSpore community](https://atomgit.com/mindspore/community)
- [Related issues](https://gitee.com/mindspore/mindformers/issues/ICQ9JF)
