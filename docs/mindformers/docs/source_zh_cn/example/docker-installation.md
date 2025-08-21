# 制作 MindSpore Transformers 的 Docker 镜像的实践案例

本案例将分享构建 **MindSpore Transformers** 的 Docker 镜像的实践，开发者可以参考本案例构建自己的镜像。

> 本案例提供制作镜像的方案和软件包均来源于开源社区，仅供参考。用户参考本案例制作的镜像，如需用于生产环境部署等商用行为，需自行保障镜像的可靠性、安全性等，MindSpore Transformers 不对其网络安全性负责，请在可信的环境中使用。

## 环境准备

在构建镜像前，需要准备主机环境，包括硬件、软件和网络。这一步确保构建顺利进行。

### 系统要求

- **硬件要求**：宿主机需安装 NPU 驱动和固件。参考文档：[昇腾社区-安装NPU驱动和固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/softwareinst/instg/instg_0005.html?Mode=DockerIns&InstallType=local&OS=Debian&Software=cannToolKit)。

- **软件要求**：Docker 版本：`26.1.4`；Git：用于克隆仓库.

- **网络要求**：稳定的互联网连接；能访问华为云（下载 CANN、MindSpore 等）；网络慢时，构建时间会延长.

> 确保主机时间和时区正确，以避免下载问题。

## 工具安装

验证安装以下工具：

```bash
docker --version
git --version
```

若没有显示版本信息，请根据官方指导安装：

- [Docker 官方安装教程](https://docs.docker.com/engine/install/)
- [Git 下载](https://git-scm.com/downloads/linux)

## 基础镜像选择

- 本案例 Dockerfile 使用 `ubuntu:24.04` 作为基础镜像
- 采用 **多阶段构建**：

  1. 第一阶段安装 Python
  2. 第二阶段安装 CANN
  3. 最终阶段整合结果

这样可以减少最终镜像大小，并提高构建效率。

DockerFile的内容可参考社区 issue：[https://gitee.com/mindspore/mindformers/issues/ICQ9JF](https://gitee.com/mindspore/mindformers/issues/ICQ9JF)

并将其中的Dockerfile保存到本地。

## 镜像构建步骤

根据以下内容构建 **MindSpore Transformers** 镜像：

- 创建文件夹

  ```shell
  # 指定镜像名称和标签，这里以 MindSpore Transformers r1.6.0 + MindSpore 2.7.0 + CANN 8.2.RC1 + Python 3.11 为例
  # 命名格式为：仓库名为mindformers，tag为<mf_ver>_<ms ver>_<cann ver>_<py ver>
  IMAGE="mindformers:r1.6.0_ms2.7.0_cann8.2.RC1_py3.11"

  # 创建并进入存放 Dockerfile 的目录
  mkdir -p mindformers-Dockerfiles
  cd mindformers-Dockerfiles
  ```

- 在[社区issue](https://gitee.com/mindspore/mindformers/issues/ICQ9JF)中将Dockerfile保存到文件夹内，保存为以下格式：

  ```text
  mindformers-Dockerfiles/
  └── Dockerfile
  ```

- 设置必要的构建参数

  ```bash
  # 设置镜像名称和标签
  # 命名格式：仓库名为 mindformers，tag 为 <mf_ver>_<ms_ver>_<cann_ver>_<py_ver>
  export IMAGE="mindformers:r1.6.0_ms2.7.0_cann8.2.RC1_py3.11"
  # 设置构建参数
  export PYTHON_VERSION="3.11.4"
  export CANN_TOOLKIT_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.2.RC1/Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run"
  export CANN_KERNELS_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.2.RC1/Ascend-cann-kernels-910b_8.2.RC1_linux-aarch64.run"
  export MS_WHL_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0/MindSpore/unified/aarch64/mindspore-2.7.0-cp311-cp311-linux_aarch64.whl"
  export MINDFORMERS_GIT_REF="r1.6.0"
  ```

- 运行 Docker 构建命令

  ```bash
  # 开始构建镜像
  docker build -f Dockerfile \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg CANN_TOOLKIT_URL="${CANN_TOOLKIT_URL}" \
    --build-arg CANN_KERNELS_URL="${CANN_KERNELS_URL}" \
    --build-arg MS_WHL_URL="${MS_WHL_URL}" \
    --build-arg MINDFORMERS_GIT_REF="${MINDFORMERS_GIT_REF}" \
    -t "${IMAGE}" .
  ```

> 构建过程可能需要 30 分钟左右，取决于网络速度和硬件性能。

### 参数说明

| 参数 | 说明 | 获取地址 |
|------|------|----------|
| `CANN_TOOLKIT_URL` | CANN toolkit包下载地址 | [昇腾社区下载页](https://www.hiascend.com/developer/download/community/result?module=cann) |
| `CANN_KERNELS_URL` | CANN kernels包下载地址 | [昇腾社区下载页](https://www.hiascend.com/developer/download/community/result?module=cann) |
| `MS_WHL_URL` | MindSpore wheel 包地址 | [MindSpore PyPI](https://repo.mindspore.cn/pypi/simple/mindspore/) |
| `MINDFORMERS_GIT_REF` | MindSpore Transformers 分支名称 | [MindSpore Transformers 仓库](https://gitee.com/mindspore/mindformers) |

## 验证构建

查看镜像是否成功：

```bash
# 查找特定镜像
docker images | grep mindformers
```

期望输出示例：

```text
REPOSITORY    TAG                                IMAGE ID       CREATED        SIZE
mindformers   r1.6.0_ms2.7.0_cann8.2.RC1_py3.11  67fa2e821694   19 hours ago   14GB
```

## 使用示例

### 启动开发容器

```bash
image_name=mindformers:r1.6.0_ms2.7.0_cann8.2.RC1_py3.11
container_name=容器名称
docker run -itd \
  --hostname $(hostname -I | awk '{print $1}' | tr '.' '-') \
  --ipc=host \
  --network=host \
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
  -v /usr/local/dcmi:/usr/local/dcmi \
  --privileged \
  -v /var/log/npu/:/usr/slog \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
  -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common \
  -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /etc/localtime:/etc/localtime \
  --name "$container_name"
  "$image_name"
  /bin/bash
```

## 参考资源

- [MindSpore 官网](https://www.mindspore.cn)
- [MindSpore Transformers 仓库](https://gitee.com/mindspore/mindformers)
- [Docker 官方文档](https://docs.docker.com)
- [Ascend 社区](https://www.hiascend.com/developer)
- [MindSpore 社区](https://gitee.com/mindspore/community)
- [相关 issue](https://gitee.com/mindspore/mindformers/issues/ICQ9JF)
