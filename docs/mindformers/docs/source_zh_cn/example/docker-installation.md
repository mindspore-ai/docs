# 制作 MindSpore Transformers 的 Docker 镜像的实践案例

本案例将分享构建 **MindSpore Transformers** 的 Docker 镜像的实践，开发者可以参考本案例构建自己的镜像。

> 本案例提供制作镜像的方案和软件包均来源于开源社区，仅供参考。用户参考本案例制作的镜像，如需用于生产环境部署等商用行为，需自行保障镜像的可靠性、安全性等，MindSpore Transformers 不对其网络安全性负责，请在可信的环境中使用。

## 环境准备

在构建镜像前，需要准备主机环境，包括硬件、软件和网络。这一步确保构建顺利进行。

### 系统要求

- **硬件要求**：宿主机需安装 NPU 驱动和固件。参考文档：[昇腾社区-安装NPU驱动和固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/softwareinst/instg/instg_0005.html?Mode=DockerIns&InstallType=local&OS=Debian&Software=cannToolKit)。

- **软件要求**：Docker 版本：`26.1.4`或更高版本；

- **网络要求**：稳定的互联网连接；能访问华为云（下载 CANN、MindSpore 等）；网络慢时，构建时间会延长。

> 确保主机时间和时区正确，以避免下载问题。

## 工具安装

验证安装以下工具：

```bash
docker --version
```

若没有显示版本信息，请根据官方指导安装：

- [Docker 官方安装教程](https://docs.docker.com/engine/install/)

## 基础镜像选择

- 本案例 Dockerfile 使用 `ubuntu:24.04` 作为基础镜像
- 采用 **多阶段构建**：

  1. 第一阶段安装 Python
  2. 第二阶段安装 CANN
  3. 最终阶段安装 MindSpore 和 MindSpore Transformers，并整合结果

这样可以减少最终镜像大小，并提高构建效率。

DockerFile的内容可参考[社区 issue](https://gitee.com/mindspore/mindformers/issues/ICQ9JF)

并将其中的Dockerfile保存到本地。

## 镜像构建步骤

根据以下内容构建 **MindSpore Transformers** 镜像：

- 创建文件夹

  ```shell
  # 创建并进入存放 Dockerfile 的目录
  mkdir -p mindformers-Dockerfiles
  cd mindformers-Dockerfiles
  ```

- 将Dockerfile保存到以下目录：

  ```text
  mindformers-Dockerfiles/
  └── Dockerfile
  ```

- 运行 Docker 构建命令

  通用指令如下：

  ```bash
  # 开始构建镜像
  docker build -f Dockerfile \
    --build-arg PYTHON_VERSION="Python版本" \
    --build-arg CANN_TOOLKIT_URL="CANN toolkit 下载链接" \
    --build-arg CANN_KERNELS_URL="CANN kernels 下载链接" \
    --build-arg MS_WHL_URL="MindSpore whl包下载链接" \
    --build-arg MINDFORMERS_GIT_REF="MindSpore Transformers代码仓库分支名称" \
    -t "镜像名称:标签" .
  ```

  MindSpore Transformers 1.6.0 版本示例如下：

  ```bash
  # 开始构建镜像，这里的标签命名方式仅作参考，包含了版本信息便于管理
  docker build -f Dockerfile \
    --build-arg PYTHON_VERSION="3.11.4" \
    --build-arg CANN_TOOLKIT_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.2.RC1/Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run" \
    --build-arg CANN_KERNELS_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.2.RC1/Ascend-cann-kernels-910b_8.2.RC1_linux-aarch64.run" \
    --build-arg MS_WHL_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0/MindSpore/unified/aarch64/mindspore-2.7.0-cp311-cp311-linux_aarch64.whl" \
    --build-arg MINDFORMERS_GIT_REF="r1.6.0" \
    -t "mindformers:r1.6.0_ms2.7.0_cann8.2.RC1_py3.11" .
  ```

### 参数说明

| 参数 | 说明 | 获取地址 |
|------|------|----------|
| `PYTHON_VERSION` | Python 版本 | [Python 官网](https://www.python.org/downloads/) |
| `CANN_TOOLKIT_URL` | CANN toolkit包下载地址 | [昇腾社区下载页](https://www.hiascend.com/developer/download/community/result?module=cann) |
| `CANN_KERNELS_URL` | CANN kernels包下载地址 | [昇腾社区下载页](https://www.hiascend.com/developer/download/community/result?module=cann) |
| `MS_WHL_URL` | MindSpore wheel 包地址 | [MindSpore PyPI](https://repo.mindspore.cn/pypi/simple/mindspore/) |
| `MINDFORMERS_GIT_REF` | MindFormers 分支名称，会自动checkout到对应分支 | [MindFormers 仓库](https://gitee.com/mindspore/mindformers) |

> 构建过程可能需要 30 分钟左右，取决于网络速度和硬件性能。

## 验证构建

查看镜像是否成功：

```bash
# 查找特定镜像
docker images | grep mindformers
```

示例：

```text
REPOSITORY    TAG                                IMAGE ID       CREATED        SIZE
mindformers   r1.6.0_ms2.7.0_cann8.2.RC1_py3.11  67fa2e821694   19 hours ago   14GB
```

## 使用示例

### 启动开发容器

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
  --name 容器名称 \
  镜像名称 \
  /bin/bash
```

## 安全风险

在使用 Docker 容器运行 MindSpore Transformers 时，需要注意以下安全风险：

- **使用 root 用户运行**：容器默认以 root 用户身份运行，可能带来安全隐患。建议在生产环境中创建非特权用户来运行应用程序。

- **缺少 CPU 和内存资源限制**：未设置资源限制可能导致容器消耗过多系统资源，影响宿主机性能。建议使用 `--cpus` 和 `--memory` 参数限制资源使用。

- **设备使用 `rwm` 权限**：为 NPU 设备分配了读写和 mknod 权限，虽然功能运行需要，但在安全敏感环境中应谨慎评估权限范围。

> 在生产环境部署时，请根据实际安全要求调整容器配置，确保系统安全性。

## 参考资源

- [MindSpore 官网](https://www.mindspore.cn)
- [MindSpore Transformers 仓库](https://gitee.com/mindspore/mindformers)
- [Docker 官方文档](https://docs.docker.com)
- [Ascend 社区](https://www.hiascend.com/developer)
- [MindSpore 社区](https://gitee.com/mindspore/community)
- [相关 issue](https://gitee.com/mindspore/mindformers/issues/ICQ9JF)
