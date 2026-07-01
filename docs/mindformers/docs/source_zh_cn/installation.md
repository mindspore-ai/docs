# 安装指南

本指南面向 **动态图（PyNative）** 场景，介绍 MindSpore Transformers 的环境依赖、安装方式与安装校验。动态图任务通过 `msrun ... run_mindformer.py --config xxx.yaml --mode 1` 启动（`--mode 1` 即 PyNative 模式），当前已支持 DeepSeek-V3等模型（完整模型清单见 [模型库](introduction/models.md)）。

安装总体分四步：**① 准备昇腾硬件并装好驱动/固件 → ② 按版本配套表安装 CANN 与 MindSpore → ③ 安装 MindSpore Transformers 本体 → ④ 安装 HyperParallel（动态图训练必需）**。其中第三步可在源码、pip、Docker 三种方式中任选其一。

> **动态图对 MindSpore 版本的额外要求**
>
> 动态图训练栈依赖 [HyperParallel](https://gitcode.com/mindspore/hyper-parallel/)，其要求 **MindSpore >= 2.10**（建议最新版本）。因此请选择 MindSpore >= 2.10，详见下方[版本配套关系](#版本配套关系)。

---

## 环境依赖

### 昇腾硬件

当前支持的硬件为 Atlas 800T A2、Atlas 800I A2、Atlas 900 A3 SuperPoD。

宿主机需提前安装 NPU 驱动与固件，参考[昇腾社区-安装 NPU 驱动和固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/softwareinst/instg/instg_0005.html)。

### Python 版本

| 项 | 说明                                                  |
|---|-----------------------------------------------------|
| **支持范围** | Python `>= 3.12`                                    |
| **推荐版本** | Python `3.12.4`，与官方 Docker 镜像内置的 `py3.12` 一致，经过完整验证 |

> "支持范围"表示能装上并运行，"推荐版本"表示官方在该版本上做过完整测试。新环境建议直接使用 3.12.4，避免低版本依赖兼容问题。

### 版本配套关系

动态图（r2.0.0 / 在研版本）需安装以下组件。由于依赖 HyperParallel，**MindSpore 必须 >= 2.10**（建议使用最新版本），CANN 与固件/驱动需与所选 MindSpore 版本对应。

| 组件 | 版本要求 | 获取方式 |
|---|---|---|
| MindSpore Transformers | 在研版本（master 分支） | [源码安装](#源码安装在研--master) |
| HyperParallel | 在研版本（与 MindSpore 配套） | [源码安装](#安装-hyperparallel动态图训练必需) |
| MindSpore | **>= 2.10**（建议最新版本） | [MindSpore 安装](https://www.mindspore.cn/install/) |

## 安装 MindSpore Transformers

完成 CANN 与 MindSpore 安装后，再安装 MindSpore Transformers 本体。下面三种方式按场景选择其一即可。

### 安装方式选型

| 方式 | 适用场景 | 取得的版本 |
|---|---|---|
| **源码安装** | 需要在研（master）最新特性、要改源码或调试 | master 分支当前代码 |
| **pip 安装** | 只想用已发布的稳定版本，环境已就绪 | PyPI 上的发布版本 |
| **Docker 安装** | 不想在宿主机上动 Python/CANN/MindSpore 环境，希望开箱即用 | 镜像内预置的发布版本 |

> 在研版本（master）目前**仅支持源码安装**；pip 安装对应的是已发布版本，请按[版本配套关系](#版本配套关系)选定版本号。

### 源码安装

```bash
# 在研版本（master 分支，最新特性）
git clone -b master https://atomgit.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

```bash
# r2.0 版本（r2.0 分支，对应 2.0.0 发布版本）
git clone -b r2.0 https://atomgit.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

`build.sh` 的实际行为（见仓库根目录 `build.sh`）：

1. 执行 `python setup.py bdist_wheel`，将源码打包为 `output/` 目录下的 `mindformers-*.whl`；
2. 为产出的 wheel 生成 `.sha256` 校验文件；
3. **默认从清华镜像** `https://pypi.tuna.tsinghua.edu.cn/simple` 执行 `pip install mindformers*whl` 完成安装。

> **自定义安装源**
>
> `build.sh` 内置清华 pip 镜像。如需改用其它源，可先 `python setup.py bdist_wheel -d output` 产出 wheel，再用自定义源安装：
>
> ```bash
> python setup.py bdist_wheel -d output
> pip install output/mindformers-*.whl -i <你的 pip 源>
> ```

编译前置依赖：源码安装会触发本地打包，请确保已安装 `wheel`、`setuptools`，且 `requirements.txt` 中依赖可正常拉取。若 `bash build.sh` 失败，优先检查：网络能否访问 pip 源、Python 版本是否在受支持范围、是否已成功安装匹配版本的 MindSpore。

### pip 安装（已发布版本）

已发布版本可通过 pip 直接安装。版本号请参考上方[版本配套关系](#版本配套关系)表，与已安装的 MindSpore 整行对应：

```bash
# 安装与配套表对应的指定版本（推荐，避免错配）
pip install mindformers==2.0.0

# 或安装 PyPI 上的最新发布版本
pip install mindformers
```

> 指定版本号能确保与已装的 CANN/MindSpore 配套；不带版本号时会安装最新发布版，需自行确认是否与本机 MindSpore 匹配。
>
> `2.0.0` 当前暂未发布到 PyPI，pip 安装尚不可用。现阶段请使用上方「源码安装」方式获取 master 分支代码。

---

## 安装 HyperParallel（动态图训练必需）

动态图（PyNative）训练栈依赖 [HyperParallel](https://gitcode.com/mindspore/hyper-parallel/) —— 昇腾超节点亲和的分布式并行加速库。它为动态图提供 `DTensor` / `DeviceMesh`、FSDP/HSDP、流水线并行（PP，1F1B/VPP）调度，以及 MindSpore 动态图反向兼容层等核心能力。

`mindformers/pynative/` 下的训练器、优化器、并行切分与融合算子均直接 `import hyper_parallel`，因此**未安装将无法运行任何动态图任务**——不仅限于并行场景，单卡训练同样需要。

> **版本要求**
>
> HyperParallel 要求 **MindSpore >= 2.10**（建议使用最新版本）。请确保本机 MindSpore 满足该要求，详见仓库 README 的安装说明。

以下演示通过源码安装 HyperParallel：

```bash
git clone https://gitcode.com/mindspore/hyper-parallel.git
cd hyper-parallel
python setup.py bdist_wheel
pip install dist/hyper_parallel-*-py3-none-any.whl
```

安装后校验是否可正常导入：

```bash
python -c "from hyper_parallel import DTensor, DeviceMesh, PipelineStage; print('hyper_parallel OK')"
```

> **Docker 用户**：使用官方预构建镜像或自行构建镜像时，请确认镜像内是否已预置 HyperParallel；若未预置，进入容器后按上述源码方式安装。

---

## Docker 安装

若不希望在宿主机直接配置 Python/CANN/MindSpore 环境，可使用 Docker 镜像运行 MindSpore Transformers。**推荐优先使用官方预构建镜像**；若需自定义版本组合，再走自行构建路径。

### 环境与工具准备

- **硬件**：宿主机需安装 NPU 驱动与固件，参考[昇腾社区-安装 NPU 驱动和固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/softwareinst/instg/instg_0005.html)。
- **软件**：Docker 版本 `26.1.4` 或更高。可通过 `docker --version` 校验；若未安装，参考 [Docker 官方安装教程](https://docs.docker.com/engine/install/)。
- **网络**：拉取/构建镜像需稳定的互联网连接，并能访问华为云。请确保主机时间和时区正确。

### 方式一：使用官方预构建镜像（推荐）

MindSpore Transformers Ascend 镜像托管在华为云 SWR 镜像仓库，开箱即用、无需本地构建。

> 镜像仓库中可能尚未推出对应版本的官方预构建镜像，请以 [docker/OVERVIEW_CN.md](https://gitcode.com/mindspore/mindformers/blob/master/docker/OVERVIEW_CN.md) 中的实际可用 tag 为准。若暂无所需版本，可使用下文的「方式二：自行构建镜像」。

**镜像仓库地址：**

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/mindformers
```

**Tag 规范**（系统架构由 Docker Manifest 自动识别，无需在 tag 中指定）：

```text
<MindSpore Transformers 版本号>-<硬件信息（芯片）>-<操作系统>-<Python 版本>
```

| 字段 | 示例值 | 说明 |
|---|---|---|
| 版本号 | `2.0.0` | MindSpore Transformers 发布版本 |
| 硬件信息（芯片） | `<芯片架构>` | 昇腾芯片型号标识，具体取值见 [docker/OVERVIEW_CN.md](https://gitcode.com/mindspore/mindformers/blob/master/docker/OVERVIEW_CN.md) |
| 操作系统 | `ubuntu22.04` / `openeuler24.03` | 基础镜像操作系统发行版 |
| Python 版本 | `py3.12` | 镜像内置 Python 大版本 |

**拉取镜像**（以 `2.0.0-<芯片架构>-ubuntu22.04-py3.12` 为例）：

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindformers:2.0.0-<芯片架构>-ubuntu22.04-py3.12
```

> `<芯片架构>` 替换为实际芯片型号，完整取值列表见 [docker/OVERVIEW_CN.md](https://gitcode.com/mindspore/mindformers/blob/master/docker/OVERVIEW_CN.md)。可用 `docker manifest inspect <镜像>` 查看镜像支持的系统架构（ARM64 / x86_64）。

### 方式二：自行构建镜像

需要自定义 CANN/MindSpore/MindSpore Transformers 版本组合时，使用仓库内官方 Dockerfile（`docker/Dockerfile`）构建。该 Dockerfile 基于 CANN 官方基础镜像 `quay.io/ascend/cann`，依次安装依赖、MindSpore 与 MindSpore Transformers。

```bash
docker build \
  --build-arg CANN_VERSION=9.1.0 \
  --build-arg CHIP_ARCH=<芯片架构> \
  --build-arg OS_SYSTEM=ubuntu22.04 \
  --build-arg PY_VERSION=py3.12 \
  --build-arg MINDSPORE_VERSION=2.10.0 \
  --build-arg MINDFORMERS_VERSION=2.0.0 \
  --build-arg PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
  -t mindformers:2.0.0-<芯片架构>-ubuntu22.04-py3.12 \
  -f docker/Dockerfile .
```

各 `build-arg` 的含义（与仓库 `docker/Dockerfile` 一致）：

| 参数 | 必填 | 说明 | 示例值 |
|---|---|---|---|
| `CANN_VERSION` | 是 | 昇腾 CANN 工具包版本（决定基础镜像 tag） | `9.1.0` |
| `CHIP_ARCH` | 是 | 昇腾芯片架构标识，具体取值见 [docker/OVERVIEW_CN.md](https://gitcode.com/mindspore/mindformers/blob/master/docker/OVERVIEW_CN.md) | `<芯片架构>` |
| `OS_SYSTEM` | 是 | 基础镜像操作系统及版本 | `ubuntu22.04` / `openeuler24.03` |
| `PY_VERSION` | 是 | 基础镜像内置 Python 版本 | `py3.12` |
| `MINDSPORE_VERSION` | 是 | MindSpore 版本号（按配套表选取） | `2.10.0` |
| `MINDFORMERS_VERSION` | 是 | MindSpore Transformers 版本号 | `2.0.0` |
| `PIP_INDEX_URL` | 否 | pip 安装源地址，默认华为云源 | `https://mirrors.huaweicloud.com/repository/pypi/simple` |

> 上述各版本号必须满足[版本配套关系](#版本配套关系)。更多细节见仓库 `docker/OVERVIEW_CN.md`。

### 启动容器

拉取或构建完成后，挂载 NPU 设备启动容器：

```bash
docker run -itd \
  --ipc=host \
  --network=host \
  --device=/dev/davinci0:rwm \
  --device=/dev/davinci_manager:rwm \
  --device=/dev/devmm_svm:rwm \
  --device=/dev/hisi_hdc:rwm \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  --name 容器名称 \
  <镜像名称:标签> \
  /bin/bash
```

> **安全提示**
>
> - 上例仅挂载了 `davinci0` 一张卡，按需增加 `--device=/dev/davinciN:rwm` 挂载更多 NPU。
> - 容器默认以 root 运行，生产环境建议创建非特权用户运行。
> - 为保证 NPU 功能可能需要 `--privileged`，会扩大容器权限，建议仅在可信环境使用，并结合 `--cpus`、`--memory` 限制资源。

---

## 验证是否成功安装

安装完成后，执行以下命令进行环境自检：

```bash
python -c "import mindformers as mf;mf.run_check()"
```

`run_check` 会搜集环境信息（MindFormers / MindSpore / CANN / 驱动版本），并据此给出三类结果。

### 结果一：完全通过

```text
- INFO - All checks passed, used **** seconds, the environment is correctly set up!
```

各组件版本配套、检查全部通过，可以开始训练。

### 结果二：版本不配套但检查通过（警告）

```text
- WARNING - The installed software are unmatched but all checks passed in **** seconds
- INFO - It's recommended to install mindformers==x.y.z mindspore==x.y.z cann-toolkit==x.y.z driver==x.y.z
```

功能可用，但当前 MindFormers/MindSpore/CANN/驱动组合不在推荐配套内。建议按提示中的推荐版本与[版本配套关系](#版本配套关系)整行重装，以规避潜在精度/性能/兼容问题。

### 结果三：检查失败

```text
- ERROR - The run check failed in **** seconds, It's recommended to install mindformers==x.y.z mindspore==x.y.z cann-toolkit==x.y.z driver==x.y.z
```

或

```text
- ERROR - The run check failed in **** seconds, please check the above info for more details
```

常见排查方向：

| 现象 | 排查方向 |
|---|---|
| MindSpore 自身报错（`MindSpore failed!`） | 参考 [MindSpore 安装](https://www.mindspore.cn/install/) 重装匹配版本的 MindSpore |
| 提示 `ASCEND_HOME_PATH` 未设置 / 找不到 CANN 信息 | 检查 CANN 是否安装、是否 `source` 了环境变量 `set_env.sh` |
| 找不到驱动 version.info | 检查 NPU 驱动/固件是否安装、`npu-smi info` 是否正常 |
| 版本不匹配 | 按推荐版本与[版本配套关系](#版本配套关系)整行重装 |
| 运行动态图任务报 `ModuleNotFoundError: hyper_parallel` | 未安装 HyperParallel，见[安装 HyperParallel](#安装-hyperparallel动态图训练必需) |

### 跑通一步动态图自检

`run_check` 通过后，建议再用一个最小动态图任务确认端到端可拉起。动态图任务以 `--mode 1`（PyNative）启动，单机多卡通过 `msrun` 拉起，例如：

```bash
msrun --worker_num=8 --local_worker_num=8 --master_port=8118 \
  run_mindformer.py --config <你的配置>.yaml --mode 1
```

完整的任务拉起步骤与最小可运行示例，见 [快速开始](quick_start/quick_start.md)。

---

## 相关文档

- 安装完成后第一站：[快速开始](quick_start/quick_start.md) —— 拉起第一个动态图任务。
- 训练总流程：[训练指南](guide/training.md)。
- 支持的模型清单：[模型库](introduction/models.md)。
- 数据与配置：[数据集](./feature/dataset.md)、[配置文件说明](./feature/configuration.md)。
