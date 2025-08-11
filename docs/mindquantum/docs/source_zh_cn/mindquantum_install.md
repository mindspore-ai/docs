# 安装 MindSpore Quantum

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindquantum/docs/source_zh_cn/mindquantum_install.md)

## 确认系统环境信息

在开始安装前，请确认您的软硬件环境满足以下要求：

- **Python 版本**：请确保您的 Python 环境是 **Python 3.9, 3.10 或 3.11** 版本。您可以使用 `python --version` 命令查看。

### 版本配套关系

安装 MindQuantum 前，请先根据下表配套关系完成对应 MindSpore 版本的安装。如果 MindSpore 版本不匹配，可能会导致安装或运行时出现兼容性问题。

| MindQuantum 版本 | MindSpore 版本要求                     |
| :--------------- | :------------------------------------- |
| 0.11.0           | MindSpore >= 2.2.0 (推荐使用 2.7.0rc1) |

- 请参考[MindSpore 安装指南](https://www.mindspore.cn/install)安装推荐的 MindSpore 版本。

## 安装方式

可以采用 pip 安装或者源码编译安装两种方式。

### pip 安装

#### 1. 安装最新稳定版

您可以从 PyPI 官方源安装 MindQuantum 的最新稳定版本。

```bash
pip install mindquantum
```

#### 2. 安装指定版本

您也可以从 MindSpore 社区下载指定版本的`whl`包进行安装。

请根据您的系统环境，在下表中选择合适的`mindquantum-0.11.0`版本进行安装。

<table class="colwidths-auto">
  <thead>
    <tr>
      <th style="text-align: center">操作系统</th>
      <th style="text-align: center">硬件平台</th>
      <th style="text-align: center">Python 版本</th>
      <th style="text-align: center">MindQuantum whl包</th>
      <th style="text-align: center">SHA-256校验文件</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6" style="text-align: center">Linux</td>
      <td rowspan="3" style="text-align: center">x86_64 (CPU/GPU)</td>
      <td style="text-align: center">3.9</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/gpu/x86_64/cuda-11.1/mindquantum-0.11.0-cp39-cp39-linux_x86_64.whl">mindquantum-0.11.0-cp39-cp39-linux_x86_64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/gpu/x86_64/cuda-11.1/mindquantum-0.11.0-cp39-cp39-linux_x86_64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td style="text-align: center">3.10</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/gpu/x86_64/cuda-11.1/mindquantum-0.11.0-cp310-cp310-linux_x86_64.whl">mindquantum-0.11.0-cp310-cp310-linux_x86_64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/gpu/x86_64/cuda-11.1/mindquantum-0.11.0-cp310-cp310-linux_x86_64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td style="text-align: center">3.11</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/gpu/x86_64/cuda-11.1/mindquantum-0.11.0-cp311-cp311-linux_x86_64.whl">mindquantum-0.11.0-cp311-cp311-linux_x86_64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/gpu/x86_64/cuda-11.1/mindquantum-0.11.0-cp311-cp311-linux_x86_64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center">aarch64 (CPU)</td>
      <td style="text-align: center">3.9</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp39-cp39-linux_aarch64.whl">mindquantum-0.11.0-cp39-cp39-linux_aarch64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp39-cp39-linux_aarch64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td style="text-align: center">3.10</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp310-cp310-linux_aarch64.whl">mindquantum-0.11.0-cp310-cp310-linux_aarch64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp310-cp310-linux_aarch64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td style="text-align: center">3.11</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp311-cp311-linux_aarch64.whl">mindquantum-0.11.0-cp311-cp311-linux_aarch64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp311-cp311-linux_aarch64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center">Windows</td>
      <td rowspan="3" style="text-align: center">x86_64 (CPU)</td>
      <td style="text-align: center">3.9</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/x86_64/mindquantum-0.11.0-cp39-cp39-win_amd64.whl">mindquantum-0.11.0-cp39-cp39-win_amd64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/x86_64/mindquantum-0.11.0-cp39-cp39-win_amd64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td style="text-align: center">3.10</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/x86_64/mindquantum-0.11.0-cp310-cp310-win_amd64.whl">mindquantum-0.11.0-cp310-cp310-win_amd64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/x86_64/mindquantum-0.11.0-cp310-cp310-win_amd64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td style="text-align: center">3.11</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/x86_64/mindquantum-0.11.0-cp311-cp311-win_amd64.whl">mindquantum-0.11.0-cp311-cp311-win_amd64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/x86_64/mindquantum-0.11.0-cp311-cp311-win_amd64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td rowspan="6" style="text-align: center">macOS</td>
      <td rowspan="3" style="text-align: center">x86_64 (CPU)</td>
      <td style="text-align: center">3.9</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/x86_64/mindquantum-0.11.0-cp39-cp39-macosx_10_15_x86_64.whl">mindquantum-0.11.0-cp39-cp39-macosx_10_15_x86_64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/x86_64/mindquantum-0.11.0-cp39-cp39-macosx_10_15_x86_64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td style="text-align: center">3.10</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/x86_64/mindquantum-0.11.0-cp310-cp310-macosx_10_15_x86_64.whl">mindquantum-0.11.0-cp310-cp310-macosx_10_15_x86_64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/x86_64/mindquantum-0.11.0-cp310-cp310-macosx_10_15_x86_64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td style="text-align: center">3.11</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/x86_64/mindquantum-0.11.0-cp311-cp311-macosx_10_15_x86_64.whl">mindquantum-0.11.0-cp311-cp311-macosx_10_15_x86_64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/x86_64/mindquantum-0.11.0-cp311-cp311-macosx_10_15_x86_64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center">arm64 (CPU)</td>
      <td style="text-align: center">3.9</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp39-cp39-macosx_11_0_arm64.whl">mindquantum-0.11.0-cp39-cp39-macosx_11_0_arm64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp39-cp39-macosx_11_0_arm64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td style="text-align: center">3.10</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp310-cp310-macosx_11_0_arm64.whl">mindquantum-0.11.0-cp310-cp310-macosx_11_0_arm64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp310-cp310-macosx_11_0_arm64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td style="text-align: center">3.11</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp311-cp311-macosx_11_0_arm64.whl">mindquantum-0.11.0-cp311-cp311-macosx_11_0_arm64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp311-cp311-macosx_11_0_arm64.whl.sha256">sha256</a></td>
    </tr>
  </tbody>
</table>

> - **关于 Linux x86_64 包**：该平台提供的是 CPU/GPU 统一版本。安装后，如果您的设备有可用的 NVIDIA GPU 和 CUDA 11.1 以上的环境，则可以使用 GPU 相关功能。

**安装命令**

您可以直接使用 pip 通过 URL 进行安装，也可以先下载 whl 包和 sha256 校验文件到本地再进行安装。

```bash
# 方式一：使用URL直接安装 (以Linux x86_64，Python 3.9为例)
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/gpu/x86_64/cuda-11.1/mindquantum-0.11.0-cp39-cp39-linux_x86_64.whl
```

**本地下载后安装与校验**

1. **下载文件**：从上表下载 `whl` 安装包和对应的 `.whl.sha256` 校验文件。

2. **校验哈希值**（可选）

   首先，计算您下载的 `whl` 文件的哈希值（请将命令中的 `your_downloaded_file.whl` 替换为您的实际文件名）：

   ```bash
   sha256sum your_downloaded_file.whl
   ```

   然后，查看 `.sha256` 文件中记录的官方哈希值：

   ```bash
   cat your_downloaded_file.whl.sha256
   ```

   **手动比对**两个命令输出的哈希字符串。如果完全一致，则说明文件正确无误。

   > **说明：** `.whl.sha256` 文件中记录的文件名可能与您下载的 `whl` 包的实际文件名不一致。这是因为构建系统有时会为了确保更广泛的兼容性而对文件名进行标准化处理（例如，将 `manylinux_2_27_x86_64.whl` 重命名为 `linux_x86_64.whl`）。请以哈希值为准。

3. **安装软件包**：确认校验无误后，使用 `pip` 安装：

   ```bash
   # 请将 your_downloaded_file.whl 替换为您的实际文件名
   pip install your_downloaded_file.whl
   ```

> - 如果您的网络环境不佳，可以尝试使用华为云镜像源加速下载：`pip install -i https://repo.huaweicloud.com/repository/pypi/simple mindquantum`。

### 源码安装

1. 从代码仓下载源码

   ```bash
   cd ~
   git clone https://gitee.com/mindspore/mindquantum.git
   ```

2. 编译安装 MindSpore Quantum

   ```bash
   cd ~/mindquantum
   python setup.py install --user
   ```

## 验证是否成功安装

执行如下命令，如果没有报错`No module named 'mindquantum'`，则说明安装成功。

```bash
python -c 'import mindquantum'
```

## Docker 安装

通过 Docker 也可以在 Mac 系统或者 Windows 系统中使用 MindQuantum。具体参考[Docker 安装指南](https://gitee.com/mindspore/mindquantum/blob/master/install_with_docker.md)。
