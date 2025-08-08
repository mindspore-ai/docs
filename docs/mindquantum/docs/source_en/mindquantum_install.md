# MindSpore Quantum Installation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.7.0rc1/docs/mindquantum/docs/source_en/mindquantum_install.md)

## Confirming System Environment Information

Before you begin, please confirm that your software and hardware environment meets the following requirements:

- **Python Version**: Please ensure your Python environment is version **3.9, 3.10, or 3.11**. You can check your version with the `python --version` command.

### Version Compatibility

Before installing MindQuantum, you must first install the corresponding version of MindSpore based on the compatibility table below. A version mismatch may cause installation or runtime issues.

| MindQuantum Version | Required MindSpore Version                   |
| :------------------ | :------------------------------------------- |
| 0.11.0              | MindSpore >= 2.2.0 (2.7.0rc1 is recommended) |

- Please refer to the [MindSpore Installation Guide](https://www.mindspore.cn/install/en) to install the recommended version of MindSpore.

## Installation Methods

You can install MindQuantum either by pip or from source code.

### Installing by pip

#### 1. Installing the Latest Stable Version

You can install the latest stable version of MindQuantum from the official PyPI source.

```bash
pip install mindquantum
```

#### 2. Installing a Specific Version

You can also download a specific version's `whl` package from the MindSpore community and install it manually.

Please select the appropriate `mindquantum-0.11.0` version from the table below based on your system environment.

<table class="colwidths-auto">
  <thead>
    <tr>
      <th style="text-align: center">Operating System</th>
      <th style="text-align: center">Hardware Platform</th>
      <th style="text-align: center">Python Version</th>
      <th style="text-align: center">MindQuantum whl Package</th>
      <th style="text-align: center">SHA-256 Checksum File</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6" style="text-align: center">Linux</td>
      <td rowspan="3" style="text-align: center">x86_64 (CPU/GPU)</td>
      <td style="text-align: center">3.9</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/gpu/x86_64/cuda-11.1/mindquantum-0.11.0-cp39-cp39-manylinux_2_27_x86_64.whl">mindquantum-0.11.0-cp39-cp39-manylinux_2_27_x86_64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/gpu/x86_64/cuda-11.1/mindquantum-0.11.0-cp39-cp39-manylinux_2_27_x86_64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td style="text-align: center">3.10</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/gpu/x86_64/cuda-11.1/mindquantum-0.11.0-cp310-cp310-manylinux_2_27_x86_64.whl">mindquantum-0.11.0-cp310-cp310-manylinux_2_27_x86_64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/gpu/x86_64/cuda-11.1/mindquantum-0.11.0-cp310-cp310-manylinux_2_27_x86_64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td style="text-align: center">3.11</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/gpu/x86_64/cuda-11.1/mindquantum-0.11.0-cp311-cp311-manylinux_2_27_x86_64.whl">mindquantum-0.11.0-cp311-cp311-manylinux_2_27_x86_64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/gpu/x86_64/cuda-11.1/mindquantum-0.11.0-cp311-cp311-manylinux_2_27_x86_64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center">aarch64 (CPU)</td>
      <td style="text-align: center">3.9</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp39-cp39-manylinux_2_26_aarch64.manylinux_2_27_aarch64.whl">mindquantum-0.11.0-cp39-cp39-manylinux_2_26_aarch64.manylinux_2_27_aarch64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp39-cp39-manylinux_2_26_aarch64.manylinux_2_27_aarch64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td style="text-align: center">3.10</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp310-cp310-manylinux_2_26_aarch64.manylinux_2_27_aarch64.whl">mindquantum-0.11.0-cp310-cp310-manylinux_2_26_aarch64.manylinux_2_27_aarch64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp310-cp310-manylinux_2_26_aarch64.manylinux_2_27_aarch64.whl.sha256">sha256</a></td>
    </tr>
    <tr>
      <td style="text-align: center">3.11</td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp311-cp311-manylinux_2_26_aarch64.manylinux_2_27_aarch64.whl">mindquantum-0.11.0-cp311-cp311-manylinux_2_26_aarch64.manylinux_2_27_aarch64.whl</a></td>
      <td style="text-align: center"><a href="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/aarch64/mindquantum-0.11.0-cp311-cp311-manylinux_2_26_aarch64.manylinux_2_27_aarch64.whl.sha256">sha256</a></td>
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

> - **About the Linux x86_64 package**: This platform provides a unified CPU/GPU version. After installation, if your device has a compatible NVIDIA GPU and a CUDA 11.1 or higher environment, GPU-related features will be available.

**Installation Commands**

You can install directly via URL using pip, or download the whl package and its sha256 checksum file to install locally.

```bash
# Method 1: Install directly from URL (e.g., for Linux x86_64, Python 3.9)
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.0rc1/MindQuantum/gpu/x86_64/cuda-11.1/mindquantum-0.11.0-cp39-cp39-manylinux_2_27_x86_64.whl

# Method 2: Install after local download
# 1. Download the whl package and its corresponding .sha256 file from the table above
# 2. Verify integrity (optional, but recommended)
sha256sum -c mindquantum-0.11.0-cp39-cp39-manylinux_2_27_x86_64.whl.sha256
# 3. Install using pip
pip install mindquantum-0.11.0-cp39-cp39-manylinux_2_27_x86_64.whl
```

> - If you have a poor network connection, you can try using the Huawei Cloud mirror to speed up the download: `pip install -i https://repo.huaweicloud.com/repository/pypi/simple mindquantum`.

### Installing by Source Code

1. Download Source Code from Gitee

   ```bash
   cd ~
   git clone -b r0.11 https://gitee.com/mindspore/mindquantum.git
   ```

2. Compiling MindSpore Quantum

   ```bash
   cd ~/mindquantum
   python setup.py install --user
   ```

## Verifying Successful Installation

If no error message such as `No module named 'mindquantum'` is reported when you run the following command, the installation is successful:

```bash
python -c 'import mindquantum'
```

## Installing with Docker

Mac or Windows users can install MindSpore Quantum through Docker. Please refer to the [Docker installation guide](https://gitee.com/mindspore/mindquantum/blob/r0.11/install_with_docker_en.md).
