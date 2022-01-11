# MindInsight Installation

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindinsight/docs/source_en/mindinsight_install.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## System Environment Information Confirmation

- The hardware platform supports Ascend, GPU and CPU.
- Ensure that Python 3.7.5 or 3.9.0 is installed. If not installed, download and install Python from:
    - Python 3.7.5 (64-bit): [Python official website](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) or [HUAWEI CLOUD](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz).
    - Python 3.9.0 (64-bit): [Python official website](https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz) or [HUAWEI CLOUD](https://mirrors.huaweicloud.com/python/3.9.0/Python-3.9.0.tgz).
- The versions of MindInsight and MindSpore must be consistent.
- If you use source code to compile and install, the following dependencies also need to be installed:
    - Confirm that [node.js](https://nodejs.org/en/download/) 10.19.0 or later is installed.
    - Confirm that [wheel](https://pypi.org/project/wheel/) 0.32.0 or later is installed.
- All other dependencies are included in [requirements.txt](https://gitee.com/mindspore/mindinsight/blob/r1.6/requirements.txt).

## Installation Methods

You can install MindInsight either by pip or by source code or by Docker.

### Installation by pip

Install from PyPI:

```bash
pip install mindinsight
```

Install with customized version:

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindInsight/any/mindinsight-{version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [requirements.txt](https://gitee.com/mindspore/mindinsight/blob/r1.6/requirements.txt)). In other cases, you need to manually install dependency items.
> - `{version}` denotes the version of MindInsight. For example, when you are downloading MindInsight 1.3.0, `{version}` should be 1.3.0.
> - MindInsight supports only Linux distro with x86 architecture 64-bit or ARM architecture 64-bit.

### Installation by Source Code

#### Downloading Source Code from Gitee

```bash
git clone https://gitee.com/mindspore/mindinsight.git -b r1.6
```

#### Compiling MindInsight

You can choose any of the following installation methods:

1. Run the following command in the root directory of the source code:

    ```bash
    cd mindinsight
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    python setup.py install
    ```

2. Build the `whl` package for installation.

    Enter the root directory of the source code, first execute the MindInsight compilation script in the `build` directory, and then execute the command to install the `whl` package generated in the `output` directory.

    ```bash
    cd mindinsight
    bash build/build.sh
    pip install output/mindinsight-{version}-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

### Installation by Docker

The MindSpore image contains the MindInsight function. For details, see the [Installation Guide](https://www.mindspore.cn/install/en) on the official website.

## Installation Verification

Execute the following command:

```bash
mindinsight start
```

If it prompts the following information, the installation is successful:

```bash
Web address: http://127.0.0.1:8080
service start state: success
```
