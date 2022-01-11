# MindSpore Hub Installation

- [MindSpore Hub Installation](#mindspore-hub-installation)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Installation Methods](#installation-methods)
        - [Installation by pip](#installation-by-pip)
        - [Installation by Source Code](#installation-by-source-code)
    - [Installation Verification](#installation-verification)

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/hub/docs/source_en/hub_installation.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## System Environment Information Confirmation

- The hardware platform supports Ascend, GPU and CPU.
- Confirm that [Python](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) 3.7.5 is installed.
- The versions of MindSpore Hub and MindSpore must be consistent.
- MindSpore Hub supports only Linux distro with x86 architecture 64-bit or ARM architecture 64-bit.
- When the network is connected, dependency items in the `setup.py` file are automatically downloaded during .whl package installation. In other cases, you need to manually install dependency items.

## Installation Methods

You can install MindSpore Hub either by pip or by source code.

### Installation by pip

Install MindSpore Hub using `pip` command. `hub` depends on the MindSpore version used in current environment.

Download and install the MindSpore Hub whl package in [Release List](https://www.mindspore.cn/versions/en).

```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/Hub/any/mindspore_hub-{version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - `{version}` denotes the version of MindSpore Hub. For example, when you are downloading MindSpore Hub 1.3.0, `{version}` should be 1.3.0.

### Installation by Source Code

1. Download source code from Gitee.

   ```bash
   git clone https://gitee.com/mindspore/hub.git
   ```

2. Compile and install in MindSpore Hub directory.

   ```bash
   cd hub
   python setup.py install
   ```

## Installation Verification

Run the following command in a network-enabled environment to verify the installation.

```python
import mindspore_hub as mshub

model = mshub.load("mindspore/cpu/1.0/lenet_v1_mnist", num_class = 10)
```

If it prompts the following information, the installation is successful:

```text
Downloading data from url https://gitee.com/mindspore/hub/raw/r1.6/mshub_res/assets/mindspore/cpu/1.0/lenet_v1_mnist.md

Download finished!
File size = 0.00 Mb
Checking /home/ma-user/.mscache/mindspore/cpu/1.0/lenet_v1_mnist.md...Passed!
```