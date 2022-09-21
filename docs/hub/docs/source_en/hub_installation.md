# MindSpore Hub Installation

- [MindSpore Hub Installation](#mindspore-hub-installation)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Installation Methods](#installation-methods)
        - [Installation by pip](#installation-by-pip)
        - [Installation by Source Code](#installation-by-source-code)
    - [Installation Verification](#installation-verification)

<a href="https://gitee.com/mindspore/docs/blob/r1.9/docs/hub/docs/source_en/hub_installation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png"></a>

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
   git clone https://gitee.com/mindspore/hub.git -b r1.9
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

model = mshub.load("mindspore/1.6/lenet_mnist", num_class=10)
```

If it prompts the following information, the installation is successful:

```text
Downloading data from url https://gitee.com/mindspore/hub/raw/r1.9/mshub_res/assets/mindspore/1.6/lenet_mnist.md

Download finished!
File size = 0.00 Mb
Checking /home/ma-user/.mscache/mindspore/1.6/lenet_mnist.md...Passed!
```

## FAQ

<font size=3>**Q: What to do when `SSL: CERTIFICATE_VERIFY_FAILED` occurs?**</font>

A: Due to your network environment, for example, if you use a proxy to connect to the Internet, SSL verification failure may occur on Python because of incorrect certificate configuration. In this case, you can use either of the following methods to solve this problem:

Configure the SSL certificate **(recommended)**.
Before import mindspore_hub, please add the codes (the fastest method).

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import mindspore_hub as mshub
model = mshub.load("mindspore/1.6/lenet_mnist", num_classes=10)
```

<font size=3>**Q: What to do when `No module named src.*` occurs**?</font>

A: When you use mindspore_hub.load to load differenet models in the same process, because the model file path needs to be inserted into sys.path. Test results show that Python only looks for src.* in the first inserted path. It's no use to delete the first inserted path. To solve the problem, you can copy all model files to the working directory. The code is as follows:

```python
# mindspore_hub_install_path/load.py
def _copy_all_file_to_target_path(path, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    path = os.path.realpath(path)
    target_path = os.path.realpath(target_path)
    for p in os.listdir(path):
        copy_path = os.path.join(path, p)
        target_dir = os.path.join(target_path, p)
        _delete_if_exist(target_dir)
        if os.path.isdir(copy_path):
            _copy_all_file_to_target_path(copy_path, target_dir)
        else:
            shutil.copy(copy_path, target_dir)

def _get_network_from_cache(name, path, *args, **kwargs):
    _copy_all_file_to_target_path(path, os.getcwd())
    config_path = os.path.join(os.getcwd(), HUB_CONFIG_FILE)
    if not os.path.exists(config_path):
        raise ValueError('{} not exists.'.format(config_path))
    ......
```

**Note**: Some files of the previous model may be replaced when the next model is loaded. However，necessary model files must exist during model training. Therefore, you must finish training the previous model before the next model loads.
