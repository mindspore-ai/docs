# MindSpore Armour Installation

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindarmour/docs/source_en/mindarmour_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>&nbsp;&nbsp;

## System Environment Information Confirmation

- The hardware platform should be Ascend, GPU or CPU.
- See our [MindSpore Installation Guide](https://www.mindspore.cn/install/en) to install MindSpore.  
    The versions of MindSpore Armour and MindSpore must be consistent.
- All other dependencies are included in [setup.py](https://gitee.com/mindspore/mindarmour/blob/master/setup.py).

## Version dependency

Due the dependency between MindSpore Armour and MindSpore, please follow the table below and install the corresponding MindSpore verision from [MindSpore download page](https://www.mindspore.cn/versions/en).

| MindSpore Armour Version | Branch                                                    | MindSpore Version |
| ------------------ | --------------------------------------------------------- | ----------------- |
| 2.0.0              | [r2.0](https://gitee.com/mindspore/mindarmour/tree/r2.0/) | >=1.7.0           |
| 1.9.0              | [r1.9](https://gitee.com/mindspore/mindarmour/tree/r1.9/) | >=1.7.0           |
| 1.8.0              | [r1.8](https://gitee.com/mindspore/mindarmour/tree/r1.8/) | >=1.7.0           |
| 1.7.0              | [r1.7](https://gitee.com/mindspore/mindarmour/tree/r1.7/) | 1.7.0             |

## Installation

You can install MindSpore Armour either by pip or by source code.

### Installation by pip

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindArmour/any/mindarmour-{version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [setup.py](https://gitee.com/mindspore/mindarmour/blob/master/setup.py)). In other cases, you need to manually install dependency items.
> - `{version}` denotes the version of MindSpore Armour. For example, when you are downloading MindSpore Armour 1.3.0, `{version}` should be 1.3.0.

### Installation by Source Code

1. Download source code from Gitee.

    ```bash
    git clone https://gitee.com/mindspore/mindarmour.git
    ```

2. Compile and install in MindSpore Armour directory.

    ```bash
    cd mindarmour
    python setup.py install
    ```

## Installation Verification

Successfully installed, if there is no error message such as `No module named 'mindarmour'` when execute the following command:

```bash
python -c 'import mindarmour'
```