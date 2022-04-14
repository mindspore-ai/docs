# MindArmour Installation

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindarmour/docs/source_en/mindarmour_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>&nbsp;&nbsp;

## System Environment Information Confirmation

- The hardware platform should be Ascend, GPU or CPU.
- See our [MindSpore Installation Guide](https://www.mindspore.cn/install/en) to install MindSpore.  
    The versions of MindArmour and MindSpore must be consistent.
- All other dependencies are included in [setup.py](https://gitee.com/mindspore/mindarmour/blob/r1.7/setup.py).

## Installation

You can install MindArmour either by pip or by source code.

### Installation by pip

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindArmour/any/mindarmour-{version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [setup.py](https://gitee.com/mindspore/mindarmour/blob/r1.7/setup.py)). In other cases, you need to manually install dependency items.
> - `{version}` denotes the version of MindArmour. For example, when you are downloading MindArmour 1.3.0, `{version}` should be 1.3.0.

### Installation by Source Code

1. Download source code from Gitee.

    ```bash
    git clone https://gitee.com/mindspore/mindarmour.git
    ```

2. Compile and install in MindArmour directory.

    ```bash
    cd mindarmour
    python setup.py install
    ```

## Installation Verification

Successfully installed, if there is no error message such as `No module named 'mindarmour'` when execute the following command:

```bash
python -c 'import mindarmour'
```