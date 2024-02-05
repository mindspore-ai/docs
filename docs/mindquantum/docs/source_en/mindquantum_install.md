# MindSpore Quantum Installation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindquantum/docs/source_en/mindquantum_install.md)

## Confirming System Environment Information

- Refer to [MindSpore Installation Guide](https://www.mindspore.cn/install/en), install MindSpore, version 1.4.0 or later is required.
- See [setup.py](https://gitee.com/mindspore/mindquantum/blob/r0.9/setup.py) for the remaining dependencies.

## Installation Methods

You can install MindInsight either by pip or by source code.

### Installing by pip

```bash
pip install mindquantum
```

> - Refer to [MindSpore](https://www.mindspore.cn/versions) to find different version of packages。

### Installing by Source Code

1. Download Source Code from Gitee

    ```bash
    cd ~
    git clone https://gitee.com/mindspore/mindquantum.git
    ```

2. Compiling MindSpore Quantum

    ```bash
    cd ~/mindquantum
    python setup.py install --user
    ```

## Verifying Successful Installation

Successfully installed, if there is no error message such as `No module named 'mindquantum'` when execute the following command:

```bash
python -c 'import mindquantum'
```

## Installing with Docker

Mac or Windows users can install MindSpore Quantum through Docker. Please refer to [Docker installation guide](https://gitee.com/mindspore/mindquantum/blob/r0.9/install_with_docker_en.md#).
