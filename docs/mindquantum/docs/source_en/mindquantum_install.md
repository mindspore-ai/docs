# MindSpore Quantum Installation

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindquantum/docs/source_en/mindquantum_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Confirming System Environment Information

- Refer to [MindSpore Installation Guide](https://www.mindspore.cn/install/en), install MindSpore, version 1.4.0 or later is required.
- See [setup.py](https://gitee.com/mindspore/mindquantum/blob/master/setup.py) for the remaining dependencies.

## Installation Methods

You can install MindInsight either by pip or by source code.

### Install by pip

```bash
pip install mindquantum
```

> - Refers to [MindSpore](https://www.mindspore.cn/versions) to find different version of packages。

### Install by Source Code

1.Download Source Code from Gitee

```bash
cd ~
git clone https://gitee.com/mindspore/mindquantum.git
```

2.Compiling MindSpore Quantum

```bash
cd ~/mindquantum
python setup.py install --user
```

## Verifying Successful Installation

Successfully installed, if there is no error message such as No module named 'mindquantum' when execute the following command:

```bash
python -c 'import mindquantum'
```

## Install with Docker

Mac or Windows users can install MindSpore Quantum through Docker. Please refer to [Docker installation guide](https://gitee.com/mindspore/mindquantum/blob/master/install_with_docker_en.md#).
