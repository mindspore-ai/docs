# MindQuantum Installation

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindquantum/docs/source_en/mindquantum_install.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

<!-- TOC --->

- [MindQuantum Installation](#mindquantum-installation)
    - [Confirming System Environment Information](#confirming-system-environment-information)
    - [Installation Methods](#installation-methods)
        - [Install by Source Code](#install-by-source-code)
        - [Install by pip](#install-by-pip)
    - [Verifying Successful Installation](#verifying-successful-installation)
    - [Install with Docker](#install-with-docker)
    - [Note](#Note)

<!-- /TOC -->

## Confirming System Environment Information

- The hardware platform should be Linux CPU with avx supported.
- Refer to [MindSpore Installation Guide](https://www.mindspore.cn/install/en), install MindSpore, version 1.2.0 or later is required.
- See [setup.py](https://gitee.com/mindspore/mindquantum/blob/r0.5/setup.py) for the remaining dependencies.

## Installation Methods

You can install MindInsight either by pip or by source code.

### Install by pip

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{ms_version}/MindQuantum/any/mindquantum-{mq_version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [setup.py](https://gitee.com/mindspore/mindquantum/blob/r0.5/setup.py)). In other cases, you need to manually install dependency items.
> - `{ms_version}` refers to the MindSpore version that matches with MindQuantum. For example, if you want to install MindQuantum 0.3.0, then,`{ms_version}` should be 1.5.0。
> - `{mq_version}` denotes the version of MindQuantum. For example, when you are downloading MindQuantum 0.3.0, `{version}` should be 0.3.0.
> - Refers to [MindSpore](https://www.mindspore.cn/versions) to find different version of packages。

### Install by Source Code

1.Download Source Code from Gitee

```bash
cd ~
git clone https://gitee.com/mindspore/mindquantum.git -b r0.5
```

2.Compiling MindQuantum

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

Mac or Windows users can install MindQuantum through Docker. Please refer to [Docker installation guide](https://gitee.com/mindspore/mindquantum/blob/r0.5/install_with_docker_en.md).

## Note

Please set the parallel core number before running MindQuantum scripts. For example, if you want to set the parallel core number to 4, please run the command below:

```bash
export OMP_NUM_THREADS=4
```

For large servers, please set the number of parallel kernels appropriately according to the size of the model to achieve optimal results.
