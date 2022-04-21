# MindQuantum Installation

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindquantum/docs/source_en/mindquantum_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## Confirming System Environment Information

- Refer to [MindSpore Installation Guide](https://www.mindspore.cn/install/en), install MindSpore, version 1.4.0 or later is required.
- See [setup.py](https://gitee.com/mindspore/mindquantum/blob/r0.6/setup.py) for the remaining dependencies.

## Installation Methods

You can install MindInsight either by pip or by source code.

### Install by pip

Linux-x86_64 Python3.7

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.1/MindQuantum/x86_64/mindquantum-0.5.0-cp37-cp37m-linux_x86_64.whl
```

Windows-x64 Python3.7

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.1/MindQuantum/x86_64/mindquantum-0.5.0-cp37-cp37m-win_amd64.whl
```

Windows-x64 Python3.9

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.1/MindQuantum/x86_64/mindquantum-0.5.0-cp39-cp39-win_amd64.whl
```

> - Refers to [MindSpore](https://www.mindspore.cn/versions) to find different version of packages。

### Install by Source Code

1.Download Source Code from Gitee

```bash
cd ~
git clone https://gitee.com/mindspore/mindquantum.git -b r0.6
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

Mac or Windows users can install MindQuantum through Docker. Please refer to [Docker installation guide](https://gitee.com/mindspore/mindquantum/blob/r0.6/install_with_docker_en.md#).

## Note

Please set the parallel core number before running MindQuantum scripts. For example, if you want to set the parallel core number to 4, please run the command below:

```bash
export OMP_NUM_THREADS=4
```

For large servers, please set the number of parallel kernels appropriately according to the size of the model to achieve optimal results.
