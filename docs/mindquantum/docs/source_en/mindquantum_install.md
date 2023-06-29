# MindQuantum Installation

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindquantum/docs/source_en/mindquantum_install.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

## Confirming System Environment Information

- The hardware platform should be Linux CPU with avx supported.
- Refer to [MindQuantum Installation Guide](https://www.mindspore.cn/install/en), install MindSpore, version 1.2.0 or later is required.
- See [setup.py](https://gitee.com/mindspore/mindquantum/blob/master/setup.py) for the remaining dependencies.

## Installation Methods

You can install MindInsight either by pip or by source code.

### Install by pip

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindQuantum/any/mindquantum-{version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [setup.py](https://gitee.com/mindspore/mindquantum/blob/master/setup.py)). In other cases, you need to manually install dependency items.
> - `{version}` denotes the version of MindQuantum. For example, when you are downloading MindQuantum 0.2.0, `{version}` should be 1.3.0.

### Install by Source Code

1.Download Source Code from Gitee

```bash
cd ~
git clone https://gitee.com/mindspore/mindquantum.git -b r0.2
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

Mac or Windows users can install MindQuantum through Docker. Please refer to [Docker installation guide](https://gitee.com/mindspore/mindquantum/blob/r0.2/install_with_docker_en.md).

## Note

Please set the parallel core number before running MindQuantum scripts. For example, if you want to set the parallel core number to 4, please run the command below:

```bash
export OMP_NUM_THREADS=4
```

For large servers, please set the number of parallel kernels appropriately according to the size of the model to achieve optimal results.
