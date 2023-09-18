# MindSpore Serving Installation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/serving/docs/source_en/serving_install.md)

## Installation

Currently, MindSpore Serving can be deployed only in the Linux environment.

MindSpore Serving wheel packages are common to various hardware platforms(Nvidia GPU, Ascend 910/310P/310, CPU). The inference task depends on the MindSpore or MindSpore Lite inference framework. We need to select one of them as the Serving Inference backend. When these two inference backend both exist, Mindspore Lite inference framework will be used.

MindSpore and MindSpore Lite have different build packages for different hardware platforms. The following table lists the target devices and model formats supported by each build package.

|Inference backend|Build platform|Target device|Supported model formats|
|---------| --- | --- | -------- |
|MindSpore| Nvidia GPU | Nvidia GPU | `MindIR` |
|  | Ascend | Ascend 910 | `MindIR` |
|  |  | Ascend 310P/310 | `MindIR`, `OM` |
|MindSpore Lite| Nvidia GPU | Nvidia GPU, CPU | `MindIR_Lite` |
|  | Ascend | Ascend 310P/310, CPU | `MindIR_Lite` |
|  | CPU | CPU | `MindIR_Lite` |

When [MindSpore](https://www.mindspore.cn/) is used as the inference backend, MindSpore Serving supports the Ascend 910/310P/310 and Nvidia GPU environments. The Ascend 310P/310 environment supports both `OM` and `MindIR` model formats, and the Ascend 910 and GPU environment only supports the `MindIR` model format.

Due to the dependency between MindSpore Serving and MindSpore, please follow the table below, download and install the corresponding MindSpore verision from [MindSpore download page](https://www.mindspore.cn/versions/en).

| MindSpore Serving Version       |                        Branch                          | MindSpore Version |
| -----------------------------   | ---------------------------------------------------    | ---------------   |
|              2.0.0              | [r2.0](https://gitee.com/mindspore/serving/tree/r2.0/) |       2.0.0       |
|              1.9.0              | [r1.9](https://gitee.com/mindspore/serving/tree/r1.9/) |       1.9.0       |
|              1.8.0              | [r1.8](https://gitee.com/mindspore/serving/tree/r1.8/) |   1.8.0, 1.8.1    |
|              1.7.0              | [r1.7](https://gitee.com/mindspore/serving/tree/r1.7/) |       1.7.0       |

For details about how to install and configure MindSpore, see [Installing MindSpore](https://gitee.com/mindspore/mindspore/blob/master/README.md#installation) and [Configuring MindSpore](https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend_install_source_en.md#configuring-environment-variables).

When [MindSpore Lite](https://www.mindspore.cn/lite) is used as the inference backend, MindSpore Serving supports Ascend 310P/310, Nvidia GPU and CPU environments. Only the `MindIR_Lite` model formats is supported. Models of format `MindIR` exported from MindSpore or models of other frameworks need be be converted to `MindIR_Lite` format by the MindSpore Lite conversion tool. The `MindIR_Lite` models converted from `Ascend310` and `Ascend310P` environments are different, and the `MindIR_Lite` models must be running on the corresponding `Ascend310` or `Ascend310P` environments. `MindIR_Lite` models converted from Nvidia GPU and CPU environments can be running only in the Nvidia GPU and CPU environments.

| Inference backend  | Running environment of Lite conversion tool  | Target device of `MindIR_Lite` models |
| -------------- | ---------------- | --------------- |
| MindSpore Lite | Nvidia GPU, CPU  | Nvidia GPU, CPU |
|                | Ascend 310       | Ascend 310      |
|                | Ascend 310P       | Ascend 310P      |

For details about how to compile and install MindSpore Lite, see the [MindSpore Lite Documentation](https://www.mindspore.cn/lite/docs/en/master/index.html).
We should configure the environment variable `LD_LIBRARY_PATH` to indicates the installation path of `libmindspore-lite.so`.

We can install MindSpore Serving either by pip or by source code.

### Installation by pip

Perform the following steps to install Serving:

If use the pip command, download the .whl package from the [MindSpore Serving page](https://www.mindspore.cn/versions/en) and install it.

```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/Serving/{arch}/mindspore_serving-{version}-{python_version}-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - `{version}` denotes the version of MindSpore Serving. For example, when you are downloading MindSpore Serving 1.1.0, `{version}` should be 1.1.0.
> - `{arch}` denotes the system architecture. For example, the Linux system you are using is x86 architecture 64-bit, `{arch}` should be `x86_64`. If the system is ARM architecture 64-bit, then it should be `aarch64`.
> - `{python_version}` spcecifies the python version for which MindSpore Serving is built. If you wish to use Python3.7, `{python_version}` should be `cp37-cp37m`. If Python3.8 is used, it should be `cp38-cp38`. If Python3.9 is used, it should be `cp39-cp39`. Please use the same Python environment whereby MindSpore Serving is installed.

### Installation by Source Code

Install Serving using the [source code](https://gitee.com/mindspore/serving).

```shell
git clone https://gitee.com/mindspore/serving.git -b master
cd serving
bash build.sh
```

For the `bash build.sh` above, we can add `-jn`, for example `-j16`, to accelerate compilation. By adding `-S on`
option, third-party dependencies can be downloaded from gitee instead of github.

MindSpore Serving compliation depends on the MindSpore inference header files. During the preceding compilation, the MindSpore source code will be downloaded. If the MindSpore whl or MindSpore Lite package has been installed, we can run the following compilation command to avoid downloading the MindSpore source code.

```shell
git clone https://gitee.com/mindspore/serving.git -b master
cd serving
bash build.sh -p ${mindspore_path}/lib
```

Use the `-p` parameter to specify the path of MindSpore whl or MindSpore Lite package. The `${mindspore_path}` indicates the installation path of MindSpore whl package or `runtime` path in MindSpore Lite tar package.

After the build is complete, find the .whl installation package of Serving in the `serving/build/package/` directory
and install it.

```python
pip install mindspore_serving-{version}-{python_version}-linux_{arch}.whl
```

## Installation Verification

Run the following commands to verify the installation. Import the Python module. If no error is reported, the installation is successful.

```python
from mindspore_serving import server
```
