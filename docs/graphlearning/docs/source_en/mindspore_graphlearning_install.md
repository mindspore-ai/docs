# Install Graph Learning

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/graphlearning/docs/source_en/mindspore_graphlearning_install.md)&nbsp;&nbsp;

## Installation

### System Environment Information Confirmation

- Ensure that the hardware platform is Ascend or GPU under the Linux system.
- Refer to [MindSpore Installation Guide](https://www.mindspore.cn/install/en) to complete the installation of MindSpore, which requires at least version 2.0.0.
- For other dependencies, please refer to [requirements.txt](https://gitee.com/mindspore/graphlearning/blob/master/requirements.txt).

### Installation Methods

You can install MindSpore Graph Learning either by pip or by source code.

#### Installation by pip

- Ascend/CPU

    ```bash
    pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/GraphLearning/cpu/{system_structure}/mindspore_gl-0.2-cp37-cp37m-linux_{system_structure}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

- GPU

    ```bash
    pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/GraphLearning/gpu/x86_64/cuda-{cuda_verison}/mindspore_gl-0.2-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

> - When the network is connected, dependency items are automatically downloaded during .whl package installation. For details about other dependency items, see [requirements.txt](https://gitee.com/mindspore/graphlearning/blob/master/requirements.txt). In other cases, you need to manually install dependency items.
> - `{system_structure}` denotes the Linux system architecture, and the option is `x86_64` and `arrch64`.
> - `{cuda_verison}` denotes the CUDA version, and the option is `10.1`, `11.1` and `11.6`。

#### Installation by Source Code

1. Download source code from Gitee.

    ```bash
    git clone https://gitee.com/mindspore/graphlearning.git
    ```

2. Compile and install in MindSpore Graph Learning directory.

    ```bash
    cd graphlearning
    bash build.sh
    pip install ./output/mindspore_gl-*.whl
    ```

### Installation Verification

Successfully installed, if there is no error message such as `No module named 'mindspore_gl'` when execute the following command:

```bash
python -c 'import mindspore_gl'
```
