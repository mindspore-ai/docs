# MindSpore SciAI Installation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/master/docs/sciai/docs/source_en/installation.md)&nbsp;&nbsp;

## System Environment Information Confirmation

- The hardware platform should be Ascend or GPU.
- See [MindSpore Installation Guide](https://www.mindspore.cn/install/en) to install MindSpore.
    The versions of MindSpore Elec and MindSpore must be consistent.\
- All other dependencies are included in [requirements.txt](https://gitee.com/mindspore/mindscience/blob/master/SciAI/requirements.txt).

## Installation

You can install MindSpore SciAI either by pip or by source code.

### Installation by pip

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{ms_version}/mindscience/{arch}/sciai_{device}-{version}-{python_version}-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - When the network is connected, dependencies of the SciAI installation package are automatically downloaded during the .whl package installation. For details about dependencies, see setup.py.
> - {ms_version} denotes the MindSpore version. For example `2.0.0rc1`.
> - {version} denotes the version of SciAI. For example, when you are installing SciAI 0.1.0, {version} should be `0.1.0`.
> - {arch} denotes the system architecture. For example, the Linux system you are using is x86 architecture 64-bit, {arch} should be `x86_64`. If the system is ARM architecture 64-bit, then it should be `aarch64`.
> - {python_version} specifies the python version of which SciAI is built. If you wish to use Python3.7.5, {python_version} should be `cp37-cp37m`. If Python3.9.0 is used, it should be `cp39-cp39`.

The following table provides corresponding installation command to each architecture and python version.

| Device | Architecture | Python    | Command                                                                                                                                                                                    |
|--------|--------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ascend | x86_64       | Python3.7 | `pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/mindscience/x86_64/sciai_ascend-0.1.0-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple`   |
|        | aarch64      | Python3.7 | `pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/mindscience/aarch64/sciai_ascend-0.1.0-cp37-cp37m-linux_aarch64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| GPU    | x86_64       | Python3.7 | `pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/mindscience/x86_64/sciai_gpu-0.1.0-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple`      |

### Installation by Source Code

1. Download the source code from the code repository.

    ```bash
    cd ~
    git clone https://gitee.com/mindspore/mindscience.git
    ```

2. Build SciAI with script `build.sh`.

    Ascend backend

    ```bash
    cd {PATH}/mindscience/SciAI
    bash build.sh -e ascend -j8
    ```

    GPU backend

    ```bash
    cd {PATH}/mindscience/SciAI
    bash build.sh -e gpu -j8
    ```

3. Install the `.whl` package

    ```bash
    cd {PATH}/mindscience/SciAI/output
    pip install sciai_{device}-{version}-cp37-cp37m-linux_{arch}.whl
    ```

### Installation Verification

To verify the installation, run the following commands. If the error message `No module named 'sciai'` is not displayed, the installation is successful.

```bash
python -c 'import sciai'
```
