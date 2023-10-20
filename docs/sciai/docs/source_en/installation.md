# MindSpore SciAI Installation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/sciai/docs/source_en/installation.md)
&nbsp;&nbsp;

## System Environment Information Confirmation

- The hardware platform should be Ascend or GPU.
- See [MindSpore Installation Guide](https://www.mindspore.cn/install/en) to install MindSpore.
  The versions of MindSpore Elec and MindSpore must be consistent.
- All other dependencies are included
  in [requirements.txt](https://gitee.com/mindspore/mindscience/blob/master/SciAI/requirements.txt).

## Installation

You can install MindSpore SciAI either by pip or by source code.

### Method 1: Install With Pip

This method installs SciAI from .whl package automatically downloaded from MindSpore website,
which does not require the download and compilation of source code.

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindScience/sciai/gpu/{arch}/cuda-11.1/sciai-{version}-cp37-cp37m-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - When the network is connected, dependencies of the SciAI installation package are automatically downloaded during
    the .whl package installation. For details about dependencies, see setup.py.
> - {version} denotes the version of SciAI. For example, when you are installing SciAI 0.1.0, {version} should
    be `0.1.0`.
> - {arch} denotes the system architecture. For example, the Linux system you are using is x86 architecture 64-bit,
    {arch} should be `x86_64`. If the system is ARM architecture 64-bit, then it should be `aarch64`.

The following table provides the corresponding installation commands to each architecture and Python version.

| Device | Architecture | Python     | Command                                                                                                                                                                                            |
|--------|--------------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ascend | x86_64       | Python=3.7 | `pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindScience/sciai/gpu/x86_64/cuda-11.1/sciai-0.1.0-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple` |
|        | aarch64      | Python=3.7 | `pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindScience/sciai/ascend/aarch64/sciai-0.1.0-cp37-cp37m-linux_aarch64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple`      |
| GPU    | x86_64       | Python=3.7 | `pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindScience/sciai/gpu/x86_64/cuda-11.1/sciai-0.1.0-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple` |

Note: If you have other MindScience package(s) installed in your conda or python env, such as `MindElec`, `MindFlow`
, `MindSponge`, please uninstall the MindScience package(s) in the environment first to avoid pip behavior conflicts.

### Method 2: Install From Source Code

1. Clone the source code from the Git repository of MindScience.

    ```bash
    cd ~
    git clone https://gitee.com/mindspore/mindscience.git
    ```

2. Build SciAI with script `build.sh`.

    ```bash
    cd mindscience/SciAI
    bash build.sh -j8
    ```

3. Install the `.whl` package

    ```bash
    bash install.sh
    ```

### Installation Verification

To verify the installation, run the following commands. If the error message `No module named 'sciai'` is not displayed,
the installation is successful.

```bash
python -c 'import sciai'
```
