# MindSpore Reinforcement Installation

<!-- TOC -->

- [MindSpore Reinforcement Installation](#mindspore-reinforcement-installation)
    - [Installation by pip](#installation-by-pip)
    - [Installation by Source Code](#installation-by-source-code)  
    - [Installation Verification](#installation-verification)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/reinforcement/docs/source_en/reinforcement_install.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

MindSpore Reinforcement depends on the MindSpore training and inference framework. Therefore, install [MindSpore](https://gitee.com/mindspore/mindspore#%E5%AE%89%E8%A3%85) and then MindSpore Reinforecement. You can install MindSpore Reinforcement either by pip or by source code.

## Installation by pip

//FIXME: 更新versions页面

If use the pip command, download the .whl package from the [MindSpore Reinforcement page](https://www.mindspore.cn/versions/en) and install it.

//FIXME: 确定Reinforcement安装包名，及发布路径

 ```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/Rl/{arch}/mindspore_rl-{version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see requ    irements.txt). In other cases, you need to manually install dependency items.
> - `{version}` denotes the version of MindSpore Reinforcement. For example, when you are downloading MindSpore Reinforcement 0.1.0, `{version}` should be 0.1.0.

## Installation by Source Code

Download [source code](https://gitee.com/mindspore/reinforcement)，and enter `reinforcement` directory。

```shell
bash build.sh
pip install output/mindspore_rl-0.1-py3-none-any.whl
```

The `build.sh` is the compile script under the `reinforcement` directory。

## Installation Verification

Execute the following command. If it prompts the following information, the installation is successful:

```python
import mindspore_rl
```

