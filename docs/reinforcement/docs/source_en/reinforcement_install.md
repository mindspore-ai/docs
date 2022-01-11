# MindSpore Reinforcement Installation

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/reinforcement/docs/source_en/reinforcement_install.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

MindSpore Reinforcement depends on the MindSpore training and inference framework. Therefore, install [MindSpore](https://gitee.com/mindspore/mindspore#%E5%AE%89%E8%A3%85) and then MindSpore Reinforcement. You can install MindSpore Reinforcement either by pip or by source code.

## Installation by pip

If use the pip command, download the .whl package from the [MindSpore Reinforcement page](https://www.mindspore.cn/versions/en) and install it.

 ```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{ms_version}/Reinforcement/any/mindspore_rl-{mr_version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see requ    irements.txt). In other cases, you need to manually install dependency items.
> - `{ms_version}` refers to the MindSpore version that matches with MindSpore Reinforcement. For example, if you want to install MindSpore Reinforcement 0.1.0, then,`{ms_version}` should be 1.5.0。
> - `{mr_version}` refers to the version of MindSpore Reinforcement. For example, when you are downloading MindSpore Reinforcement 0.1.0, `{mr_version}` should be 0.1.0.

## Installation by Source Code

Download [source code](https://gitee.com/mindspore/reinforcement)，and enter `reinforcement` directory.

```shell
bash build.sh
pip install output/mindspore_rl-0.1.0-py3-none-any.whl
```

The `build.sh` is the compile script under the `reinforcement` directory。

## Installation Verification

Execute the following command. If it prompts the following information, the installation is successful:

```python
import mindspore_rl
```

