# Installing MindSpore Golden Stick

<a href="https://gitee.com/mindspore/docs/blob/master/docs/golden_stick/docs/source_en/install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Environmental restrictions

The following table lists the environment required for installing, compiling and running MindSpore Golden Stick:

| software | version |
| :-----: | :-----: |
| Ubuntu  |  18.04  |
| Python  |  3.7-3.9 |

> Please refer to [requirements](https://gitee.com/mindspore/golden-stick/blob/master/requirements.txt) for other third party dependencies.
> MindSpore Golden Stick can only run on Ubuntu18.04.

## Version dependency

The MindSpore Golden Stick depends on the MindSpore training and inference framework, please refer the table below and [MindSpore Installation Guide](https://mindspore.cn/install) to install the corresponding MindSpore verision.

| MindSpore Golden Stick Version |                            Branch                            | MindSpore version |
| :-----------------------------: | :----------------------------------------------------------: | :-------: |
|          0.1.0          | [r0.1](https://gitee.com/mindspore/golden-stick/tree/r0.1/) |   1.8.0   |

After MindSpore is installed, you can use pip or source code build for MindSpore Golden Stick installation.

## Installing from pip command

If you use the pip command, please download the whl package from [MindSpore Golden Stick](https://www.mindspore.cn/versions/en) page and install it.

```shell
pip install  https://ms-release.obs.cn-north-4.myhuaweicloud.com/{MindSpore_version}/GoldenStick/any/mindspore_rl-{mg_version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - Installing whl package will download MindSpore Golden Stick dependencies automatically (detail of dependencies is shown in requirement.txt), other dependencies should install manually.
> - `{MindSpore_version}` stands for the version of MindSpore. For the version matching relationship between MindSpore and MindSpore Golden Stick, please refer to [page](https://www.mindspore.cn/versions).
> - `{ms_version}` stands for the version of MindSpore Golden Stick. For example, if you would like to download version 0.1.0, you should fill 1.8.0 in `{MindSpore_version}` and fill 0.1.0 in `{mg_version}`.

## Installing from source code

Download [source code](https://gitee.com/mindspore/golden-stick), then enter the `golden-stick` directory.

```shell
bash build.sh
pip install output/mindspore_gs-{mg_version}-py3-none-any.whl
```

`build.sh` is the compiling script in `golden-stick` directory.

### Verification

If you can successfully execute following command, then the installation is completed.

```python
import mindspore_gs
```