# Installing MindSpore Golden Stick

<a href="https://gitee.com/mindspore/docs/blob/r1.8/docs/golden_stick/docs/source_en/install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source.png"></a>

The MindSpore Golden Stick depends on the MindSpore training and inference framework. You need to install [MindSpore](https://gitee.com/mindspore/mindspore/blob/r1.8/README.md#installation) before installing MindSpore Golden Stick. You can use pip or source code build for installation.

## Installation Using pip

Run the pip command to download the .whl package from the [MindSpore Golden Stick download page](https://www.mindspore.cn/versions) and install it.

 ```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{ms_version}/golden_stick/any/mindspore_gs-{mg_version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - When the system is connected to the Internet, the dependency items of the installation package are automatically downloaded during the installation of the .whl package. For details about the dependency items, see the **requirement.txt** file. In other cases, you need to manually install the dependency items.
> - `{ms_version}` indicates the MindSpore version number that matches the MindSpore Golden Stick. For example, when downloading the MindSpore Golden Stick 0.1.0, set `{ms_version}` to `1.8.0`.
> - `{mg_version}` indicates the version number of MindSpore Golden Stick. For example, when downloading MindSpore Golden Stick 0.1.0, set `{mg_version}` to `0.1.0`.

## Installation Using Source Code Build

Download the [source code](https://gitee.com/mindspore/golden-stick) and go to the `golden_stick` directory.

```shell
bash build.sh
pip install output/mindspore_gs-0.1.0-py3-none-any.whl
```

`build.sh` is the build script in the `golden-stick` directory.

## Verifying the Installation

Run the following commands to verify the installation. Import the Python module. If no error is reported, the installation is successful.

```python
import mindspore_gs
```
