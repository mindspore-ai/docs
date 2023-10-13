# MindSpore Recommender Installation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/recommender/docs/source_en/install.md)

MindSpore Recommender relies on the MindSpore training framework, so after installing [MindSpore](https://gitee.com/mindspore/mindspore/blob/r2.2/README.md#installation), install MindSpore Recommender. You can use either a pip installation or a source code compilation installation.

## Installing from pip Command

To install through the pip command, download and install the whl package from the [MindSpore Recommender download page](https://www.mindspore.cn/versions).

```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{ms_version}/Recommender/any/mindspore_rec-{mr_version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- The dependencies of the MindSpore Recommender installation package will be downloaded automatically when the whl package is installed while the network is connected (see requirement.txt for details of the dependencies), otherwise you need to install them yourself.
- `{ms_version}` indicates the MindSpore version number that matches the MindSpore Recommender.
- `{mr_version}` indicates the version number of MindSpore Recommender, for example, when downloading version 0.2.0 of MindSpore Recommender, `{mr_version}` should be set as 0.2.0.

## Installing from Source Code

Download the [source code](https://github.com/mindspore-lab/mindrec) and go to the `mindrec` directory after downloading.

```shell
bash build.sh
pip install output/mindspore_rec-0.2.0-py3-none-any.whl
```

`build.sh` is the compilation script file in the `recommender` directory.

## Verification

Execute the following command to verify the installation result. The installation is successful if no error is reported when importing Python modules.

```python
import mindspore_rec
```
