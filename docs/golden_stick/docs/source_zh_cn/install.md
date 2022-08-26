# 安装MindSpore Golden Stick

<a href="https://gitee.com/mindspore/docs/blob/r1.8/docs/golden_stick/docs/source_zh_cn/install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source.png"></a>

## 环境限制

下表列出了安装、编译和运行MindSpore Golden Stick所需的系统环境：

| 软件名称 |  版本   |
| :-----: | :-----: |
| Ubuntu  |  18.04  |
| Python  |  3.7-3.9 |

> 其他的三方依赖请参考[requirements文件](https://gitee.com/mindspore/golden-stick/blob/r0.1/requirements.txt)。
> 当前MindSpore Golden Stick仅能在Ubuntu18.04上运行。

## MindSpore版本依赖关系

MindSpore Golden Stick依赖MindSpore训练推理框架，请按照根据下表中所指示的对应关系，并参考[MindSpore安装指导](https://mindspore.cn/install)安装对应版本的MindSpore。

| MindSpore Golden Stick版本 |                             分支                             | MindSpore版本 |
| :---------------------: | :----------------------------------------------------------: | :-------: |
|          0.1.0          | [r0.1](https://gitee.com/mindspore/golden-stick/tree/r0.1/) |   1.8.0   |

安装完MindSpore后，继续安装MindSpore Golden Stick。可以采用pip安装或者源码编译安装两种方式。

## pip安装

使用pip命令安装，请从[MindSpore Golden Stick下载页面](https://www.mindspore.cn/versions)下载并安装whl包。

 ```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{ms_version}/GoldenStick/any/mindspore_gs-{mg_version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载MindSpore Golden Stick安装包的依赖项（依赖项详情参见requirement.txt），其余情况需自行安装。
> - `{ms_version}`表示与MindSpore Golden Stick匹配的MindSpore版本号，例如下载0.1.0版本MindSpore Golden Stick时，`{ms_version}`应写为1.8.0。
> - `{mg_version}`表示MindSpore Golden Stick版本号，例如下载0.1.0版本MindSpore Golden Stick时，`{mg_version}`应写为0.1.0。

## 源码编译安装

下载[源码](https://gitee.com/mindspore/golden-stick)，下载后进入`golden_stick`目录。

```shell
bash build.sh
pip install output/mindspore_gs-0.1.0-py3-none-any.whl
```

其中，`build.sh`为`golden_stick`目录下的编译脚本文件。

## 验证安装是否成功

执行以下命令，验证安装结果。导入Python模块不报错即安装成功。

```python
import mindspore_gs
```
