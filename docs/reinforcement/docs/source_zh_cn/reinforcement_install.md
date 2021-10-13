# 安装MindSpore Reinforcement

<!-- TOC -->

- [安装MindSpore Reinforcement](#安装mindspore-reinforcement)
    - [pip安装](#pip安装)
    - [源码编译安装](#源码编译安装)
    - [验证安装是否成功](#验证安装是否成功)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/reinforcement/docs/source_zh_cn/reinforcement_install.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

MindSpore Reinforcement依赖MindSpore训练推理框架，安装完[MindSpore](https://gitee.com/mindspore/mindspore#%E5%AE%89%E8%A3%85)，再安装MindSpore Reinforcement。可以采用pip安装或者源码编译安装两种方式。

## pip安装

使用pip命令安装，请从[MindSpore Reinforcement下载页面](https://www.mindspore.cn/versions)下载并安装whl包。

 ```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{ms_version}/Reinforcement/any/mindspore_rl-{mr_version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载MindSpore Reinforcement安装包的依赖项（依赖项详情参见requirement.txt），其余情况需自行安装。
> - `{ms_version}`表示与MindSpore Reinforcement匹配的MindSpore版本号，例如下载0.1.0版本MindSpore Reinforcement时，`{ms_version}`应写为1.5.0。
> - `{mr_version}`表示MindSpore Reinforcement版本号，例如下载0.1.0版本MindSpore Reinforcement时，`{mr_version}`应写为0.1.0。

## 源码编译安装

下载[源码](https://gitee.com/mindspore/reinforcement)，下载后进入`reinforcement`目录。

```shell
bash build.sh
pip install output/mindspore_rl-0.1-py3-none-any.whl
```

其中，`build.sh`为`reinforcement`目录下的编译脚本文件。

## 验证安装是否成功

执行以下命令，验证安装结果。导入Python模块不报错即安装成功：

```python
import mindspore_rl
```
