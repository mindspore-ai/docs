# 安装MindSpore Recommender

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/recommender/docs/source_zh_cn/install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

MindSpore Recommender依赖MindSpore训练框架，安装完[MindSpore](https://gitee.com/mindspore/mindspore#安装)，再安装MindSpore Recommender。可以采用pip安装或者源码编译安装两种方式。

## pip安装

使用pip命令安装，请从[MindSpore Recommender下载页面](https://www.mindspore.cn/versions)下载并安装whl包。

```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{ms_version}/Recommender/any/mindspore_rec-{mr_version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- 在联网状态下，安装whl包时会自动下载MindSpore Recommender安装包的依赖项（依赖项详情参见requirement.txt），其余情况需自行安装。
- `{ms_version}`表示与MindSpore Recommender匹配的MindSpore版本号。
- `{mr_version}`表示MindSpore Recommender版本号，例如下载0.2.0版本MindSpore Recommender时，`{mr_version}`应写为0.2.0。

## 源码编译安装

下载[源码](https://gitee.com/mindspore/recommender)，下载后进入`recommender`目录。

```shell
bash build.sh
pip install output/mindspore_rec-0.2.0-py3-none-any.whl
```

其中，`build.sh`为`recommender`目录下的编译脚本文件。

## 验证安装是否成功

执行以下命令，验证安装结果。导入Python模块不报错即安装成功：

```python
import mindspore_rec
```
