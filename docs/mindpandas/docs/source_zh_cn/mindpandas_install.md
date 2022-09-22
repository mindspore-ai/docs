# 安装MindPandas

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindpandas/docs/source_zh_cn/mindpandas_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 环境限制

下表列出了安装、编译和运行MindPandas所需的系统环境：

| 软件名称 |  版本   |
| :------: | :-----: |
|  Ubuntu  |  18.04  |
|  Python  | 3.7-3.9 |

> - 当MindPandas设置多进程后端执行模式时，Python版本需为3.8，且仅能在Ubuntu18.04上运行。
> - 其他的三方依赖请参考[requirements文件](https://gitee.com/mindspore/mindpandas/blob/master/requirements.txt)。

## pip安装

使用pip命令安装，请从[MindPandas下载页面](https://www.mindspore.cn/versions)下载并安装whl包。

 ```shell
pip install mindpandas -i https://pypi.tuna.tsinghua.edu.cn/simple
 ```

> 在联网状态下，安装whl包时会自动下载MindPandas安装包的依赖项（依赖项详情参见requirement.txt），其余情况需自行安装。

## 源码编译安装

下载[源码](https://gitee.com/mindspore/mindpandas.git)，下载后进入`mindpandas/build`目录，运行bash.sh脚本。

```shell
git clone https://gitee.com/mindspore/mindpandas.git
cd mindpandas/build
bash build.sh
pip install ../output/mindpandas-0.1.0-cp38-cp38m-linux_x86_64.whl
```

其中，`build.sh`为`mindpandas`目录下的编译脚本文件。

## 验证安装是否成功

执行以下命令，验证安装结果。导入Python模块不报错即安装成功。

```python
import mindpandas
```