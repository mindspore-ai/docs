# 安装MindPandas

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindpandas/docs/source_zh_cn/mindpandas_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 确认系统环境信息

下表列出了安装、编译和运行MindPandas所需的系统环境：

| 软件名称 |                版本                |
| :------: |:--------------------------------:|
|  Linux-x86_64操作系统 | Ubuntu \>=18.04<br/>Euler \>=2.9 |
|  Python  |             3.8-3.9              |
|  glibc  |             \>=2.25              |

- 请确保环境中安装了libxml2-utils。
- 其他的第三方依赖请参考[requirements文件](https://gitee.com/mindspore/mindpandas/blob/master/requirements.txt)。

## 安装方式

### pip安装

请从[MindPandas下载页面](https://www.mindspore.cn/versions)下载whl包，使用`pip`指令安装。

> 在联网状态下，安装whl包时会自动下载MindPandas安装包的依赖项（依赖项详情参见requirements.txt），其余情况需自行安装。

### 源码安装

下载[源码](https://gitee.com/mindspore/mindpandas.git)，下载后进入mindpandas目录，运行build.sh脚本。

```shell
git clone https://gitee.com/mindspore/mindpandas.git
cd mindpandas
bash build.sh
```

编译完成后，whl包在output目录下，使用pip安装，以Python3.8为例，安装命令如下所示：

```shell
pip install output/mindpandas-0.2.0-cp38-cp38-linux_x86_64.whl
```

## 验证安装是否成功

在shell中执行以下命令，如果没有报错`No module named 'mindpandas'`，则说明安装成功。

```shell
python -c "import mindpandas"
```
