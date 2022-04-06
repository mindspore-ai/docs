# 安装MindInsight

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindinsight/docs/source_zh_cn/mindinsight_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 确认系统环境信息

- 硬件平台支持Ascend、GPU和CPU。
- 确认安装Python 3.7.5或3.9.0版本。如果未安装或者已安装其他版本的Python，可以选择下载并安装：
    - Python 3.7.5版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)或[华为云](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz)。
    - Python 3.9.0版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz)或[华为云](https://mirrors.huaweicloud.com/python/3.9.0/Python-3.9.0.tgz)。
- MindInsight与MindSpore的版本需保持一致。
- 若采用源码编译安装，还需确认安装以下依赖。
    - 确认安装[node.js](https://nodejs.org/en/download/) 10.19.0及以上版本。
    - 确认安装[wheel](https://pypi.org/project/wheel/) 0.32.0及以上版本。
- 其他依赖参见[requirements.txt](https://gitee.com/mindspore/mindinsight/blob/master/requirements.txt)。

## 安装方式

可以采用pip安装，源码编译安装和Docker安装三种方式。

### pip安装

安装PyPI上的版本:

```bash
pip install mindinsight
```

安装自定义版本:

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindInsight/any/mindinsight-{version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载MindInsight安装包的依赖项（依赖项详情参见[requirements.txt](https://gitee.com/mindspore/mindinsight/blob/master/requirements.txt)），其余情况需自行安装。
> - `{version}`表示MindInsight版本号，例如下载1.3.0版本MindInsight时，`{version}`应写为1.3.0。
> - MindInsight支持使用x86 64位或ARM 64位架构的Linux发行版系统。

### 源码编译安装

#### 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindinsight.git
```

#### 编译安装MindInsight

可选择以下任意一种安装方式：

1. 在源码根目录下执行如下命令。

    ```bash
    cd mindinsight
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    python setup.py install
    ```

2. 构建`whl`包进行安装。

    进入源码的根目录，先执行`build`目录下的MindInsight编译脚本，再执行命令安装`output`目录下生成的`whl`包。

    ```bash
    cd mindinsight
    bash build/build.sh
    pip install output/mindinsight-{version}-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

### Docker安装

MindSpore的镜像包含MindInsight功能，请参考官网[安装指导](https://www.mindspore.cn/install)。

## 验证是否成功安装

执行如下命令：

```bash
mindinsight start
```

如果出现下列提示，说明安装成功：

```bash
Web address: http://127.0.0.1:8080
service start state: success
```
