# 确认系统环境信息

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindtransformer/docs/source_zh_cn/mindtransformer_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

- 硬件平台支持Ascend，GPU和CPU。
- 确认安装[Python](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) 3.7.5版本。
- Mindformer与MindSpore的版本需保持一致。目前配套关系如下

|版本对应关系| Mindformer  | MindSpore |
|-----------| -----------| ----------|
|版本号      | 0.2.0      | 2.0     |
|版本号      | 0.1.0      | 1.8     |

- 若采用源码编译安装，还需确认安装以下依赖。

    - 确认安装[node.js](https://nodejs.org/en/download/) 10.19.0及以上版本。

    - 确认安装[wheel](https://pypi.org/project/wheel/) 0.32.0及以上版本。

- 其他依赖参见[requirements.txt](https://gitee.com/mindspore/transformer/blob/r2.0.0-alpha/requirements.txt)。

## 安装方式

可以采用源码编译安装方式。

### 源码编译安装

#### 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/transformer.git
```

#### 编译安装Mindformer

可选择以下任意一种安装方式：

1. 在源码根目录下执行如下命令。

    ```bash
    cd transformer
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    python setup.py install
    ```

2. 构建`whl`包进行安装。

    进入源码的根目录，先执行`build`目录下的Mindformer编译脚本，再执行命令安装`output`目录下生成的`whl`包。

    ```bash
    cd transformer
    bash build.sh
    pip install output/mindformer-{version}-py3-none-any.whl
    ```

### 验证是否成功安装

执行如下命令：

```bash
python -c "import mindformer"
```

若能正常导入，则说明安装成功。
