# 安装MindSpore Hub

- [安装MindSpore Hub](#安装mindspore-hub)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装方式](#安装方式)
        - [pip安装](#pip安装)
        - [源码安装](#源码安装)
    - [验证是否成功安装](#验证是否成功安装)

<a href="https://gitee.com/mindspore/docs/blob/master/docs/hub/docs/source_zh_cn/hub_installation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 确认系统环境信息

- 硬件平台支持Ascend、GPU和CPU。
- 确认安装[Python](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) 3.7.5版本。
- MindSpore Hub与MindSpore的版本需保持一致。
- MindSpore Hub支持使用x86 64位或ARM 64位架构的Linux发行版系统。
- 在联网状态下，安装whl包时会自动下载`setup.py`中的依赖项，其余情况需自行安装。

## 安装方式

可以采用pip安装或者源码安装两种方式。

### pip安装

下载并安装[发布版本列表](https://www.mindspore.cn/versions)中的MindSpore Hub whl包。

```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/Hub/any/mindspore_hub-{version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - `{version}`表示MindSpore Hub版本号，例如下载1.3.0版本MindSpore Hub时，`{version}`应写为1.3.0。

### 源码安装

1. 从Gitee下载源码。

   ```bash
   git clone https://gitee.com/mindspore/hub.git
   ```

2. 编译安装MindSpore Hub。

   ```bash
   cd hub
   python setup.py install
   ```

## 验证是否成功安装

在能联网的环境中执行以下命令，验证安装结果。

```python
import mindspore_hub as mshub

model = mshub.load("mindspore/1.6/lenet_mnist", num_class=10)
```

如果出现下列提示，说明安装成功：

```text
Downloading data from url https://gitee.com/mindspore/hub/raw/master/mshub_res/assets/mindspore/1.6/lenet_mnist.md

Download finished!
File size = 0.00 Mb
Checking /home/ma-user/.mscache/mindspore/1.6/lenet_mnist.md...Passed!
```

## FAQ

<font size=3>**Q: 遇到`SSL: CERTIFICATE_VERIFY_FAILED`怎么办？**</font>

A: 由于你的网络环境，例如你使用代理连接互联网，往往会由于证书配置问题导致python出现ssl verification failed的问题，此时有两种解决方法：

- 配置好SSL证书 **（推荐）**
- 在加载mindspore_hub前增加如下代码进行解决（最快）

   ```python
   import ssl
   ssl._create_default_https_context = ssl._create_unverified_context

   import mindspore_hub as mshub
   model = mshub.load("mindspore/1.6/lenet_mnist", num_classes=10)
   ```

<font size=3>**Q: 遇到`No module named src.*`怎么办？**</font>

A: 同一进程中使用load接口加载不同的模型，由于每次加载模型需要将模型文件目录插入到环境变量中，经测试发现：Python只会去最开始插入的目录下查找src.*，尽管你将最开始插入的目录删除，Python还是会去这个目录下查找。解决办法：不添加环境变量，将模型目录下的所有文件都复制到当前工作目录下。代码如下：

```python
# mindspore_hub_install_path/load.py
def _copy_all_file_to_target_path(path, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    path = os.path.realpath(path)
    target_path = os.path.realpath(target_path)
    for p in os.listdir(path):
        copy_path = os.path.join(path, p)
        target_dir = os.path.join(target_path, p)
        _delete_if_exist(target_dir)
        if os.path.isdir(copy_path):
            _copy_all_file_to_target_path(copy_path, target_dir)
        else:
            shutil.copy(copy_path, target_dir)

def _get_network_from_cache(name, path, *args, **kwargs):
    _copy_all_file_to_target_path(path, os.getcwd())
    config_path = os.path.join(os.getcwd(), HUB_CONFIG_FILE)
    if not os.path.exists(config_path):
        raise ValueError('{} not exists.'.format(config_path))
    ......
```

**注意**： 在load后一个模型时可能会将前一个模型的一些文件替换掉，但是模型训练需保证必要模型文件存在，你必须在加载新模型之前完成对前一个模型的训练。
