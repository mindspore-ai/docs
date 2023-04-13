# 安装MindSpore Armour

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindarmour/docs/source_zh_cn/mindarmour_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>&nbsp;&nbsp;

## 确认系统环境信息

- 硬件平台为Ascend、GPU或CPU。
- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装。  
    MindSpore Armour与MindSpore的版本需保持一致。
- 其余依赖请参见[setup.py](https://gitee.com/mindspore/mindarmour/blob/r2.0/setup.py)。

## MindSpore版本依赖关系

由于MindSpore Armour与MindSpore有依赖关系，请按照下表所示的对应关系，在[MindSpore下载页面](https://www.mindspore.cn/versions)下载并安装对应的whl包。

| MindSpore Armour | 分支                                                      | MindSpore |
| ---------- | --------------------------------------------------------- | --------- |
| 2.0.0      | [r2.0](https://gitee.com/mindspore/mindarmour/tree/r2.0/) | >=1.7.0   |
| 1.9.0      | [r1.9](https://gitee.com/mindspore/mindarmour/tree/r1.9/) | >=1.7.0   |
| 1.8.0      | [r1.8](https://gitee.com/mindspore/mindarmour/tree/r1.8/) | >=1.7.0   |
| 1.7.0      | [r1.7](https://gitee.com/mindspore/mindarmour/tree/r1.7/) | 1.7.0     |

## 安装方式

可以采用pip安装或者源码编译安装两种方式。

### pip安装

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindArmour/any/mindarmour-{version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载MindSpore Armour安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindarmour/blob/r2.0/setup.py)），其余情况需自行安装。
> - `{version}`表示MindSpore Armour版本号，例如下载1.3.0版本MindSpore Armour时，`{version}`应写为1.3.0。

### 源码安装

1. 从Gitee下载源码。

    ```bash
    git clone https://gitee.com/mindspore/mindarmour.git
    ```

2. 在源码根目录下，执行如下命令编译并安装MindSpore Armour。

    ```bash
    cd mindarmour
    python setup.py install
    ```

## 验证是否成功安装

执行如下命令，如果没有报错`No module named 'mindarmour'`，则说明安装成功。

```bash
python -c 'import mindarmour'
```