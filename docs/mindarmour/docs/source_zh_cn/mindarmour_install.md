# 安装MindArmour

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindarmour/docs/source_zh_cn/mindarmour_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>&nbsp;&nbsp;

## 确认系统环境信息

- 硬件平台为Ascend、GPU或CPU。
- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装。  
    MindArmour与MindSpore的版本需保持一致。
- 其余依赖请参见[setup.py](https://gitee.com/mindspore/mindarmour/blob/r1.7/setup.py)。

## 安装方式

可以采用pip安装或者源码编译安装两种方式。

### pip安装

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindArmour/any/mindarmour-{version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载MindArmour安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindarmour/blob/r1.7/setup.py)），其余情况需自行安装。
> - `{version}`表示MindArmour版本号，例如下载1.3.0版本MindArmour时，`{version}`应写为1.3.0。

### 源码安装

1. 从Gitee下载源码。

    ```bash
    git clone https://gitee.com/mindspore/mindarmour.git -b r1.7
    ```

2. 在源码根目录下，执行如下命令编译并安装MindArmour。

    ```bash
    cd mindarmour
    python setup.py install
    ```

## 验证是否成功安装

执行如下命令，如果没有报错`No module named 'mindarmour'`，则说明安装成功。

```bash
python -c 'import mindarmour'
```