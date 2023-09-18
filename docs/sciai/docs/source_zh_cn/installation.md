# MindSpore SciAI安装

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/sciai/docs/source_zh_cn/installation.md)&nbsp;&nbsp;

## 确认系统环境信息

- 硬件平台为Ascend或GPU。
- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装。  
- 其余依赖请参见[requirements.txt](https://gitee.com/mindspore/mindscience/blob/master/SciAI/requirements.txt)。

## 安装方式

可以采用pip安装或者源码编译安装两种方式。

### pip安装

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{ms_version}/mindscience/{arch}/sciai-{version}-{python_version}-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载SciAI安装包的依赖项（依赖项详情参见setup.py）。
> - {ms_version}表示MindSpore版本号，例如`2.0.0rc1`。
> - {version}表示SciAI版本号，例如下载0.1.0版本SciAI时，{version}应写为`0.1.0`。
> - {arch}表示系统架构，例如使用的Linux系统是x86架构64位时，{arch}应写为`x86_64`。如果系统是ARM架构64位，则写为`aarch64`。
> - {python_version}表示用户的Python版本，Python版本为3.7.5时，{python_version}应写为`cp37-cp37m`。Python版本为3.9.0时，则写为`cp39-cp39`。

下表提供了各架构和Python版本对应的安装命令。

| 设备     | 架构      | Python    | 安装命令                                                                                                                                                                                       |
|--------|---------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ascend | x86_64  | Python3.7 | `pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/mindscience/x86_64/sciai_ascend-0.1.0-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple`   |
|        | aarch64 | Python3.7 | `pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/mindscience/aarch64/sciai_ascend-0.1.0-cp37-cp37m-linux_aarch64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| GPU    | x86_64  | Python3.7 | `pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/mindscience/x86_64/sciai_gpu-0.1.0-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple`      |

### 源码安装

1. 下载MindScience代码仓库。

    ```bash
    cd ~
    git clone https://gitee.com/mindspore/mindscience.git
    ```

2. 使用脚本`build.sh`编译SciAI。

    昇腾Ascend后端

    ```bash
    cd {PATH}/mindscience/SciAI
    bash build.sh -e ascend -j8
    ```

    GPU后端

    ```bash
    cd {PATH}/mindscience/SciAI
    bash build.sh -e gpu -j8
    ```

3. 编译完成后，通过如下命令安装编译所得`.whl`包。

    ```bash
    cd {PATH}/mindscience/SciAI/output
    pip install sciai_{device}-{version}-cp37-cp37m-linux_{arch}.whl
    ```

### 验证安装

执行如下命令，如果没有报错`No module named 'sciai'`，则说明安装成功。

```bash
python -c 'import sciai'
```
