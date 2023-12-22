# 安装 MindSpore XAI

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/xai/docs/source_zh_cn/installation.md)

## 系統要求

- 操作系統: EulerOS-aarch64、CentOS-aarch64、CentOS-x86、Ubuntu-aarch64 或 Ubuntu-x86
- 硬件平台: Atlas训练系列产品 或 GPU CUDA 10.1、11.1
- Python 3.7.5 或 3.9.0
- MindSpore 1.7 或以上

## pip安装

从 [MindSpore XAI下载页面](https://www.mindspore.cn/versions) 下载并安装whl包。

```bash
pip install mindspore_xai-{version}-py3-none-any.whl
```

## 从源码安装

1. 从gitee.com下载源码：

    ```bash
    git clone https://gitee.com/mindspore/xai.git
    ```

2. 安装所有依赖的Python包：

    ```bash
    cd xai
    pip install -r requirements.txt
    ```

3. 从源码安装XAI：

    ```bash
    python setup.py install
    ```

4. 你也可以跳过第三步，打包一个`.whl`安装包:

    ```bash
    bash package.sh
    pip install output/mindspore_xai-{version}-py3-none-any.whl
    ```

## 验证是否安装成功

成功安装后，在Python运行以下代码会印出已安装的XAI版本：

```python
import mindspore_xai
print(mindspore_xai.__version__)
```
