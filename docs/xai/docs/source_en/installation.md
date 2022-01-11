# MindSpore XAI Installation

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/xai/docs/source_en/installation.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## System Requirements

- OS: EulerOS-aarch64, CentOS-aarch64, CentOS-x86, Ubuntu-aarch64 or Ubuntu-x86
- Device: Ascend 910 or GPU CUDA 10.1, 11.1
- Python 3.7.5 or above
- MindSpore 1.4 or above

## Installing by pip

Download the `.whl` package from [MindSpore XAI download page](https://www.mindspore.cn/versions/en) and install with `pip`.

```bash
pip install mindspore_xai-{version}-py3-none-any.whl
```

## Installing from Source Code

1. Download source code from gitee.com:

    ```bash
    git clone https://gitee.com/mindspore/xai.git -b r1.6
    ```

2. Install the dependency python modules:

    ```bash
    cd xai
    pip install -r requirements.txt
    ```

3. Install the XAI module from source code:

    ```bash
    python setup.py install
    ```

4. Optionally, you may build a `.whl` package for installation without step 3:

    ```bash
    bash package.sh
    pip install output/mindspore_xai-{version}-py3-none-any.whl
    ```

## Installation Verification

Upon successful installation, importing 'mindspore_xai' module in Python will cause no error:

```python
import mindspore_xai
print(mindspore_xai.__version__)
```

## Note

[MindInsight](https://www.mindspore.cn/mindinsight/en) is an optional tool for visualizing the model explanation from XAI. Please refer to [Installation](https://www.mindspore.cn/mindinsight/docs/en/r1.6/mindinsight_install.html) for more details.
