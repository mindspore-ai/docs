# 安装MindSpore Vision

<a href="https://gitee.com/mindspore/docs/blob/master/docs/vision/source_zh_cn/mindvision_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 环境依赖

- numpy 1.17+
- opencv-python 4.1+
- pytest 4.3+
- [MindSpore](https://www.mindspore.cn/install) 1.5+
- ml_collection
- tqdm
- pillow

## 安装

### 环境准备

- 创建一个conda虚拟环境并且激活。

    ```shell
    conda create -n mindvision python=3.7.5 -y
    conda activate mindvision
    ```

- 安装MindSpore

    ```shell
    pip install mindspore
    ```

### 安装MindSpore Vision

- 使用git克隆MindSpore Vision仓库。

    ```shell
    git clone https://gitee.com/mindspore/vision.git
    cd vision
    ```

- 源码安装

    ```shell
    python setup.py install
    ```

- 使用pip安装

    ```shell
    pip install mindvision
    ```

### 验证

为了验证MindVision和所需的环境是否正确安装，我们可以运行示例代码来初始化一个分类器然后推理一张图片。

推理所用的[图片](https://gitee.com/mindspore/vision/blob/master/tests/st/classification/dataset/mnist/mnist.jpg) 来自MNIST数据集，用户可以通过参数`device_target`来自定义推理所用的平台。

```shell
python ./examples/classification/lenet/lenet_mnist_infer.py \
        --data_url ./tests/st/classification/dataset/mnist/mnist.jpg \
        --pretrained True \
        --device_target CPU
```

```text
{4: 'four'}
```

如果您成功安装，以上代码应该会成功运行。
