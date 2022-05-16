# Install MindSpore Vision

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/vision/source_en/mindvision_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## Prerequisites

- numpy 1.17+
- opencv-python 4.1+
- pytest 4.3+
- [mindspore](https://www.mindspore.cn/install) 1.5+
- ml_collection
- tqdm
- pillow

## Installation

### Prepare environment

- Create a conda virtual environment and activate it.

    ```shell
    conda create -n mindvision python=3.7.5 -y
    conda activate mindvision
    ```

- Install MindSpore

    ```shell
    pip install mindspore
    ```

### Install MindSpore Vision

- Clone the MindSpore Vision repository.

    ```shell
    git clone https://gitee.com/mindspore/vision.git
    cd vision
    ```

- Installing MindSpore Vision by Source Code

    ```shell
    python setup.py install
    ```

- Installing MindSpore Vision by pip

    ```shell
    pip install mindvision
    ```

### Verification

To verify whether MindVision and the required environment are installed correctly, we can run sample Python code to
initialize a classificer and run inference on a demo image.

The [image](https://gitee.com/mindspore/vision/blob/r0.1/tests/st/classification/dataset/mnist/mnist.jpg) used for inference is from the MNIST dataset. Users can use the parameter `device_target` to customize the platform for inference.

```shell
python ./examples/classification/lenet/lenet_mnist_infer.py --data_url ./tests/st/classification/dataset/mnist/mnist.jpg --pretrained True --device_target CPU
```

```text
{4: 'four'}
```

The above code is supposed to run successfully upon you finish the installation.