# Using Benchmarks

<a href="https://gitee.com/mindspore/docs/blob/master/docs/xai/docs/source_en/using_benchmarks.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## What are Benchmarks

Benchmarks are algorithms evaluating the goodness of saliency maps from explainers. MindSpore XAI currently provides 4 benchmarks for image classification scenario: `Robustness`, `Faithfulness`, `ClassSensitivity` and `Localization`.

## Preparations

The tutorial below is referencing [using_benchmarks.py](https://gitee.com/mindspore/xai/blob/master/examples/using_benchmarks.py).

Please follow the [Downloading Data Package](https://www.mindspore.cn/xai/docs/en/master/using_explainers.html#downloading-data-package) instructions to download the necessary files for the tutorial.

With the tutorial package, we have to get the sample image, trained classifier, explainer and optionally the saliency map ready:

```python
# have to change the current directory to xai/examples/ first
from mindspore import load_checkpoint, load_param_into_net, set_context, PYNATIVE_MODE
from mindspore_xai.explanation import GradCAM

from common.resnet import resnet50
from common.dataset import load_image_tensor

# only PYNATIVE_MODE is supported
set_context(mode=PYNATIVE_MODE)

# 20 classes
num_classes = 20

# load the trained classifier
net = resnet50(num_classes)
param_dict = load_checkpoint("xai_examples_data/ckpt/resnet50.ckpt")
load_param_into_net(net, param_dict)

# [1, 3, 224, 224] Tensor
boat_image = load_image_tensor('xai_examples_data/test/boat.jpg')

# explainer
grad_cam = GradCAM(net, layer='layer4')

# 5 is the class id of 'boat'
saliency = grad_cam(boat_image, targets=5)
```

## Using Robustness

`Robustness` is the simplest benchmark, it perturbs the inputs by adding random noise and outputs the maximum sensitivity as evaluation score from the perturbations:

```python
from mindspore.nn import Softmax
from mindspore_xai.explanation import Robustness

# the classifier use Softmax as activation function
robustness = Robustness(num_classes, activation_fn=Softmax())
# the 'saliency' argument is optional
score = robustness.evaluate(grad_cam, boat_image, targets=5, saliency=saliency)
```

The returned `score` is a 1D tensor with only one float value for an 1xCx224x224 image tensor.

## Using Faithfulness and ClassSensitivity

The ways of using `Faithfulness` and `ClassSensitivity` are very similar to `Robustness`. However, `ClassSensitivity` is class agnostic, `targets` can not be specified.

## Using Localization

If the object region or bounding box is provided, `Localization` can be used. It evaluates base on how many saliency pixels fall inside the object region:

```python
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore_xai.explanation import Localization

# top-left:80,66 bottom-right:223,196 is the bounding box of a boat
mask = np.zeros([1, 1, 224, 224])
mask[:, :, 66:196, 80:223] = 1

mask = Tensor(mask, dtype=ms.float32)

localization = Localization(num_classes)

score = localization.evaluate(grad_cam, boat_image, targets=5, mask=mask)
```
