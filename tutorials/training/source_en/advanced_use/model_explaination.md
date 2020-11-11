# Explain models

`Linux` `Ascend` `GPU` `Model Optimization` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Explain Models](#explain-models)
    - [Overview](#overview)
    - [Operation Process](#operation-process)
        - [Preparing the Script](#preparing-the-script)
        - [Enabling MindInsight](#enabling-mindInsight)
    - [Pages and Functions](#pages-and-functions)
        - [Saliency Map Visualization](#saliency-map-visualization)
        - [Explanation Method Assessment](#explanation-method-assessment)
            - [Comprehensive Assessment](#comprehensive-assessment)
            - [Classification Assessment](#classification-assessment)

<!--/ TOC -->

<a href="https://gitee.com/mindspore/docs/tree/master/tutorials/training/source_en/advanced_use/model_explaination.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

Currently, most deep learning models are black-box models with good performance but poor explainability. The model explanation module aims to provide users with explanation of the model decision basis, help users better understand the model, trust the model, and improve the model when an error occurs in the model.

In some critical application scenarios, such as automatic driving, financial decision-making, etc., AI model cannot be truly applied if it is not interpretable for legal and policy supervision reasons. Therefore, the interpretability of the model is becoming more and more important. As a consequence, model explaination is an important part of improving mindspore's  applicability and humanization.

To be specific, in the task of image classification, there is one explaination method that highlights the most critical area that affects the classification decision of the model. We call it "saliency map". If people see that the part of the model that is concerned is just the key feature of the classification label, then the features learned by the model are correct, and we can trust the effect and decision of the model. If the model focuses on irrelevant parts, even if the prediction label is correct, it does not mean that the model is reliable, we still need to optimize and improve the model. This may be due to the association of some irrelevant elements in the training data. Model developers can consider targeted data enhancement to correct the bias learned by the model.

After a variety of interpretation methods are available, we also provide a set of measurement methods to evaluate the effect of interpretation methods from various dimensions. It helps users compare and select the interpretation methods that are most suitable for a particular scene.

## Operation Process

### Preparing the Script

Currently, MindSpore provides the explanation methods and explanation measurement Python API.  You can use the provided explanation methods by  ```mindspore.explainer.explanation``` and the provided explanation meaturement by ```mindspore.explainer.benchmark```. You need to prepare the black-box model and data to be explained, instantiate explanation methods or explanation measurement according to your need and call the explanation API in your script to collect the explanation result and explanation measurement result.

MindSpore also provides ```mindspore.explainer.ExplainRunner``` to run all explanation methods and explanation measurement automatically. You just need to put the instantiated object into ```run``` of ```ExplainRunner``` and then all explanation methods and explanation metric are executed and explanation logs containing explanation results and explanation measurement results are automatically generated.

The following uses ResNet-50 and 20 types of multi-tag data as an example. Add the initialization and calling code of the explanation method on the basis of the original script. The explanation method GradCAM is used to explain the model and the measurement methods are used to evaluate the explanation method. The sample code is as follows:

```python
import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net

from mindspore.explainer.explanation import GradCAM, GuidedBackprop
from mindspore.explainer.benchmark import Faithfulness, Localization
from mindspore.explainer import ExplainRunner

num_classes = 20
# please refer to model_zoo for the model architecture of resnet50
model = resnet50(num_classes)
param_dict = load_checkpoint("resnet50.ckpt")
load_param_into_net(model, param_dict)


# combine the model architecture with its final activation layer, eg.Sigmoid() for multi-label models or Softmax() for single-label models
model = nn.SequentialCell([model, nn.Sigmoid()])
model.set_grad(False)
model.set_train(False)

# initialize explainers with the loaded black-box model
gradcam = GradCAM(model, layer='0.layer4')
guidedbackprop = GuidedBackprop(model)


# initialize benchmarkers to evaluate the chosen explainers
faithfulness = Faithfulness(num_labels=num_classes, metric='InsertionAUC')
localization = Localization(num_labels=num_classes, metric='PointingGame')

# returns the dataset to be explained, when localization is chosen, the dataset is required to provide bounding box
# the columns of the dataset should be in [image], [image, labels], or [image, labels, bbox] (order matters).
# You may refer to 'mindspore.dataset.project' for columns managements.
dataset_path = "/dataset_dir"
dataset = get_dataset(dataset_path)

# specify the class names of the dataset
classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]

dataset_with_classes = (dataset, classes)
explainers = [gradcam, guidedbackprop]
benchmarkers = [faithfulness, localization]

# initialize runner with specified summary_dir
runner = ExplainRunner(summary_dir='./summary_dir')

# execute runner.run to generate explanation and evaluation results to save it to summary_dir
runner.run(dataset_with_classes, explainers, benchmarkers)
```

> - Only support CNN of image classification models, such as Lenet, Resnet, Alexnet.
> - Only support PyNative mode.

### Enabling MindInsight

Enable MindInsight and click **Model Explanation** on the top of the page. All explanation log paths are displayed. When a log path meets the conditions, the **Saliency Map Visualization** buttons are displayed in the **Operation** column.

![xai_index](./images/xai_index.png)

## Pages and Functions

### Saliency Map Visualization

Saliency map visualization is used to display the image area that has the most significant impact on the model decision-making result. Generally, the image can be considered as a key feature of the objective classification.

![xai_saliency_map](./images/xai_saliency_map.png)

The following information is displayed on the **Saliency Map Visualization** page:

- Objective dataset set by a user through the Python API of the dataset.
- Ground truth tags, prediction tags, and the prediction probabilities of the model for the corresponding tags. The system adds the TP, TN, FP and FN flags(meanings are provided in the page's information) in the upper left corner of the corresponding tag based on the actual requirements.
- A saliency map given by the selected explanation method.

Operations:

1. Select the required explanation methods. Currently, we support four explanation methods. More explanation methods will be provided in the future.
2. Click **Overlay on Original Image** in the upper right corner of the page to overlay the saliency map on the original image.
3. Click different tags to display the saliency map analysis results of the model for different tags. For different classification results, the focus of the model is usually different.
4. Use the tag filtering function on the upper part of the page to filter out images with specified tags.
5. Select an image display sequence from **Sort Images By** in the upper right corner of the page.
6. Click **View Score** on the right of an explanation method. The page for assessing all explanation methods is displayed.
7. Click image you will see the higher resolution image.

![xai_saliency_map_detail](./images/xai_saliency_map_detail.png)

### Explanation Method Assessment

#### Comprehensive Assessment

The provided explanation methods are scored from different dimensions. We provide various dimensions scores to help users compare the performance and select the most suitable one. You can configure weights for metrics in a specific scenario to obtain the comprehensive score.

![xai_metrix_comprehensive](./images/xai_metrix_comprehensive.png)

#### Classification Assessment

The classification assessment page provides two types of comparison. One is to compare scores of different measurement dimensions of the same explanation method in each tag. The other is to compare scores of different explanation methods of the same measurement dimension in each tag.

![xai_metrix_class](./images/xai_metrix_class.png)
