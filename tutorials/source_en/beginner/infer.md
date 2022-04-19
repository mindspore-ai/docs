# Inference and Deployment

<a href="https://gitee.com/mindspore/docs/blob/r1.7/tutorials/source_en/beginner/infer.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

This chapter uses the `mobilenet_v2` network fine-tuning approach in MindSpore Vision to develop an AI application to classify dogs and croissants, and deploy the trained network model on the Android phone to perform inference and deployment.

## Data Preparation and Loading

### Downloading a Dataset

First, you need to download the [dog and croissants classification dataset](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/beginner/DogCroissants.zip) which contains two classes, dog and croissants. Each class contains about 150 training images, 20 verification images, and 1 inference image.

The dataset is as follows:

![datset-dog](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/tutorials/source_zh_cn/beginner/images/datset_dog.png)

Use the `DownLoad` interface in MindSpore Vision to download and decompress the dataset to the specified path. The sample code is as follows:

```python
from mindvision.dataset import DownLoad

dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/beginner/DogCroissants.zip"
path = "./datasets"

dl = DownLoad()
# Download and decompress the dataset.
dl.download_and_extract_archive(dataset_url, path)
```

The directory structure of the dataset is as follows:

```text
datasets
└── DogCroissants
    ├── infer
    │   ├── croissants.jpg
    │   └── dog.jpg
    ├── train
    │   ├── croissants
    │   └── dog
    └── val
        ├── croissants
        └── dog
```

### Loading the Dataset

Define the `create_dataset` function to load the dog and croissant dataset, perform image argumentation on the dataset, and set batch_size of the dataset.

```python
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as transforms

def create_dataset(path, batch_size=10, train=True, image_size=224):
    dataset = ds.ImageFolderDataset(path, num_parallel_workers=8, class_indexing={"croissants": 0, "dog": 1})

    # Image augmentation
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    if train:
        trans = [
            transforms.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            transforms.RandomHorizontalFlip(prob=0.5),
            transforms.Normalize(mean=mean, std=std),
            transforms.HWC2CHW()
        ]
    else:
        trans = [
            transforms.Decode(),
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.Normalize(mean=mean, std=std),
            transforms.HWC2CHW()
        ]

    dataset = dataset.map(operations=trans, input_columns="image", num_parallel_workers=8)
    # Set the value of the batch_size. Discard the samples if the number of samples last fetched is less than the value of batch_size.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset
```

Load the training dataset and validation dataset for subsequent model training and validation.

```python
# Load the training dataset.
train_path = "./datasets/DogCroissants/train"
dataset_train = create_dataset(train_path, train=True)

# Load the validation dataset.
val_path = "./datasets/DogCroissants/val"
dataset_val = create_dataset(val_path, train=False)
```

## Model Training

In this case, we use a pre-trained model to fine-tune the model on the dog and croissant classification dataset, and convert the trained CKPT model file to the MINDIR format for subsequent deployment on the mobile phone.

> Currently, model training supports only the Linux environment.

### Principles of the MobileNet V2 Model

MobileNet is a lightweight CNN proposed by the Google team in 2017 to focus on mobile, embedded, or IoT devices. Compared with traditional convolutional neural networks, MobileNet uses depthwise separable convolution to greatly reduce the model parameters and computation amount with a slight decrease in accuracy. In addition, the width coefficient $\alpha$ and resolution coefficient $\beta$ are introduced to meet the requirements of different application scenarios.

Because a large amount of data is lost when the ReLU activation function in the MobileNet processes low-dimensional feature information, the MobileNetV2 proposes to use an inverted residual block and Linear Bottlenecks to design the network, to improve accuracy of the model and make the optimized model smaller.

![mobilenet](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/tutorials/source_zh_cn/beginner/images/mobilenet.png)

In the inverted residual block structure, the 1 x 1 convolution is used for dimension increase, the 3 x 3 DepthWise convolution is used, and the 1 x 1 convolution is used for dimension reduction. This structure is opposite to the residual block structure. For the residual block, the 1 x 1 convolution is first used for dimension reduction, then the 3 x 3 convolution is used, and finally the 1 x 1 convolution is used for dimension increase.

> For details, see the [MobileNet V2 paper.](https://arxiv.org/pdf/1801.04381.pdf)

### Downloading the Pre-trained Model

Download the [ckpt file of the MobileNetV2 pre-trained model](https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_224.ckpt) required by the case. The width coefficient of the pre-training model is $\alpha=1.0$, and the input image size is (224, 224). Save the downloaded pre-trained model to the current directory. Use `DownLoad` in MindSpore Vision to download the pre-trained model file to the current directory. The sample code is as follows:

```python
from mindvision.dataset import DownLoad

models_url = "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_224.ckpt"

dl = DownLoad()
# Download the pre-trained model file.
dl.download_url(models_url)
```

### MobileNet V2 Model Fine-tuning

This chapter uses MobileNet V2 pre-trained model for fine-tuning, and uses the dog and croissant classification dataset to retrain the model to update the model parameter by deleting the last parameter of the 1 x 1 convolution layer for classification in the MobileNet V2 pre-trained model.

```python
import mindspore.nn as nn
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net

from mindvision.classification.models import mobilenet_v2
from mindvision.engine.loss import CrossEntropySmooth

# Create a model, in which the number of target classifications is 2 and the input image size is (224,224).
network = mobilenet_v2(num_classes=2, resize=224)

# Save model parameters to param_dict.
param_dict = load_checkpoint("./mobilenet_v2_1.0_224.ckpt")

# Obtain the parameter name of the last convolutional layer of the mobilenet_v2 network.
filter_list = [x.name for x in network.head.classifier.get_parameters()]

# Delete the last convolutional layer of the pre-trained model.
def filter_ckpt_parameter(origin_dict, param_filter):
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break

filter_ckpt_parameter(param_dict, filter_list)

# Load the pre-trained model parameters as the network initialization weight.
load_param_into_net(network, param_dict)

# Define the optimizer.
network_opt = nn.Momentum(params=network.trainable_params(), learning_rate=0.01, momentum=0.9)

# Define the loss function.
network_loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.1, classes_num=2)

# Define evaluation metrics.
metrics = {"Accuracy": nn.Accuracy()}

# Initialize the model.
model = Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)
```

```text
[WARNING] ME(375486:140361546602304,MainProcess): [mindspore/train/serialization.py:644] 2 parameters in the 'net' are not loaded, because they are not in the 'parameter_dict'.
[WARNING] ME(375486:140361546602304,MainProcess): [mindspore/train/serialization.py:646] head.classifier.weight is not loaded.
[WARNING] ME(375486:140361546602304,MainProcess): [mindspore/train/serialization.py:646] head.classifier.bias is not loaded.

Delete parameter from checkpoint:  head.classifier.weight
Delete parameter from checkpoint:  head.classifier.bias
Delete parameter from checkpoint:  moments.head.classifier.weight
Delete parameter from checkpoint:  moments.head.classifier.bias
```

> The preceding warning is generated because the last convolutional layer parameter of the pre-trained model needs to be deleted for model fine-tuning. When the pre-trained model is loaded, the system displays a message indicating that the `head.classifier` parameter is not loaded. The `head.classifier` parameter uses the initial value during model creation.

### Model Training and Evaluation

Train and evaluate the network, and use the `mindvision.engine.callback.ValAccMonitor` interface in MindSpore Vision to print the loss value and the evaluation accuracy of the training. After the training is completed, save the CKPT file with the highest evaluation accuracy, `best.ckpt`, in the current directory.

```python
from mindvision.engine.callback import ValAccMonitor
from mindspore.train.callback import TimeMonitor

num_epochs = 10

# Train and verify the model. After the training is completed, save the CKPT file with the highest evaluation accuracy, `best.ckpt`, in the current directory.
model.train(num_epochs,
            dataset_train,
            callbacks=[ValAccMonitor(model, dataset_val, num_epochs), TimeMonitor()])
```

```text
--------------------
Epoch: [  1 /  10], Train Loss: [0.388], Accuracy:  0.975
epoch time: 7390.423 ms, per step time: 254.842 ms
--------------------
Epoch: [  2 /  10], Train Loss: [0.378], Accuracy:  0.975
epoch time: 1876.590 ms, per step time: 64.710 ms
--------------------
Epoch: [  3 /  10], Train Loss: [0.372], Accuracy:  1.000
epoch time: 2103.431 ms, per step time: 72.532 ms
--------------------
Epoch: [  4 /  10], Train Loss: [0.346], Accuracy:  1.000
epoch time: 2246.303 ms, per step time: 77.459 ms
--------------------
Epoch: [  5 /  10], Train Loss: [0.376], Accuracy:  1.000
epoch time: 2164.527 ms, per step time: 74.639 ms
--------------------
Epoch: [  6 /  10], Train Loss: [0.353], Accuracy:  1.000
epoch time: 2191.490 ms, per step time: 75.569 ms
--------------------
Epoch: [  7 /  10], Train Loss: [0.414], Accuracy:  1.000
epoch time: 2183.388 ms, per step time: 75.289 ms
--------------------
Epoch: [  8 /  10], Train Loss: [0.362], Accuracy:  1.000
epoch time: 2219.950 ms, per step time: 76.550 ms
--------------------
Epoch: [  9 /  10], Train Loss: [0.354], Accuracy:  1.000
epoch time: 2174.555 ms, per step time: 74.985 ms
--------------------
Epoch: [ 10 /  10], Train Loss: [0.364], Accuracy:  1.000
epoch time: 2190.957 ms, per step time: 75.550 ms
================================================================================
End of validation the best Accuracy is:  1.000, save the best ckpt file in ./best.ckpt
```

### Visualizing Model Predictions

Define the `visualize_model` function, use the model with the highest validation accuracy described above to predict the input image, and visualize the prediction result.

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mindspore import Tensor

def visualize_model(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((224, 224))
    plt.imshow(image)

    # Normalization processing
    mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
    std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])
    image = np.array(image)
    image = (image - mean) / std
    image = image.astype(np.float32)

    # Convert the image channel from (h, w, c) to (c, h, w).
    image = np.transpose(image, (2, 0, 1))

    # Extend the data dimension to (1, c, h, w)
    image = np.expand_dims(image, axis=0)

    # Define and load the network.
    net = mobilenet_v2(num_classes=2, resize=224)
    param_dict = load_checkpoint("./best.ckpt")
    load_param_into_net(net, param_dict)
    model = Model(net)

    # Use the model for prediction.
    pre = model.predict(Tensor(image))
    result = np.argmax(pre)

    class_name = {0: "Croissants", 1: "Dog"}
    plt.title(f"Predict: {class_name[result]}")
    return result

image1 = "./datasets/DogCroissants/infer/croissants.jpg"
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
visualize_model(image1)

image2 = "./datasets/DogCroissants/infer/dog.jpg"
plt.subplot(1, 2, 2)
visualize_model(image2)

plt.show()
```

### Model Export

After model training is complete, the trained network model (CKPT file) is converted into the MindIR format for subsequent inference on the mobile phone. The `mobilenet_v2_1.0_224.mindir` file is generated in the current directory through the `export` interface.

```python
from mindspore import export, Tensor

# Define and load the network parameters.
net = mobilenet_v2(num_classes=2, resize=224)
param_dict = load_checkpoint("best.ckpt")
load_param_into_net(net, param_dict)

# Export the model from the CKPT format to the MINDIR format.
input_np = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)
export(net, Tensor(input_np), file_name="mobilenet_v2_1.0_224", file_format="MINDIR")
```

## Inference and Deployment on the Mobile Phone

To implement the inference function of the model file on the mobile phone, perform the following steps:

- Convert file format: Convert MindIR file format to the MindSpore Lite recognizable file on the Android phone.
- Application deployment: Deploy the app APK on the mobile phone, that is, download a MindSpore Vision suite Android APK.
- Application experience: After importing the MS model file to the mobile phone, experience the function of recognizing dogs and croissants.

### Converting the File Format

Use the [conversion tool](https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/converter_tool.html) applied on the device side to convert the mobilenet_v2_1.0_224.mindir file generated during the training process into the mobilenet_v2_1.0_224.ms file which can be recognized by the MindSpore Lite on-device inference framework.

The following describes how to convert the model file format:

1. Use MindSpore Lite Converter to convert the file format in Linux. [Linux-x86_64 tool download link](https://www.mindspore.cn/lite/docs/en/r1.7/use/downloads.html)

```shell
# Download and decompress the software package and set the path of the software package. {converter_path} indicates the path of the decompressed tool package, and PACKAGE_ROOT_PATH indicates the environment variable.
export PACKAGE_ROOT_PATH={converter_path}

# Add the dynamic link library required by the converter to the environment variable LD_LIBRARY_PATH.
export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/tools/converter/lib:${LD_LIBRARY_PATH}

# Run the conversion command on the mindspore-lite-linux-x64/tools/converter/converter.
./converter_lite --fmk=MINDIR --modelFile=mobilenet_v2_1.0_224.mindir  --outputFile=mobilenet_v2_1.0_224
```

2. Use MindSpore Lite Converter to convert the file format in Windows. [Windows-x64 tool download link](https://www.mindspore.cn/lite/docs/en/r1.7/use/downloads.html)

```shell
# Download and decompress the software package and set the path of the software package. {converter_path} indicates the path of the decompressed tool package, and PACKAGE_ROOT_PATH indicates the environment variable.
set PACKAGE_ROOT_PATH={converter_path}

# Add the dynamic link library required by the converter to the environment variable PATH.
set PATH=%PACKAGE_ROOT_PATH%\tools\converter\lib;%PATH%

# Run the following command in the mindspore-lite-win-x64\tools\converter\converter directory:
call converter_lite --fmk=MINDIR --modelFile=mobilenet_v2_1.0_224.mindir --outputFile=mobilenet_v2_1.0_224
```

After the conversion is successful, `CONVERT RESULT SUCCESS:0` is displayed, and the `mobilenet_v2_1.0_224.ms` file is generated in the current directory.

> For details about how to download MindSpore Lite Converter in other environments, see [Download MindSpore Lite](https://www.mindspore.cn/lite/docs/en/r1.7/use/downloads.html).

### Application Deployment

Download [Android app APK](https://gitee.com/mindspore/vision/releases/) of the MindSpore Vision Suite and install the APK on your phone. The app name is `MindSpore Vision`.

> The MindSpore Vision APK is used as an example of the visual development tool. It provides basic UI functions such as photographing and image selection, and provides AI application demos such as classification, detection, and face recognition.

Open the app, tap the `classification` module on the home screen, and then tap the middle button to take photos or tap the image album button on the top bar to select an image for classification.

![main](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/tutorials/source_zh_cn/beginner/images/app1.png)

By default, the MindSpore Vision `classification` module has a built-in general AI network model for image identification and classification.

![result](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/tutorials/source_zh_cn/beginner/images/app2.png)

### Application Experience

Finally, the custom network model `mobilenet_v2_1.0_224.ms` trained above is deployed to the Android mobile phone to experience the recognition function of dogs and croissants.

#### Customizing the Model Label Files

To deploy a custom model, you need to define the information required by the network model in the following format, that is, customize a label file, and create a label file in JSON format named `custom.json` on the local computer.

```text
{
    "title": 'dog and croissants',
    "file": 'mobilenet_v2_1.0_224.ms',
    "label": ['croissants', 'dog']
}
```

The JSON label file must contain the `title`, `file`, and `label` key fields, which are described as follows:

- title: custom module titles (dog and croissants).
- file: the name of the model file converted above.
- label: `array` information about the custom label.

#### Labels and Model Files Deployed to Mobile Phones

On the home page of the `MindSpore Vision APK`, hold down the `classification` button to enter the custom classification mode and select the labels and model files to be deployed.

To implement the identification function of the dogs and croissants on the mobile phone, you need to place the label file `custom.json` and model file `mobilenet_v2_1.0_224.ms` to the specified directory on the mobile phone. The `Android/data/Download/` folder is used as an example. Place the label file and model file in the preceding mobile phone address, as shown in the following figure. Click the custom button. The system file function is displayed. Click icon in the upper left corner and find the directory where the JSON label file and model file are stored, and select the corresponding JSON file.

![step](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/tutorials/source_zh_cn/beginner/images/app3.png)

After the label and model file are deployed on the mobile phone, you can click the middle button to take photos and obtain images, or click the image button on the upper side bar to select an image album for images. In this way, the dogs and croissants can be classified and identified.

![result1](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/tutorials/source_zh_cn/beginner/images/app4.png)

> This chapter only covers the simple deployment process on the mobile phone. For more information about inference, please refer to [MindSpore Lite](https://www.mindspore.cn/lite/docs/en/r1.7/index.html).
