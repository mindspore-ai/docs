# Inference and Deployment

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/infer.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

This chapter uses the `mobilenet_v2` network fine-tuning approach in MindSpore Vision to develop an AI application (classification of the dog and the croissants) and deploy the trained network model to the Android phone to perform inference and deployment functions.

## Data Preparation and Loading

### Downloading the dataset

First, you need to download the [dog and croissants classification dataset](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/beginner/DogCroissants.zip) used in this case, which has two categories, dog and croissants, and each class has about 150 training images, 20 verification images, and 1 inference image.

The specific dataset is as follows:

![datset-dog](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/beginner/images/datset_dog.png)

Use the `DownLoad` interface in MindSpore Vision to download and extract the dataset to the specified path, and the sample code is as follows:

```python
from mindvision.dataset import DownLoad

dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/beginner/DogCroissants.zip"
path = "./datasets"

dl = DownLoad()
# Download and extract the dataset
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

Define the `create_dataset` function to load the dog and croissants dataset, perform image enhancement operations on the dataset, and set the dataset batch_size size.

```python
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as transforms

def create_dataset(path, batch_size=10, train=True, image_size=224):
    dataset = ds.ImageFolderDataset(path, num_parallel_workers=8, class_indexing={"croissants": 0, "dog": 1})

    # Image augmentation operation
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
    # Sets the size of the batch_size and discards if the number of samples last fetched is less than batch_size
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset
```

Load the training dataset and validation dataset for subsequent model training and validation.

```python
# Load the training dataset
train_path = "./datasets/DogCroissants/train"
dataset_train = create_dataset(train_path, train=True)

# Load the validation dataset
val_path = "./datasets/DogCroissants/val"
dataset_val = create_dataset(val_path, train=False)
```

## Model Training

In this case, we use a pre-trained model to fine-tune the model on the classification dataset of the dog and croissants, and convert the trained CKPT model file to the MINDIR format for subsequent deployment on the phone side.

> Model training currently only supports running in the Linux environment.

### Principles of the MobileNet V2 Model

MobileNet network is a lightweight CNN network focused on mobile, embedding or IoT devices proposed by the Google team in 2017. Compared to the traditional convolutional neural network, MobileNet network uses depthwise separable convolution idea in the premise of a small reduction in accuracy, which greatly reduces the model parameters and amount of operation. And the introduction of width coefficient and resolution coefficient makes the model meet the needs of different application scenarios.

Since there is a large amount of loss when the Relu activation function processes low-dimensional feature information in the MobileNet network, the MobileNet V2 network proposes to use the inverted residual block and Linear Bottlenecks to design the network, to improve the accuracy of the model and make the optimized model smaller.

![mobilenet](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/beginner/images/mobilenet.png)

The Inverted residual block structure in the figure first uses 1x1 convolution for upswing, uses 3x3 DepthWise convolution, and finally uses 1x1 convolution for dimensionality reduction, which is in contrast to the Residual block structure. The Residual block first uses 1x1 convolution for dimensionality reduction, uses 3x3 convolution, and finally uses 1x1 convolution for upswing.

> For detailed contents, refer to [MobileNet V2 thesis](https://arxiv.org/pdf/1801.04381.pdf).

### Downloading the Pre-trained Model

Download the [ckpt file of the MobileNetV2 pre-trained model](https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_224.ckpt) required for the case and the width coefficient of the pre-trained model, and the input image size is (224, 224). The downloaded pre-trained model is saved in the current directory. Use the `DownLoad` in MindSpore Vision to download the pre-trained model file to the current directory, and the sample code is as follows:

```python
from mindvision.dataset import DownLoad

models_url = "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_224.ckpt"

dl = DownLoad()
# Download the pre-trained model file
dl.download_url(models_url)
```

### MobileNet V2 Model Fine-tuning

This chapter uses MobileNet V2 pretrained model for fine-tuning, and uses the classification dataset of the dog and croissants to retrain the model by deleting the last parameter of the 1x1 convolution layer for classification in the MobileNet V2 pretrained model, to update the model parameter.

```python
import mindspore.nn as nn
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net

from mindvision.classification.models import mobilenet_v2
from mindvision.engine.loss import CrossEntropySmooth

# Build a model with a target classification number of 2 and an image input size of (224,224)
network = mobilenet_v2(num_classes=2, resize=224)

# Save the model parameter in param_dict
param_dict = load_checkpoint("./mobilenet_v2_1.0_224.ckpt")

# Obtain the parameter name of the last convolutional layer of the mobilenet_v2 network
filter_list = [x.name for x in network.head.classifier.get_parameters()]

# Delete the last convolutional layer of the pre-trained model
def filter_ckpt_parameter(origin_dict, param_filter):
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break

filter_ckpt_parameter(param_dict, filter_list)

# Load the pre-trained model parameters as the network initialization weight
load_param_into_net(network, param_dict)

# Define the optimizer
network_opt = nn.Momentum(params=network.trainable_params(), learning_rate=0.01, momentum=0.9)

# Define the loss function
network_loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.1, classes_num=2)

# Define evaluation metrics
metrics = {"Accuracy": nn.Accuracy()}

# Initialize the model
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

> Due to the model fine-tuning, the above WARNING needs to remove the parameters of the last convolutional layer of the pre-trained model, so loading the pre-trained model will show that the `head.classifier` parameter is not loaded. The `head.classifier` parameter will use the initialization value when the model was built.

### Model Training and Evaluation

Train and evaluate the network, and use the `mindvision.engine.callback.ValAccMonitor` interface in MindSpore Vision to print the loss value and the evaluation accuracy of the training. After the training is completed, save the CKPT file with the highest evaluation accuracy, `best.ckpt`, in the current directory.

```python
from mindvision.engine.callback import ValAccMonitor
from mindspore.train.callback import TimeMonitor

num_epochs = 10

# Model training and validation, after the training is completed, save the CKPT file with the highest evaluation accuracy, `best.ckpt`, in the current directory
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

Define the `visualize_model` function, use the model with the highest validation accuracy described above to make predictions about the input images and visualize the predictions.

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

    # Image channel switches (h, w, c) to (c, h, w)
    image = np.transpose(image, (2, 0, 1))

    # Extend the data dimension to (1，c, h, w)
    image = np.expand_dims(image, axis=0)

    # Define and load the network
    net = mobilenet_v2(num_classes=2, resize=224)
    param_dict = load_checkpoint("./best.ckpt")
    load_param_into_net(net, param_dict)
    model = Model(net)

    # Model prediction
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

After the model is trained, the network model (i.e. CKPT file) after the training is completed is converted to MindIR format for subsequent inference on the phone side. The `export` interface generates `mobilenet_v2_1.0_224.mindir` files in the current directory.

```python
from mindspore import export, Tensor

# Define and load the network parameters
net = mobilenet_v2(num_classes=2, resize=224)
param_dict = load_checkpoint("best.ckpt")
load_param_into_net(net, param_dict)

# Export the model from the ckpt format to the MINDIR format
input_np = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)
export(net, Tensor(input_np), file_name="mobilenet_v2_1.0_224", file_format="MINDIR")
```

## Inference and Deployment on the Phone Side

To implement the inference function of the model file on the phone side, the steps are as follows:

- Convert file format: Convert MindIR file format to the MindSpore Lite recognizable file on the Android phone;

- Application deployment: Deploy the app APK on the phone side, that is, download a MindSpore Vision suite Android APK; and

- Application experience: After finally importing the ms model file to the phone side, experience the recognition function of the dog and croissants.

### Converting the file format

Use the [conversion tool](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html) applied on the use side, and convert the mobilenet_v2_1.0_224.mindir file generated during the training process into a file format recognizable by the MindSpore Lite end-side inference framework mobilenet_v2_1.0_224.ms file.

The specific model file format conversion method is as follows:

1. Use MindSpore Lite Converter to convert file formats in the Linux, in the [Linux-x86_64 tool downloading link](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html).

```shell
# Set the path of the package after downloading and extracting, {converter_path}is the path to the extracted toolkit, PACKAGE_ROOT_PATH is set
export PACKAGE_ROOT_PATH={converter_path}

# Include the dynamic-link libraries required by the conversion tool in the environment variables LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/tools/converter/lib:${LD_LIBRARY_PATH}

# Execute the conversion command in mindspore-lite-linux-x64/tools/converter/converter
./converter_lite --fmk=MINDIR --modelFile=mobilenet_v2_1.0_224.mindir  --outputFile=mobilenet_v2_1.0_224
```

2. Use MindSpore Lite Converter under Windows to convert file formats, in the [Windows-x64 tool downloading link](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html)

```shell
# Set the path of the package after downloading and extracting, {converter_path}is the path to the extracted toolkit, PACKAGE_ROOT_PATH is the environment variable that is set
set PACKAGE_ROOT_PATH={converter_path}

# Include the dynamic-link libraries required by the conversion tool in the environment variables PATH
set PATH=%PACKAGE_ROOT_PATH%\tools\converter\lib;%PATH%

# Execute the conversion command in mindspore-lite-win-x64\tools\converter\converter
call converter_lite --fmk=MINDIR --modelFile=mobilenet_v2_1.0_224.mindir --outputFile=mobilenet_v2_1.0_224
```

After the conversion is successful, `CONVERTL RESULT SUCCESS:0` is printed, and the `mobilenet_v2_1.0_224.ms` file is generated in the current directory.

> For other environments to download MindSpore Lite Converter, see [Download MindSpore Lite](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html).

### Application Deployment

Download [Android apps  APK](https://gitee.com/mindspore/vision/releases/) of the MindSpore Vision Suite and install the APK on your phone, whose app name appears as `MindSpore Vision`.

> MindSpore Vision APK is mainly used as an example of a visual development tool, providing basic UI functions such as taking pictures and selecting pictures, and providing AI application DEMO such as classification, detection, and face recognition.

After opening the APP and clicking on the `classification` module on the home page, you can click the middle button to take a picture and get the picture, or click the image button in the upper sidebar to select the picture album for the image classification function.

![main](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/beginner/images/app1.png)

By default, the MindSpore Vision `classification` module has a built-in universal AI network model to identify and classify images.

![result](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/beginner/images/app2.png)

### Application Experience

Finally, the custom network model `mobilenet_v2_1.0_224.ms` trained above is deployed to the Android phone side to experience the recognition function of dog and croissants.

#### Customizing the Model Label Files

Customizing model deployment requires the following format to define the information for the network model, that is, customizing the label files, and creating a json format label file that must be named after `custom.json` on the local computer side.

```text
"title": 'dog and croissants',
"file": 'mobilenet_v2_1.0_224.ms',
"label": ['croissants', 'dag']
```

The Json label file should contain three Key value fields of `title`, `file`, and `label`, the meaning of which is as follows:

- title: customize the module titles (dog and croissants);
- file: the name of the model file converted above; and
- label: `array` information for customizing the label.

#### Labels and Model Files Deployed to the Phone

By pressing the `classification` button on the home page of the `MindSpore Vision APK`, you can enter the customization classification mode and select the tags and model files that need to be deployed.

In order to achieve the recognition function of the mobile phone between dog and croissants, the label file `custom.json` file and the model file `mobilenet_v2_1.0_224.ms` should be placed together in the specified directory on the mobile phone. Here to take the `Android/data/Download/` folder as an example, you need to put the tag file and the model file at the same time in the above mobile phone directory  first, as shown in the figure, then click the customize button, and the system file function will pop up. You can click the open file in the upper left corner, and then find the directory address where the Json tag file and the model file are stored, and select the corresponding Json file.

![step](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/beginner/images/app3.png)

After the label and model file are deployed to the mobile phone, you can click the middle button to take a picture to get the picture, or click the image button in the upper sidebar to select the picture album for the image, and you can classify the dog and the croissants.

![result1](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/beginner/images/app4.png)

> This chapter only covers the simple deployment process on the phone side. For more information about inference, please refer to [MindSpore Lite](https://www.mindspore.cn/lite/docs/en/master/index.html).







