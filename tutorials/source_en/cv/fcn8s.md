[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_en/cv/fcn8s.md)

# FCN for Image Semantic Segmentation

Fully convolutional network (FCN) is a framework for image semantic segmentation proposed by Jonathan Long et al. of UC Berkeley in 2015 in Fully Convolutional Networks for Semantic Segmentation<sup>[1]</sup>.

FCN is the first end-to-end network for pixel-level prediction.

![fcn-1](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/tutorials/source_zh_cn/cv/images/fcn_1.png)

## Semantic Segmentation

Before the FCN is specifically described, semantic segmentation is first introduced.

Semantic segmentation is an important part of image understanding in image processing and machine vision technologies. It is an important branch in the AI field and is often applied to fields such as face recognition, object detection, medical imaging, satellite image analysis, and autonomous driving perception.

A purpose of semantic segmentation is to classify each pixel in an image. Different from a common classification task that outputs only a class, the semantic segmentation task outputs an image whose size is the same as that of the input, and each pixel of the output image corresponds to a class of each pixel of the input image. In the image field, semantics refers to the content of an image. The following figure shows some semantic segmentation instances.

![fcn-2](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/tutorials/source_zh_cn/cv/images/fcn_2.png)

## Model Introduction

FCN is mainly used in the image segmentation field. It is an end-to-end segmentation method of deep learning in image semantic segmentation. Pixel-level prediction is performed to obtain the label map whose size is the same as that of the original image. FCN replaces the fully-connected layers with fully convolutional layers, all layers of the network are convolutional layers, and therefore the network is referred to as a fully convolutional network.

The fully convolutional neural network mainly uses the following three technologies:

1. Convolutional

    VGG-16 is used as the FCN backbone. The input of VGG-16 is a 224 x 224 RGB image, and the output is 1000 prediction values. VGG-16 accepts only fixed-size input, discards spatial coordinates, and generates non-spatial output. There are three fully-connected layers in total in the VGG-16, and the fully-connected layers may also be considered as convolutions covering an entire region. Converting a fully-connected layer into a convolutional layer can change a network output from a one-dimensional non-spatial output to a two-dimensional matrix, and generate a heatmap mapped to an input image by using the output.

   ![fcn-3](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/tutorials/source_zh_cn/cv/images/fcn_3.png)

2. Upsampling

    A convolution operation and a pooling operation in a convolution process reduce the size of a feature map. To obtain dense image prediction of the size of an original image, an upsampling operation needs to be performed on the obtained feature map. The parameters of bilinear interpolation are used to initialize the parameters of upsampling inverse convolution, and then the nonlinear upsampling is learned through backpropagation. Upsampling is performed in the network for end-to-end learning through backpropagation of pixel loss.

    ![fcn-4](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/tutorials/source_zh_cn/cv/images/fcn_4.png)

3. Skip layer

    Upsampling is performed on the feature map of the last layer to obtain the segmentation of the original image size. The segmentation is a prediction with a step of 32 pixels, which is called FCN-32s. Because the feature map at the last layer is too small and too many details are lost, the skips structure is used to combine the prediction at the last layer with the prediction at the shallower layer. Then, the prediction result can obtain more local details. The 2x upsampling is performed on the prediction (FCN-32s) of the bottom layer (stride 32) to obtain an image of the original size, and the image is fused (added) with the prediction performed from the pool 4 layer (stride 16). This part of network is called FCN-16s. Then, the 2x upsampling is performed on this part of prediction again and fused with the prediction obtained from the pool 3 layer. This part of network is called FCN-8s. The skips structure combines deep global information with shallow local information.

    ![fcn-5](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/tutorials/source_zh_cn/cv/images/fcn_5.png)

## Network Features

1. A fully convolutional network without a fully-connected layer can adapt to input of any size.
2. A deconv layer with an increased data size can output a refined result.
3. The skip structure combines the results of different depth layers to ensure robustness and accuracy.

## Data Processing

Before the experiment, ensure that the Python environment and MindSpore have been installed on the local PC.

```python
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/dataset_fcn8s.tar"

download(url, "./dataset", kind="tar", replace=True)
```

### Data Preprocessing

Most images in the PASCAL VOC 2012 dataset have different resolutions and cannot be placed in the same tensor. Therefore, standardization is required before input.

### Data Loading

Mix the PASCAL VOC 2012 dataset with the SDB dataset.

```python
import numpy as np
import cv2
import mindspore.dataset as ds

class SegDataset:
    def __init__(self,
                 image_mean,
                 image_std,
                 data_file='',
                 batch_size=32,
                 crop_size=512,
                 max_scale=2.0,
                 min_scale=0.5,
                 ignore_label=255,
                 num_classes=21,
                 num_readers=2,
                 num_parallel_calls=4):

        self.data_file = data_file
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.num_readers = num_readers
        self.num_parallel_calls = num_parallel_calls
        max_scale > min_scale

    def preprocess_dataset(self, image, label):
        image_out = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        label_out = cv2.imdecode(np.frombuffer(label, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        sc = np.random.uniform(self.min_scale, self.max_scale)
        new_h, new_w = int(sc * image_out.shape[0]), int(sc * image_out.shape[1])
        image_out = cv2.resize(image_out, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        label_out = cv2.resize(label_out, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        image_out = (image_out - self.image_mean) / self.image_std
        out_h, out_w = max(new_h, self.crop_size), max(new_w, self.crop_size)
        pad_h, pad_w = out_h - new_h, out_w - new_w
        if pad_h > 0 or pad_w > 0:
            image_out = cv2.copyMakeBorder(image_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            label_out = cv2.copyMakeBorder(label_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_label)
        offset_h = np.random.randint(0, out_h - self.crop_size + 1)
        offset_w = np.random.randint(0, out_w - self.crop_size + 1)
        image_out = image_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size, :]
        label_out = label_out[offset_h: offset_h + self.crop_size, offset_w: offset_w+self.crop_size]
        if np.random.uniform(0.0, 1.0) > 0.5:
            image_out = image_out[:, ::-1, :]
            label_out = label_out[:, ::-1]
        image_out = image_out.transpose((2, 0, 1))
        image_out = image_out.copy()
        label_out = label_out.copy()
        label_out = label_out.astype("int32")
        return image_out, label_out

    def get_dataset(self):
        ds.config.set_numa_enable(True)
        dataset = ds.MindDataset(self.data_file, columns_list=["data", "label"],
                                 shuffle=True, num_parallel_workers=self.num_readers)
        transforms_list = self.preprocess_dataset
        dataset = dataset.map(operations=transforms_list, input_columns=["data", "label"],
                              output_columns=["data", "label"],
                              num_parallel_workers=self.num_parallel_calls)
        dataset = dataset.shuffle(buffer_size=self.batch_size * 10)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset


# Define parameters for creating a dataset.
IMAGE_MEAN = [103.53, 116.28, 123.675]
IMAGE_STD = [57.375, 57.120, 58.395]
DATA_FILE = "dataset/dataset_fcn8s/mindname.mindrecord"

# Define model training parameters.
train_batch_size = 4
crop_size = 512
min_scale = 0.5
max_scale = 2.0
ignore_label = 255
num_classes = 21

# Instantiate a dataset.
dataset = SegDataset(image_mean=IMAGE_MEAN,
                     image_std=IMAGE_STD,
                     data_file=DATA_FILE,
                     batch_size=train_batch_size,
                     crop_size=crop_size,
                     max_scale=max_scale,
                     min_scale=min_scale,
                     ignore_label=ignore_label,
                     num_classes=num_classes,
                     num_readers=2,
                     num_parallel_calls=4)

dataset = dataset.get_dataset()
```

### Training Set Visualization

Run the following code to observe the loaded dataset image (normalization has been performed during data processing):

```python
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8))

# Display data in the training set.
for i in range(1, 9):
    plt.subplot(2, 4, i)
    show_data = next(dataset.create_dict_iterator())
    show_images = show_data["data"].asnumpy()
    show_images = np.clip(show_images, 0, 1)
# Convert the image to the HWC format and display it.
    plt.imshow(show_images[0].transpose(1, 2, 0))
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05, hspace=0)
plt.show()
```

## Building a Network

### Network Process

The following figure shows the FCN process.

1. After pool 1, the size of the input image changes to 1/2 of the original size.
2. After pool 2, the size changes to 1/4 of the original size.
3. After pool 3, pool 4, and pool 5, their sizes change to 1/8, 1/16, and 1/32 of the original sizes respectively.
4. After conv 6 and conv 7, the output size is still 1/32 of the original image.
5. FCN-32s is the last to use deconvolution so that the size of the output image is the same as that of the input image.
6. FCN-16s deconvolutes the output of conv 7 to double the size of the original image to 1/16 of the original image, fuses the output with the feature map output by pool 4, and then expands the output to the original size through deconvolution.
7. FCN-8s deconvolutes the output of conv 7 to increase the size by four times, deconvolutes the feature map output by pool 4 to increase the size by two times, and takes out the feature map output by pool 3. After the three are fused, the size is increased to the original size through deconvolution.

![fcn-6](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/tutorials/source_zh_cn/cv/images/fcn_6.png)

Use the following code to build an FCN-8s network.

```python
import mindspore.nn as nn

class FCN8s(nn.Cell):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1 = nn.SequentialCell(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.SequentialCell(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.SequentialCell(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.SequentialCell(
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, weight_init='xavier_uniform'),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = nn.SequentialCell(
            nn.Conv2d(in_channels=512, out_channels=4096,
                      kernel_size=7, weight_init='xavier_uniform'),
            nn.BatchNorm2d(4096),
            nn.ReLU(),
        )
        self.conv7 = nn.SequentialCell(
            nn.Conv2d(in_channels=4096, out_channels=4096,
                      kernel_size=1, weight_init='xavier_uniform'),
            nn.BatchNorm2d(4096),
            nn.ReLU(),
        )
        self.score_fr = nn.Conv2d(in_channels=4096, out_channels=self.n_class,
                                  kernel_size=1, weight_init='xavier_uniform')
        self.upscore2 = nn.Conv2dTranspose(in_channels=self.n_class, out_channels=self.n_class,
                                           kernel_size=4, stride=2, weight_init='xavier_uniform')
        self.score_pool4 = nn.Conv2d(in_channels=512, out_channels=self.n_class,
                                     kernel_size=1, weight_init='xavier_uniform')
        self.upscore_pool4 = nn.Conv2dTranspose(in_channels=self.n_class, out_channels=self.n_class,
                                                kernel_size=4, stride=2, weight_init='xavier_uniform')
        self.score_pool3 = nn.Conv2d(in_channels=256, out_channels=self.n_class,
                                     kernel_size=1, weight_init='xavier_uniform')
        self.upscore8 = nn.Conv2dTranspose(in_channels=self.n_class, out_channels=self.n_class,
                                           kernel_size=16, stride=8, weight_init='xavier_uniform')

    def construct(self, x):
        x1 = self.conv1(x)
        p1 = self.pool1(x1)
        x2 = self.conv2(p1)
        p2 = self.pool2(x2)
        x3 = self.conv3(p2)
        p3 = self.pool3(x3)
        x4 = self.conv4(p3)
        p4 = self.pool4(x4)
        x5 = self.conv5(p4)
        p5 = self.pool5(x5)
        x6 = self.conv6(p5)
        x7 = self.conv7(x6)
        sf = self.score_fr(x7)
        u2 = self.upscore2(sf)
        s4 = self.score_pool4(p4)
        f4 = s4 + u2
        u4 = self.upscore_pool4(f4)
        s3 = self.score_pool3(p3)
        f3 = s3 + u4
        out = self.upscore8(f3)
        return out
```

## Training Preparation

### Importing VGG-16 Partial Pre-trained Weights

FCN uses VGG-16 as the backbone network for image encoding. Use the following code to import some pre-traind weights of the VGG-16 pre-trained model.

```python
from download import download
from mindspore import load_checkpoint, load_param_into_net

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/fcn8s_vgg16_pretrain.ckpt"
download(url, "fcn8s_vgg16_pretrain.ckpt", replace=True)
def load_vgg16():
    ckpt_vgg16 = "fcn8s_vgg16_pretrain.ckpt"
    param_vgg = load_checkpoint(ckpt_vgg16)
    load_param_into_net(net, param_vgg)
```

### Loss Function

Semantic segmentation is to classify each pixel in an image, which is still a classification problem. Therefore, the cross-entropy loss function is selected to calculate the cross-entropy loss between the FCN output and the mask. Here we use mindspore.nn.CrossEntropyLoss() as the loss function.

### Customized Evaluation Metrics

This part evaluates the effect of the trained model. For ease of explanation, assume that there are $k+1$ classes (from $L_0$ to $L_k$, including an empty class or background), $p_{i j}$ indicates the number of pixels that belong to the $i$ class but are predicted to be the $j$ class. That is, $p_{i i}$ represents the true quantity, while $p_{i j} p_{j i}$ is interpreted as false positive and false negative, respectively, although both are the sum of false positive and false negative.

- Pixel accuracy (PA): This is the simplest measure of the ratio of correctly marked pixels to the total pixels.

$$ P A=\frac{\sum_{i=0}^k p_{i i}}{\sum_{i=0}^k \sum_{j=0}^k p_{i j}} $$

- Mean pixel accuracy (MPA): It is a simple improvement of PA. It calculates the ratio of correctly classified pixels in each class and then calculates the average value of all classes.

$$ M P A=\frac{1}{k+1} \sum_{i=0}^k \frac{p_{i i}}{\sum_{j=0}^k p_{i j}} $$

- Mean intersection over union (MloU): a standard measure of semantic segmentation. It calculates an intersection set and a union set of two sets. In a semantic segmentation problem, the two sets are real values (ground truth) and prediction values (predicted segmentation). This ratio can be transformed into a positive true value (intersection) to the sum of true positive, false negative, and false positive values (union). loU is calculated on each class and then averaged.

$$ M I o U=\frac{1}{k+1} \sum_{i=0}^k \frac{p_{i i}}{\sum_{j=0}^k p_{i j}+\sum_{j=0}^k p_{j i}-p_{i i}} $$

- Frequency weighted intersection over union (FWIoU): It is an improvement of the MloU. In this method, the weight of each class is set according to the occurrence frequency of the class.

$$ F W I o U=\frac{1}{\sum_{i=0}^k \sum_{j=0}^k p_{i j}} \sum_{i=0}^k \frac{p_{i i}}{\sum_{j=0}^k p_{i j}+\sum_{j=0}^k p_{j i}-p_{i i}} $$

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.train as train

class PixelAccuracy(train.Metric):
    def __init__(self, num_class=21):
        super(PixelAccuracy, self).__init__()
        self.num_class = num_class

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def clear(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def update(self, *inputs):
        y_pred = inputs[0].asnumpy().argmax(axis=1)
        y = inputs[1].asnumpy().reshape(4, 512, 512)
        self.confusion_matrix += self._generate_matrix(y, y_pred)

    def eval(self):
        pixel_accuracy = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return pixel_accuracy


class PixelAccuracyClass(train.Metric):
    def __init__(self, num_class=21):
        super(PixelAccuracyClass, self).__init__()
        self.num_class = num_class

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def update(self, *inputs):
        y_pred = inputs[0].asnumpy().argmax(axis=1)
        y = inputs[1].asnumpy().reshape(4, 512, 512)
        self.confusion_matrix += self._generate_matrix(y, y_pred)

    def clear(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def eval(self):
        mean_pixel_accuracy = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        mean_pixel_accuracy = np.nanmean(mean_pixel_accuracy)
        return mean_pixel_accuracy


class MeanIntersectionOverUnion(train.Metric):
    def __init__(self, num_class=21):
        super(MeanIntersectionOverUnion, self).__init__()
        self.num_class = num_class

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def update(self, *inputs):
        y_pred = inputs[0].asnumpy().argmax(axis=1)
        y = inputs[1].asnumpy().reshape(4, 512, 512)
        self.confusion_matrix += self._generate_matrix(y, y_pred)

    def clear(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def eval(self):
        mean_iou = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        mean_iou = np.nanmean(mean_iou)
        return mean_iou


class FrequencyWeightedIntersectionOverUnion(train.Metric):
    def __init__(self, num_class=21):
        super(FrequencyWeightedIntersectionOverUnion, self).__init__()
        self.num_class = num_class

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def update(self, *inputs):
        y_pred = inputs[0].asnumpy().argmax(axis=1)
        y = inputs[1].asnumpy().reshape(4, 512, 512)
        self.confusion_matrix += self._generate_matrix(y, y_pred)

    def clear(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def eval(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        frequency_weighted_iou = (freq[freq > 0] * iu[freq > 0]).sum()
        return frequency_weighted_iou
```

## Model Training

After the VGG-16 pre-trained parameters are imported, instantiate the loss function and optimizer, use the Model interface to compile the network, and train the FCN-8s network.

```python
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.train import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Model

train_batch_size = 4
num_classes = 21
# Initialize the model structure.
net = FCN8s(n_class=21)
# Import VGG-16 pre-trained parameters.
load_vgg16()
# Calculate the learning rate.
min_lr = 0.0005
base_lr = 0.05
train_epochs = 1
iters_per_epoch = dataset.get_dataset_size()
total_step = iters_per_epoch * train_epochs

lr_scheduler = mindspore.nn.cosine_decay_lr(min_lr,
                                            base_lr,
                                            total_step,
                                            iters_per_epoch,
                                            decay_epoch=2)
lr = Tensor(lr_scheduler[-1])

# Define the loss function.
loss = nn.CrossEntropyLoss(ignore_index=255)
# Define the optimizer.
optimizer = nn.Momentum(params=net.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=0.0001)
# Define loss_scale.
scale_factor = 4
scale_window = 3000
loss_scale_manager = ms.amp.DynamicLossScaleManager(scale_factor, scale_window)
# Initialize the model.
model = Model(net, loss_fn=loss, optimizer=optimizer, loss_scale_manager=loss_scale_manager, metrics={"pixel accuracy": PixelAccuracy(), "mean pixel accuracy": PixelAccuracyClass(), "mean IoU": MeanIntersectionOverUnion(), "frequency weighted IoU": FrequencyWeightedIntersectionOverUnion()})

# Set the parameters for saving the CKPT file.
time_callback = TimeMonitor(data_size=iters_per_epoch)
loss_callback = LossMonitor()
callbacks = [time_callback, loss_callback]
save_steps = 330
keep_checkpoint_max = 5
config_ckpt = CheckpointConfig(save_checkpoint_steps=10,
                               keep_checkpoint_max=keep_checkpoint_max)
ckpt_callback = ModelCheckpoint(prefix="FCN8s",
                                directory="./ckpt",
                                config=config_ckpt)
callbacks.append(ckpt_callback)
model.train(train_epochs, dataset, callbacks=callbacks)
```

```text
epoch: 1 step: 1, loss is 3.0904884338378906
epoch: 1 step: 2, loss is 3.0506458282470703
epoch: 1 step: 3, loss is 2.9796371459960938
...                                        ...
epoch: 1 step: 1141, loss is 1.3178229331970215
epoch: 1 step: 1142, loss is 1.307207703590393
epoch: 1 step: 1143, loss is 1.1257227659225464
Train epoch time: 637526.089 ms, per step time: 557.766 ms
```

> The FCN requires a large amount of training data and training epochs during training. Here, only the training of a single epoch with small data is provided to demonstrate the loss convergence process. The following uses the trained weight file to validate the model and display the inference effect.

## Model Evaluation

```python
IMAGE_MEAN = [103.53, 116.28, 123.675]
IMAGE_STD = [57.375, 57.120, 58.395]
DATA_FILE = "dataset/dataset_fcn8s/mindname.mindrecord"

# Download the trained weight file.
url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/FCN8s.ckpt"
download(url, "FCN8s.ckpt", replace=True)
net = FCN8s(n_class=num_classes)

ckpt_file = "FCN8s.ckpt"
param_dict = load_checkpoint(ckpt_file)
load_param_into_net(net, param_dict)

model = Model(net, loss_fn=loss, optimizer=optimizer, loss_scale_manager=loss_scale_manager, metrics={"pixel accuracy": PixelAccuracy(), "mean pixel accuracy": PixelAccuracyClass(), "mean IoU": MeanIntersectionOverUnion(), "frequency weighted IoU": FrequencyWeightedIntersectionOverUnion()})

# Instantiate a dataset.
dataset = SegDataset(image_mean=IMAGE_MEAN,
                     image_std=IMAGE_STD,
                     data_file=DATA_FILE,
                     batch_size=train_batch_size,
                     crop_size=crop_size,
                     max_scale=max_scale,
                     min_scale=min_scale,
                     ignore_label=ignore_label,
                     num_classes=num_classes,
                     num_readers=2,
                     num_parallel_calls=4)
dataset_eval = dataset.get_dataset()
model.eval(dataset_eval)
```

```text
{'pixel accuracy': 0.9734971889839162,
 'mean pixel accuracy': 0.9417477500248013,
 'mean IoU': 0.8956981094249392,
 'frequency weighted IoU': 0.9488965587686847}
```

## Model Inference

Use the trained network to display the model inference result.

```python
import cv2
import matplotlib.pyplot as plt

net = FCN8s(n_class=num_classes)
# Set hyperparameters.
ckpt_file = "FCN8s.ckpt"
param_dict = load_checkpoint(ckpt_file)
load_param_into_net(net, param_dict)
eval_batch_size = 4
img_lst = []
mask_lst = []
res_lst = []
# Inference effect display (The upper part is the input image, and the lower part is the inference effect image.)
plt.figure(figsize=(8, 5))
show_data = next(dataset_eval.create_dict_iterator())
show_images = show_data["data"].asnumpy()
mask_images = show_data["label"].reshape([4, 512, 512])
show_images = np.clip(show_images, 0, 1)
for i in range(eval_batch_size):
    img_lst.append(show_images[i])
    mask_lst.append(mask_images[i])
res = net(show_data["data"]).asnumpy().argmax(axis=1)
for i in range(eval_batch_size):
    plt.subplot(2, 4, i + 1)
    plt.imshow(img_lst[i].transpose(1, 2, 0))
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05, hspace=0.02)
    plt.subplot(2, 4, i + 5)
    plt.imshow(res[i])
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05, hspace=0.02)
plt.show()

```

## Summary

The core contribution of FCN is to use full convolutional layer to implement end-to-end image segmentation through learning. Compared with the traditional image segmentation method using CNN, FCN has two obvious advantages. First, it can accept input images of any size without requiring all training images and test images to have fixed sizes. Second, it is more efficient, and a problem of repeated storage and convolution calculation caused by using a pixel block is avoided.

However, the FCN needs to be improved.

First, the results are still not refined enough. Although the effect of 8x upsampling is much better than that of 32x upsampling, the upsampling result is still blurry and smooth, especially at the boundary. The network is insensitive to the details in the image.
Second, the pixel classification does not fully consider the relationship between pixels (such as discontinuity and similarity). The spatial regularization step used in the common pixel classification-based segmentation method is ignored, and there is a lack of spatial consistency.

## Reference

[1]Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for Semantic Segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.
