<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/transforms.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

[Introduction](https://www.mindspore.cn/tutorials/en/master/beginner/introduction.html) || [Quick Start](https://www.mindspore.cn/tutorials/en/master/beginner/quick_start.html) || [Tensor](https://www.mindspore.cn/tutorials/en/master/beginner/tensor.html) || [Dataset](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html) || **Transforms** || [Model](https://www.mindspore.cn/tutorials/en/master/beginner/model.html) || [Autograd](https://www.mindspore.cn/tutorials/en/master/beginner/autograd.html) || [Train](https://www.mindspore.cn/tutorials/en/master/beginner/train.html) || [Save and Load](https://www.mindspore.cn/tutorials/en/master/beginner/save_load.html)

# Transforms

Usually, the directly-loaded raw data cannot be directly fed into the neural network for training, and we need to preprocess the data at this time. MindSpore provides different kinds of data transforms that can be used with the Data Processing Pipeline for data preprocessing. All Transforms can be passed in via the `map` method to process the specified data columns.

`mindspore.dataset` provides Transforms for different data types such as image, text and audio, and also supports using Lambda functions. The descriptions are as follows.

```python
import numpy as np
from PIL import Image
from download import download
from mindspore.dataset import transforms, vision, text
from mindspore.dataset import GeneratorDataset, MnistDataset
```

## Common Transforms

The `mindspore.dataset.transforms` module supports a set of common Transforms. Here we take `Compose` as an example to introduce its usage.

### Compose

`Compose` takes a sequence of data enhancement operations and then combines them into a single data enhancement operation. We still present the application effect of Transforms based on the Mnist dataset.

```python
# Download data from open datasets

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)

train_dataset = MnistDataset('MNIST_Data/train')
```

```text
Downloading data from https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip (10.3 MB)

file_sizes: 100%|██████████████████████████| 10.8M/10.8M [00:01<00:00, 5.57MB/s]
Extracting zip file...
Successfully downloaded / unzipped to ./
```

```python
image, label = next(train_dataset.create_tuple_iterator())
print(image.shape)
```

```text
(28, 28, 1)
```

```python
composed = transforms.Compose(
    [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
)
```

```python
train_dataset = train_dataset.map(composed, 'image')
image, label = next(train_dataset.create_tuple_iterator())
print(image.shape)
```

```text
(1, 28, 28)
```

For more common Transforms, see [mindspore.dataset.transforms](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.transforms.html).

## Vision Transforms

The `mindspore.dataset.vision` module provides a series of Transforms for image data. `Rescale`, `Normalize` and `HWC2CHW` transforms are used in the Mnist data processing. The descriptions are as follows.

### Rescale

The `Rescale` transform is used to resize the image pixel values and consists of two parameters:

- rescale: scaling factor.
- shift: shift factor.

Each pixel of the image will be adjusted according to these two parameters and the output pixel value will be $output_{i} = input_{i} * rescale + shift$.

Here we first use numpy to generate a random image with pixel values in \[0, 255\] and scale its pixel values.

```python
random_np = np.random.randint(0, 255, (48, 48), np.uint8)
random_image = Image.fromarray(random_np)
print(random_np)
```

```text
[[ 59  38 206 ... 126 244 226]
 [ 27 113 135 ... 248   3   0]
 [106  13 154 ... 149   7 126]
 ...
 [142 135 222 ... 253  58 228]
 [110 239 114 ...  75 142  65]
 [  0 108 141 ... 145 159  11]]
```

To present a more visual comparison of the data before and after Transform, we use [Eager mode](https://mindspore.cn/tutorials/en/master/advanced/dataset/eager.html) demo of Transforms. First instantiate the Transform object, and then call the object for data processing.

```python
rescale = vision.Rescale(1.0 / 255.0, 0)
rescaled_image = rescale(random_image)
print(rescaled_image)
```

```text
[[0.23137257 0.14901961 0.8078432  ... 0.49411768 0.9568628  0.8862746 ]
 [0.10588236 0.4431373  0.5294118  ... 0.9725491  0.01176471 0.        ]
 [0.4156863  0.0509804  0.6039216  ... 0.58431375 0.02745098 0.49411768]
 ...
 [0.5568628  0.5294118  0.8705883  ... 0.9921569  0.227451   0.8941177 ]
 [0.43137258 0.93725497 0.44705886 ... 0.29411766 0.5568628  0.25490198]
 [0.         0.42352945 0.5529412  ... 0.5686275  0.62352943 0.04313726]]
```

It can be seen that each pixel value is scaled after using `Rescale`.

### Normalize

The `Normalize` transform is used for normalization of the input image and consists of three parameters:

- mean: the mean value of each channel in the image.
- std: the standard deviation of each channel in the image.
- is_hwc: whether the format of input image is (height, width, channel) or (channel, height, width).

Each channel of the image will be adjusted according to `mean` and `std`, and the formula is $output_{c} = \frac{input_{c} - mean_{c}}{std_{c}}$, where $c$ represents the channel index.

```python
normalize = vision.Normalize(mean=(0.1307,), std=(0.3081,))
normalized_image = normalize(rescaled_image)
print(normalized_image)
```

```text
[[ 0.32675287  0.05945994  2.1978035  ...  1.1795447   2.6814764
   2.452368  ]\n",
 [-0.08055063  1.0140777   1.2940987  ...  2.7323892  -0.38602826
  -0.42421296]\n",
 [ 0.92498    -0.2587459   1.5359352  ...  1.472294   -0.33511534
   1.1795447 ]\n",
 ...\n",
 [ 1.3831964   1.2940987   2.4014552  ...  2.7960305   0.31402466
   2.4778247 ]\n",
 [ 0.9758929   2.617835    1.0268059  ...  0.5304046   1.3831964
   0.40312228]\n",
 [-0.42421296  0.9504364   1.3704681  ...  1.4213811   1.5995764
  -0.2842024 ]]
```

### HWC2CWH

The `HWC2CWH` transform is used to convert the image format. The two different formats (height, width, channel) or (channel, height, width) may be targeted and optimized in different hardware devices. MindSpore sets HWC as the default image format and uses this transform for processing when CWH format is required.

Here we first process the `normalized_image` in the previous section to HWC format, and then convert it. You can see the change of the shape before and after the conversion.

```python
hwc_image = np.expand_dims(normalized_image, -1)
hwc2cwh = vision.HWC2CHW()
chw_image = hwc2cwh(hwc_image)
print(hwc_image.shape, chw_image.shape)
```

```text
(48, 48, 1) (1, 48, 48)
```

For more Vision Transforms, see [mindspore.dataset.vision](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.vision.html).

## Text Transforms

The `mindspore.dataset.text` module provides a series of Transforms for text data. Unlike image data, text data requires operations such as Tokenize, building word lists, and Token to Index. Here is a brief description of its usage.

First we define three pieces of text as the data to be processed and load them by using `GeneratorDataset`.

```python
texts = [
    'Welcome to Beijing',
    '北京欢迎您！',
    '我喜欢China!',
]
```

```python
test_dataset = GeneratorDataset(texts, 'text')
```

### BasicTokenizer

Tokenize is a basic method to process text data. MindSpore provides many different Tokenizers. Here we choose the basic `BasicTokenizer` as an example. Here we choose the basic `BasicTokenizer` as an example. Together with `map`, we perform Tokenize on the three pieces of text and it can be seen that Tokenize is successfully performed on the processed data.

```python
test_dataset = test_dataset.map(text.BasicTokenizer())
print(next(test_dataset.create_tuple_iterator()))
```

```text
[Tensor(shape=[5], dtype=String, value= ['我', '喜', '欢', 'China', '!'])]
```

### Lookup

`Lookup` is a vocabulary mapping transformation used to convert Token to Index. Before using `Lookup`, you need to construct a vocabulary, either by loading an existing vocabulary or by using `Vocab` to generate a vocabulary. Here we choose to use `Vocab.from_dataset` method to generate a vocabulary from a dataset.

```python
vocab = text.Vocab.from_dataset(test_dataset)
```

After obtaining the vocabulary, we can use the `vocab` method to view the vocabulary.

```python
print(vocab.vocab())
```

```text
{'迎': 11, '我': 10, '您': 9, '京': 6, 'to': 5, '！': 12, '喜': 8, 'Welcome': 4, 'China': 3, '北': 7, 'Beijing': 2, '!': 1, '欢': 0}
```

After generating the vocabulary, you can perform the vocabulary mapping transformation with the `map` method to convert Token to Index.

```python
test_dataset = test_dataset.map(text.Lookup(vocab))
print(next(test_dataset.create_tuple_iterator()))
```

```text
[Tensor(shape=[3], dtype=Int32, value= [4, 5, 2])]
```

For more Text Transforms, see [mindspore.dataset.text](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.text.html).

## Lambda Transforms

Lambda functions are anonymous functions that do not require a name and consist of a single expression that is evaluated when called. Lambda Transforms can load arbitrarily-defined Lambda functions, providing enough flexibility. Here, we start with a simple Lambda function that multiplies the input data by 2:

```python
test_dataset = GeneratorDataset([1, 2, 3], 'data', shuffle=False)
test_dataset = test_dataset.map(lambda x: x * 2)
print(list(test_dataset.create_tuple_iterator()))
```

```text
[[Tensor(shape=[], dtype=Int64, value= 2)], [Tensor(shape=[], dtype=Int64, value= 4)], [Tensor(shape=[], dtype=Int64, value= 6)]]
```

You can see that after `map` is passed into the Lambda function, the data is iteratively obtained for the multiply-2 operation.

We can also define more complex functions that work with the Lambda function to achieve complex data processing:

```python
def func(x):
    return x * x + 2

test_dataset = test_dataset.map(lambda x: func(x))
```

```python
print(list(test_dataset.create_tuple_iterator()))
```

```text
[[Tensor(shape=[], dtype=Int64, value= 6)], [Tensor(shape=[], dtype=Int64, value= 18)], [Tensor(shape=[], dtype=Int64, value= 38)]]
```
