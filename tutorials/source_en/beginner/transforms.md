<a href="https://gitee.com/mindspore/docs/blob/r1.9/tutorials/source_en/beginner/transforms.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png"></a>

[Introduction](https://www.mindspore.cn/tutorials/en/r1.9/beginner/introduction.html) || [Quick Start](https://www.mindspore.cn/tutorials/en/r1.9/beginner/quick_start.html) || [Tensor](https://www.mindspore.cn/tutorials/en/r1.9/beginner/tensor.html) || [Dataset](https://www.mindspore.cn/tutorials/en/r1.9/beginner/dataset.html) || **Transforms** || [Model](https://www.mindspore.cn/tutorials/en/r1.9/beginner/model.html) || [Autograd](https://www.mindspore.cn/tutorials/en/r1.9/beginner/autograd.html) || [Train](https://www.mindspore.cn/tutorials/en/r1.9/beginner/train.html) || [Save and Load](https://www.mindspore.cn/tutorials/en/r1.9/beginner/save_load.html) || [Infer](https://www.mindspore.cn/tutorials/en/r1.9/beginner/infer.html)

# Transforms

Usually, the directly-loaded raw data cannot be directly fed into the neural network for training, and we need to preprocess the data at this time. MindSpore provides different kinds of data transforms that can be used with the Data Processing Pipeline for data preprocessing. All Transforms can be passed in via the `map` method to process the specified data columns.

`mindspore.dataset` provides Transforms for different data types such as image, text and audio, and also supports using Lambda functions. The descriptions are as follows.

```python
import numpy as np
from PIL import Image
from mindvision import dataset
from mindspore.dataset import transforms, vision, text, GeneratorDataset
```

## Common Transforms

The `mindspore.dataset.transforms` module supports a set of common Transforms. Here we take `Compose` as an example to introduce its usage.

### Compose

`Compose` takes a sequence of data enhancement operations and then combines them into a single data enhancement operation. We still present the application effect of Transforms based on the Mnist dataset.

```python
# Download training data from open datasets
training_data = dataset.Mnist(
    path="dataset",
    split="train",
    download=True
)
train_dataset = training_data.dataset
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

For more common Transforms, see [mindspore.dataset.transforms](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore.dataset.transforms.html).

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
array([[220, 139, 217, ..., 118, 227,  14],
       [150, 162,  87, ...,  83, 101,  96],
       [ 40,  53, 130, ...,  30,  97, 136],
       ...,
       [158, 227, 155, ...,  77, 122, 149],
       [102, 106,  40, ...,  60, 130,  88],
       [106, 162, 250, ..., 204,  82,  28]], dtype=uint8)
```

To present a more visual comparison of the data before and after Transform, we use [Eager mode](https://www.mindspore.cn/tutorials/experts/en/r1.9/dataset/eager.html) demo of Transforms. First instantiate the Transform object, and then call the object for data processing.

```python
rescale = vision.Rescale(1.0 / 255.0, 0)
rescaled_image = rescale(random_image)
print(rescaled_image)
```

```text
array([[0.86274517, 0.54509807, 0.85098046, ..., 0.46274513, 0.89019614,
        0.05490196],
       [0.5882353 , 0.63529414, 0.34117648, ..., 0.3254902 , 0.39607847,
        0.37647063],
       [0.15686275, 0.20784315, 0.50980395, ..., 0.11764707, 0.3803922 ,
        0.53333336],
       ...,
       [0.61960787, 0.89019614, 0.60784316, ..., 0.3019608 , 0.4784314 ,
        0.58431375],
       [0.40000004, 0.4156863 , 0.15686275, ..., 0.23529413, 0.50980395,
        0.34509805],
       [0.4156863 , 0.63529414, 0.9803922 , ..., 0.8000001 , 0.32156864,
        0.10980393]], dtype=float32)
```

It can be seen that each pixel value is scaled after using `Rescale`.

### Normalize

The `Normalize` transform is used for normalization of the input image and consists of three parameters:

- mean: the mean value of each channel in the image.
- std: the standard deviation of each channel in the image.
- is_hwc: whether the format of input image is (height, width, channel) or (channel, height, width).

Each channel of the image will be adjusted according to `mean` and `std`, and the formula is $output_{c} = (input_{c} - \frac{mean_{c}}{std_{c}})$, where $c$ represents the channel index.

```python
normalize = vision.Normalize(mean=(0.1307,), std=(0.3081,))
normalized_image = normalize(rescaled_image)
print(normalized_image)
```

```text
array([[ 2.3759987 ,  1.3450117 ,  2.337814  , ...,  1.0777187 ,
         2.4650965 , -0.24601768],
       [ 1.4850222 ,  1.637761  ,  0.6831434 , ...,  0.63223046,
         0.8613388 ,  0.7976976 ],
       [ 0.08491641,  0.2503835 ,  1.2304575 , ..., -0.04236592,
         0.8104258 ,  1.306827  ],
       ...,
       [ 1.5868481 ,  2.4650965 ,  1.5486634 , ...,  0.55586106,
         1.1286317 ,  1.472294  ],
       [ 0.87406707,  0.92498   ,  0.08491641, ...,  0.33948112,
         1.2304575 ,  0.69587165],
       [ 0.92498   ,  1.637761  ,  2.7578456 , ...,  2.172347  ,
         0.61950225, -0.06782239]], dtype=float32)
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
((48, 48, 1), (1, 48, 48))
```

For more Vision Transforms, see [mindspore.dataset.vision](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore.dataset.vision.html).

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

```text
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
{'Beijing': 2,
 '欢': 0,
 '喜': 8,
 'Welcome': 4,
 'China': 3,
 'to': 5,
 '京': 6,
 '北': 7,
 '您': 9,
 '我': 10,
 '!': 1,
 '迎': 11,
 '！': 12}
```

After generating the vocabulary, you can perform the vocabulary mapping transformation with the `map` method to convert Token to Index.

```python
test_dataset = test_dataset.map(text.Lookup(vocab))
print(next(test_dataset.create_tuple_iterator()))
```

```text
[Tensor(shape=[6], dtype=Int32, value= [ 7,  6,  0, 11,  9, 12])]
```

For more Text Transforms, see [mindspore.dataset.text](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore.dataset.text.html).

## Lambda Transforms

Lambda functions are anonymous functions that do not require a name and consist of a single expression that is evaluated when called. Lambda Transforms can load arbitrarily-defined Lambda functions, providing enough flexibility. Here, we start with a simple Lambda function that multiplies the input data by 2:

```python
test_dataset = GeneratorDataset([1, 2, 3], 'data', shuffle=False)
test_dataset = test_dataset.map(lambda x: x * 2)
print(list(test_dataset.create_tuple_iterator()))
```

```text
[[Tensor(shape=[], dtype=Int64, value= 2)],
 [Tensor(shape=[], dtype=Int64, value= 4)],
 [Tensor(shape=[], dtype=Int64, value= 6)]]
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
[[Tensor(shape=[], dtype=Int64, value= 6)],
 [Tensor(shape=[], dtype=Int64, value= 18)],
 [Tensor(shape=[], dtype=Int64, value= 38)]]
```
