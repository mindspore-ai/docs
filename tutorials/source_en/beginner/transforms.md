[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.1/tutorials/source_en/beginner/transforms.md)

[Introduction](https://www.mindspore.cn/tutorials/en/r2.1/beginner/introduction.html) || [Quick Start](https://www.mindspore.cn/tutorials/en/r2.1/beginner/quick_start.html) || [Tensor](https://www.mindspore.cn/tutorials/en/r2.1/beginner/tensor.html) || [Dataset](https://www.mindspore.cn/tutorials/en/r2.1/beginner/dataset.html) || **Transforms** || [Model](https://www.mindspore.cn/tutorials/en/r2.1/beginner/model.html) || [Autograd](https://www.mindspore.cn/tutorials/en/r2.1/beginner/autograd.html) || [Train](https://www.mindspore.cn/tutorials/en/r2.1/beginner/train.html) || [Save and Load](https://www.mindspore.cn/tutorials/en/r2.1/beginner/save_load.html)

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

file_sizes: 100%|██████████████████████████| 10.8M/10.8M [00:01<00:00, 9.01MB/s]
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

For more common Transforms, see [mindspore.dataset.transforms](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore.dataset.transforms.html).

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
[[170  10 218 ...  81 128  96]
 [  2 107 146 ... 239 178 165]
 [232 137 235 ... 222 109 216]
 ...
 [193 140  60 ...  72 133 144]
 [232 175  58 ...  55 110  94]
 [152 241 105 ... 187  45  43]]
```

To present a more visual comparison of the data before and after Transform, we use [Eager mode](https://mindspore.cn/tutorials/en/r2.1/advanced/dataset/eager.html) demo of Transforms. First instantiate the Transform object, and then call the object for data processing.

```python
rescale = vision.Rescale(1.0 / 255.0, 0)
rescaled_image = rescale(random_image)
print(rescaled_image)
```

```text
[[0.6666667  0.03921569 0.854902   ... 0.31764707 0.5019608  0.37647063]
 [0.00784314 0.41960788 0.57254905 ... 0.93725497 0.69803923 0.64705884]
 [0.909804   0.5372549  0.9215687  ... 0.8705883  0.427451   0.8470589 ]
 ...
 [0.7568628  0.54901963 0.23529413 ... 0.28235295 0.52156866 0.5647059 ]
 [0.909804   0.6862745  0.227451   ... 0.21568629 0.43137258 0.36862746]
 [0.59607846 0.9450981  0.41176474 ... 0.73333335 0.1764706  0.16862746]]
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
[[ 1.7395868  -0.29693064  2.3505423  ...  0.60677403  1.2050011
   0.7976976 ]
 [-0.3987565   0.9377082   1.4341093  ...  2.617835    1.8414128
   1.6759458 ]
 [ 2.5287375   1.3195552   2.5669222  ...  2.4014552   0.9631647
   2.3250859 ]
 ...
 [ 2.0323365   1.3577399   0.33948112 ...  0.49221992  1.2686423
   1.4086528 ]
 [ 2.5287375   1.803228    0.31402466 ...  0.27583995  0.9758929
   0.77224106]
 [ 1.5104787   2.6432917   0.9122518  ...  1.9559668   0.14855757
   0.12310111]]
```

### HWC2CHW

The `HWC2CHW` transform is used to convert the image format. The two different formats (height, width, channel) or (channel, height, width) may be targeted and optimized in different hardware devices. MindSpore sets HWC as the default image format and uses this transform for processing when CHW format is required.

Here we first process the `normalized_image` in the previous section to HWC format, and then convert it. You can see the change of the shape before and after the conversion.

```python
hwc_image = np.expand_dims(normalized_image, -1)
hwc2chw = vision.HWC2CHW()
chw_image = hwc2chw(hwc_image)
print(hwc_image.shape, chw_image.shape)
```

```text
(48, 48, 1) (1, 48, 48)
```

For more Vision Transforms, see [mindspore.dataset.vision](https://mindspore.cn/docs/en/r2.1/api_python/mindspore.dataset.transforms.html#module-mindspore.dataset.vision).

## Text Transforms

The `mindspore.dataset.text` module provides a series of Transforms for text data. Unlike image data, text data requires operations such as Tokenize, building word lists, and Token to Index. Here is a brief description of its usage.

First we define three pieces of text as the data to be processed and load them by using `GeneratorDataset`.

```python
texts = ['Welcome to Beijing']
```

```python
test_dataset = GeneratorDataset(texts, 'text')
```

### PythonTokenizer

Tokenize is a basic transformation to process text data. MindSpore provides many different Tokenizers. Take `PythonTokenizer` as example, it allows users to customize the token strategy. Then we can perform tokenization on the input text based on the `map` operation.

```python
def my_tokenizer(content):
    return content.split()

test_dataset = test_dataset.map(text.PythonTokenizer(my_tokenizer))
print(next(test_dataset.create_tuple_iterator()))
```

```text
[Tensor(shape=[3], dtype=String, value= ['Welcome', 'to', 'Beijing'])]
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
{'to': 2, 'Beijing': 0, 'Welcome': 1}
```

After generating the vocabulary, you can perform the vocabulary mapping transformation with the `map` method to convert Token to Index.

```python
test_dataset = test_dataset.map(text.Lookup(vocab))
print(next(test_dataset.create_tuple_iterator()))
```

```text
[Tensor(shape=[3], dtype=Int32, value= [1, 2, 0])]
```

For more Text Transforms, see [mindspore.dataset.text](https://mindspore.cn/docs/en/r2.1/api_python/mindspore.dataset.transforms.html#module-mindspore.dataset.text).

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
