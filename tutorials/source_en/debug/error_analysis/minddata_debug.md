# Data Processing Debugging Methods and Common Errors Analysis

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/debug/error_analysis/minddata_debug.md)&nbsp;&nbsp;

## Data Processing Debugging Methods

### Method 1: Errors in Data Processing Execution, Print Logs or Add Debug Points to Code Debugging

When using `GeneratorDataset` or `map` to load/process data, there may be syntax errors, calculation overflow and other issues that cause data errors, you can generally follow the steps below to troubleshoot and debug:

1. Observe the error stack information and locate the error code block according to the error stack information.

2. Add a print or debugging point near the block of code where the error occurred, to further debugging.

The following shows a data pipeline with syntax/value problems and how to fix the errors according to the above scheme.

```python
import mindspore.dataset as ds

class Loader:
    def __init__(self):
        self.data = [1, 6, 0, 1, 2]
        self.dividend = 1
    def __getitem__(self, index):
        a = self.dividend
        b = self.data[index]
        return a / b
    def __len__(self):
        return len(self.data)


dataloader = ds.GeneratorDataset(Loader(), column_names=["data"])
for data in dataloader:
    print("data", data)
```

After running the error reported as follows, you can observe that the error message is divided into three blocks:

* Dataset Pipeline Error Message: error summary, here suggests that due to the Python code execution error caused by the error exit.
* Python Call Stack: Call information from the Python code, showing the call stack before the Python exception was generated.
* C++ Call Stack: C++ code call information for framework developers to debug.

```text
------------------------------------------------------------------
- Python Call Stack:
------------------------------------------------------------------
Traceback (most recent call last):
  File "/.../mindspore/dataset/engine/datasets_user_defined.py", line 99, in _cpp_sampler_fn
    val = dataset[i]
  File "test_cv.py", line 11, in __getitem__
    return a / b
ZeroDivisionError: division by zero

------------------------------------------------------------------
- Dataset Pipeline Error Message:
------------------------------------------------------------------
[ERROR] Execute user Python code failed, check 'Python Call Stack' above.

------------------------------------------------------------------
- C++ Call Stack: (For framework developers)
------------------------------------------------------------------
mindspore/ccsrc/minddata/dataset/engine/datasetops/source/generator_op.cc(247).
```

Dataset Pipeline Error Message suggests that there is an exception in running the user's Python script, add continue to check the Python Call Stack.
According to the Python Stack information, the exception is thrown from the `__getitem__` function and suggests related code near `return a / b`. Therefore, add a print or debug point to the log near the code that prompts the error report.

```python
import mindspore.dataset as ds

class Loader:
    def __init__(self):
        self.data = [1, 6, 0, 1, 2]
        self.dividend = 1

    def __getitem__(self, index):
        try:
            print(">>> debug: come into __getitem__", flush=True)
            a = self.dividend
            b = self.data[index]
            print(">>> debug: a is", a, flush=True)
            print(">>> debug: b is", b, flush=True)
            return a / b
        except Exception as e:
            print("exception occurred", str(e))
            import pdb
            pdb.set_trace()
            # do anything you want to check variable

    def __len__(self):
        return len(self.data)


dataloader = ds.GeneratorDataset(Loader(), column_names=["data"])
# Make the pipeline single-threaded before you run it
ds.config.set_num_parallel_workers(1)
for i, data in enumerate(dataloader):
    print("data count", i)
```

Rerun the data pipeline with the relevant debugging information to see that the exception was caught and the pdb debugger was entered.
At this point, you can print the relevant variables as needed (following pdb syntax) and debug, and find the 1/0 error that caused the divide-by-zero error.

```text
>>> debug: come into __getitem__
>>> debug: a is 1
>>> debug: b is 2
>>> debug: come into __getitem__
data count 0
>>> debug: a is 1
>>> debug: b is 0
exception occurred division by zero
--Return--
> /test_cv.py(19)__getitem__()->None
-> pdb.set_trace()
(Pdb)
```

### Method 2: Data-enhanced Map Operation Error, Testing the Each Data Processing Operator in the Map Operation

Embedding data augmentation transformations into the `map` operation of a data pipeline can sometimes result in errors that are not easily debugged.
The following example shows an example of embedding a `RandomResize` and `Crop` enhancements into a `map` operation to crop data,
but an error is reported due to an error in the transformed shape of the input object.

#### Way One: Debugging Through the Execution of Individual Operators

```python
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

class MyDataset:
    def __init__(self):
        self.data = [np.ones((32, 32, 3)), np.ones((3, 48, 48))]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)


dataset = ds.GeneratorDataset(MyDataset(), column_names=["data"])
transforms_list = [vision.RandomResize((3, 16)), vision.Crop(coordinates=(0, 0), size=(8, 8))]
dataset = dataset.map(operations=transforms_list)
for data in dataset:
    print("data", data)
```

When executing the above example you get the following error, but based on the error message it is more difficult to get what the input object is and what the shape is.

```text
------------------------------------------------------------------
- Dataset Pipeline Error Message:
------------------------------------------------------------------
[ERROR] map operation: [Crop] failed. Crop: Crop height dimension: 8 exceeds image height: 3.

------------------------------------------------------------------
- C++ Call Stack: (For framework developers)
------------------------------------------------------------------
mindspore/ccsrc/minddata/dataset/kernels/image/crop_op.cc(33).
```

From the Dataset Pipeline Error Message, we can see that the error is thrown by `Crop` during the calculation. So you can rewrite the dataset pipeline a bit to print the input and output of `Crop` and add printing for debugging.

```python
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

class MyDataset:
    def __init__(self):
        self.data = [np.ones((32, 32, 3)), np.ones((3, 48, 48))]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def MyWrapper(data):
    transforms_list = [vision.RandomResize((3, 16)), vision.Crop(coordinates=(0, 0), size=(8, 8))]
    for transforms in transforms_list:
        print(">>> debug: apply transforms: ", type(transforms))
        print(">>> debug: before apply transforms, data shape", data.shape)
        data = transforms(data)
        print(">>> debug: after apply transforms, data shape", data.shape)
    return data


dataset = ds.GeneratorDataset(MyDataset(), column_names=["data"], shuffle=False)
dataset = dataset.map(MyWrapper)
ds.config.set_num_parallel_workers(1)
for data in dataset:
    print("data", data[0].shape)
```

Running it again yields the following.

```text
>>> debug: apply transforms:  <class 'mindspore.dataset.vision.transforms.RandomResize'>
>>> debug: before apply transforms, data shape (32, 32, 3)
>>> debug: after apply transforms, data shape (3, 16, 3)
>>> debug: apply transforms:  <class 'mindspore.dataset.vision.transforms.Crop'>
>>> debug: before apply transforms, data shape (3, 16, 3)

RuntimeError: Exception thrown from user defined Python function in dataset.

------------------------------------------------------------------
- Dataset Pipeline Error Message:
------------------------------------------------------------------
[ERROR] Crop: Crop height dimension: 8 exceeds image height: 3.

------------------------------------------------------------------
- C++ Call Stack: (For framework developers)
------------------------------------------------------------------
mindspore/ccsrc/minddata/dataset/kernels/image/crop_op.cc(33).
```

According to the printed information you can see that `Crop` processed the first sample and reported an error. The shape of the first sample (32, 32, 3), was transformed by `RandomResize` to (3, 16, 3), but the shape transformed by `Crop` did not printed and then an error is reported. So it is the fact that the shape cannot be processed by `Crop` that causes the error. Further, according to the Dataset Pipeline Error Message, the input sample has a height of only 3, but is expected to be cropped to a region with a high dimension of 8, hence the error is reported.

Checking the [API description](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset_vision/mindspore.dataset.vision.Crop.html) of `Crop` , `Crop` requires the input sample to be in shape <H, W> or <H, W, C>, so `Crop` treats (3, 48, 48) as <H, W, C>, and naturally it can't crop out the region with H=8, W=8 when H=3, W=48, C=48.

To quickly fix this, We just need to change the parameter size of `RandomResize` from (3, 16) to (16, 16), and run it again to find that the use case passes.

```text
>>> debug: apply transforms:  <class 'mindspore.dataset.vision.transforms.RandomResize'>
>>> debug: before apply transforms, data shape (32, 32, 3)
>>> debug: after apply transforms, data shape (16, 16, 3)
>>> debug: apply transforms:  <class 'mindspore.dataset.vision.transforms.Crop'>
>>> debug: before apply transforms, data shape (16, 16, 3)
>>> debug: after apply transforms, data shape (8, 8, 3)
data (8, 8, 3)
>>> debug: apply transforms:  <class 'mindspore.dataset.vision.transforms.RandomResize'>
>>> debug: before apply transforms, data shape (3, 48, 48)
>>> debug: after apply transforms, data shape (16, 16, 48)
>>> debug: apply transforms:  <class 'mindspore.dataset.vision.transforms.Crop'>
>>> debug: before apply transforms, data shape (16, 16, 48)
>>> debug: after apply transforms, data shape (8, 8, 48)
data (8, 8, 48)
```

#### Way Two: Debugging Map Operation Through Data Pipline Debugging Mode

We can also turn on the dataset pipline debug mode by calling the [set_debug_mode](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset/mindspore.dataset.config.set_debug_mode.html) .
When debug mode is enabled, the random seed is set to 1 if it is not already set, so that executing the dataset pipeline in debug mode can yield deterministic results.

The process is as follows:

1. Print the shape and type of the input and output data for each transform op in the `map` operator.
2. Enable the dataset pipeline debug mode and use either a predefined debug hook provided by MindData or a user-defined debug hook. It must define the class inherited from [DebugHook](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset/mindspore.dataset.debug.DebugHook.html).

The following is a modification of the `Way One` use case, using the predefined debug hooks provided by MindData.

```python
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.debug as debug
import mindspore.dataset.vision as vision

class MyDataset:
    def __init__(self):
        self.data = [np.ones((32, 32, 3)), np.ones((3, 48, 48))]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Enable dataset pipeline debug mode and use pre-defined debug hook provided by MindData.
ds.config.set_debug_mode(True)

# Define dataset pipeline
dataset = ds.GeneratorDataset(MyDataset(), column_names=["data"])

transforms_list = [vision.RandomResize((3, 16)), vision.Crop(coordinates=(0, 0), size=(8, 8))]
dataset = dataset.map(operations=transforms_list)
for i, data in enumerate(dataset):
    print("data count", i)
```

Running it yields the following correlation.

```text
[Dataset debugger] Print the [INPUT] of the operation [RandomResize].
Column 0. The dtype is [float64]. The shape is [(32, 32, 3)].
[Dataset debugger] Print the [OUTPUT] of the operation [RandomResize].
Column 0. The dtype is [float64]. The shape is [(3, 16, 3)].
    ......
E           RuntimeError: Exception thrown from dataset pipeline. Refer to 'Dataset Pipeline Error Message'.
E
E           ------------------------------------------------------------------
E           - Dataset Pipeline Error Message:
E           ------------------------------------------------------------------
E           [ERROR] map operation: [Crop] failed. Crop: Crop height dimension: 8 exceeds image height: 3.
E
E           ------------------------------------------------------------------
E           - C++ Call Stack: (For framework developers)
E           ------------------------------------------------------------------
E           mindspore/ccsrc/minddata/dataset/kernels/image/crop_op.cc(33).
```

Based on the printed information, we can clearly see that `Crop` is getting an error when processing the input shape of
(3, 16, 3). Refer to `Crop`'s [API description](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset_vision/mindspore.dataset.vision.Crop.html), and we just need to change the parameter size of `RandomResize` from (3, 16) to (16, 16), and run it again to see that the use case passes.

```text
[Dataset debugger] Print the [INPUT] of the operation [RandomResize].
Column 0. The dtype is [float64]. The shape is [(32, 32, 3)].
[Dataset debugger] Print the [OUTPUT] of the operation [RandomResize].
Column 0. The dtype is [float64]. The shape is [(16, 16, 3)].
[Dataset debugger] Print the [OUTPUT] of the operation [Crop].
Column 0. The dtype is [float64]. The shape is [(8, 8, 3)].
******data count 0
[Dataset debugger] Print the [INPUT] of the operation [RandomResize].
Column 0. The dtype is [float64]. The shape is [(3, 48, 48)].
[Dataset debugger] Print the [OUTPUT] of the operation [RandomResize].
Column 0. The dtype is [float64]. The shape is [(16, 16, 48)].
[Dataset debugger] Print the [OUTPUT] of the operation [Crop].
Column 0. The dtype is [float64]. The shape is [(8, 8, 48)].
******data count 1
```

Alternatively, you can use a custom debug hook to manually insert, add breakpoints to the `compute` function of the `MyHook` class, and print a log to see the type and shape of the data.

```python
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.debug as debug
import mindspore.dataset.vision as vision

# Enable dataset pipeline debug mode and use user-defined debug hook. It must define a
# class inherited from DebugHook.
class MyHook(debug.DebugHook):
    def __init__(self):
        super().__init__()

    def compute(self, *args):
        print("come into my hook function, block with pdb", flush=True)
        import pdb
        print("the input shape is: ", args[0][0].shape, flush=True)
        pdb.set_trace()
        return args


class MyDataset:
    def __init__(self):
        self.data = [np.ones((32, 32, 3)), np.ones((3, 48, 48))]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Enable dataset pipeline debug mode and use pre-defined debug hook provided by MindData.
ds.config.set_debug_mode(True, debug_hook_list=[MyHook()])

# Define dataset pipeline.
dataset = ds.GeneratorDataset(MyDataset(), column_names=["data"])

# Insert debug hook before `Crop` operation.
transforms_list = [vision.RandomResize((3, 16)), MyHook(), vision.Crop(coordinates=(0, 0), size=(8, 8))]
dataset = dataset.map(operations=transforms_list)
for i, data in enumerate(dataset):
    print("data count", i)
```

As above, the problem can be localized by looking at the input shape step-by-step, and next you can start your debugging:

```text
[Dataset debugger] Print the [INPUT] of the operation [RandomResize].
come into my hook function, block with pdb
the input shape is:  (3, 48, 48)

>>>>>>>>>>>>>>>>>>>>>PDB set_trace>>>>>>>>>>>>>>>>>>>>>
> /test_demo.py(18)compute
-> return args
(Pdb)
```

### Method 3: Testing Data Processing Performance

When training is initiated using MindSpore and the training log keeps printing with many entries, it is likely that there is a problem with slower data processing.

```text
[WARNING] MD(90635,fffdf0ff91e0,python):2023-03-25-15:29:14.801.601 [mindspore/ccsrc/minddata/dataset/engine/datasetops/source/generator_op.cc:220] operator()] Bad performance attention,
it takes more than 25 seconds to generator.__next__ new row, which might cause `GetNext` timeout problem when sink_mode=True.
You can increase the parameter num_parallel_workers in GeneratorDataset / optimize the efficiency of obtaining samples in the user-defined generator function.
```

Here is a way to debug the performance of the dataset, even if the above WARNING message does not appear, as a reference
Construct a simple lenet training network and simply modify the code so that there is a warning message in the run results.

```python
import time
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore.dataset.vision import Inter

def create_dataset(data_path, num_parallel_workers=1):
    mnist = ds.MnistDataset(data_path, num_samples=1000, shuffle=False)

    class udf:
        def __init__(self, dataset):
            self.dataset = dataset
            self.cnt = 0

            self.iterator = self.dataset.create_tuple_iterator(num_epochs=1)
            self.data = []
            for i in range(1000):
                self.data.append(self.iterator.__next__())

        def __len__(self):
            return 1000

        def __getitem__(self, index):
            if index >= 7:
                time.sleep(60)
            return self.data[index]


    mnist_ds = ds.GeneratorDataset(udf(mnist), ["image", "label"])

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml * rescale, shift_nml)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # use map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label")
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image")
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image")
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image")
    mnist_ds = mnist_ds.batch(4, drop_remainder=True)
    return mnist_ds


class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


dataset_train = create_dataset("mnist/train")

ms.set_context(mode=ms.GRAPH_MODE)
network = LeNet5(num_class=10)
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
model = ms.Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'accuracy'})
model.train(10, dataset_train, callbacks=[ms.LossMonitor()])
```

While training, we will get very many WARNINGs suggesting that our dataset performance is slow, but observe that there are Epoch time, per step time messages, so the training is actually going on, just slower.

```text
[WARNING] MD(90635,fffdf0ff91e0,python):2023-03-25-15:29:14.801.601 [mindspore/ccsrc/minddata/dataset/engine/datasetops/source/generator_op.cc:220] operator()] Bad performance attention, it takes more than 25 seconds to generator.__next__ new row, which might cause `GetNext` timeout problem when sink_mode=True. You can increase the parameter num_parallel_workers in GeneratorDataset / optimize the efficiency of obtaining samples in the user-defined generator function.
[WARNING] MD(90635,fffd72ffd1e0,python):2023-03-25-15:29:14.802.398 [mindspore/ccsrc/minddata/dataset/engine/datasetops/data_queue_op.cc:903] DetectPerBatchTime] Bad performance attention, it takes more than 25 seconds to fetch a batch of data from dataset pipeline, which might result `GetNext` timeout problem. You may test dataset processing performance(with creating dataset iterator) and optimize it.
Epoch time: 60059.685 ms, per step time: 30029.843 ms, avg loss: 2.301
```

At this point, it is possible to iterate through the dataset individually and see the processing time for each piece of data to determine how well the dataset is performing:
After `dataset_train = create_dataset("mnist/train")` in the above code, the following code can be added to debug the dataset

```python
import time

st = time.time()
for i, data in enumerate(dataset_train):
    print("data step", i, ", time", time.time() - st, flush=True)
    st = time.time()
    if i > 50:
        break
```

After adding the code and running it again, you will see the processing time of the dataset:

```text
data step 0 , time 0.0055468082427978516
data step 1 , time 60.034635634525
data step 2 , time 480.046234134121
data step 3 , time 480.023415324343
data step 4 , time 480.051423635473
```

As you can see, from the 2nd data, each data actually has to wait for more than 60s before processing is completed, for the above "modified code" is actually a good solution, check the code will find that:

```python
def __getitem__(self, index):
    if index >= 7:
        time.sleep(60)
    return self.data[index]

```

From the 7th piece of data, every piece of data will sleep 60 seconds before output, and it is here that the data processing slows down. Because the batch size is 4, so the first batch contains only the first 4 data (0,1,2,3), natural processing time is not a problem, and then to the second batch, because it contains (4,5,6,7) 4 data, so the seventh piece of data will wait an additional 60s before output, which causes that in the second batch, the data time is extended to 60s. Same for the third and fourth batch after that. So you only need to remove the logic of sleep to bring the data processing back to the normal level.

In real training scenarios, there are different reasons for slow network training, but the analysis method is similar. We can iterate through the data individually to determine if the slow data processing is the cause of the low training performance.

### Method 4: Checking For Exception Data In Data Processing

In the process of processing data, abnormal result values may be generated due to computational errors, numerical overflow, etc., which can lead to problems such as operator computation overflow and abnormal weight updates when training the network. This scheme describes how to debug and check abnormal data behavior/data results.

#### Turning Off Shuffling and Fixing Random Seeds to Ensure Reproductivity

In some data processing scenarios, we use randomized functions as part of data operations. Due to the nature of the random operation itself, the data results are not the same in every run, so that this will most likely result in abnormal values in the results of the previous run, but in the next run, the abnormal values are not checked, then it is likely because of the effect of the random index/random computation. In this case, it is possible to turn off the shuffling option for the dataset and fix a different random seeds to look for possible introduction of random problems through multiple runs.

The following example treats a random value as a divisor, which by chance will divide by zero.

```python
import numpy as np
import mindspore as ms

class Gen():
    def __init__(self):
        self.data = [np.array(i) / np.random.randint(0, 3) for i in range(1, 4)]
    def __getitem__(self, index):
        data = self.data[index]
        return data
    def __len__(self):
        return len(self.data)


dataset = ms.dataset.GeneratorDataset(Gen(), ["data"])
for data in dataset:
    print(data)
```

Setting the random seed with `set_seed` to produce a fixed random number to achieve a deterministic result allows further troubleshooting of the code to see if the randomization is working as expected.

```python
ms.set_seed(1)
ms.dataset.GeneratorDataset(Loader(), ["data"], shuffle=False)
```

The results are consistent across multiple runs, as can be seen by the division by zero results that occur for the 1st and 3rd data. Indirectly, it can be shown that there is an anomaly in the computation of the 1st and 3rd data that leads to the value of inf.

```text
[Tensor(shape=[], dtype=Float64, value= inf)]
[Tensor(shape=[], dtype=Float64, value= 1)]
[Tensor(shape=[], dtype=Float64, value= inf)]
```

#### A Quick Check of the Results Using a Tool Such as NumPy

In the previous example, the amount of data is small enough that you can basically check the code to find out where the anomalies are. For some large high-dimensional arrays, it is less convenient to check the code or print the values. At this time, you can configure MindSpoer's dataset to return data in the form of NumPy, and use some of NumPy's commonly used means of checking the contents of the array to check whether there are abnormal values in the array.

The following example constructs a large, high-dimensional array and performs random operations on the values in it.

```python
import numpy as np
import mindspore as ms

class Gen():
    def __init__(self):
        self.data = np.random.randint(0, 255, size=(16, 50, 50))
    def __getitem__(self, index):
        data = self.data[index] / np.random.randint(0, 2)
        return data
    def __len__(self):
        return 16


dataset = ms.dataset.GeneratorDataset(Gen(), ["data"])
for data in dataset:
    print(data)
```

To check for the presence of unusual values such as nan, inf, etc. during data operations, you can specify the output of the dataset object to be of type NumPy when traversing it.

After specifying the output type, each element of the printed data object is of NumPy type, based on which you can use some very convenient functions in NumPy to check whether the values are abnormal or not.

```python
for data_index, data in enumerate(dataset.create_tuple_iterator(output_numpy=True)):
    if(np.isinf(data).any()):             # Checking for inf values
        print("np.isinf index: ", data_index) # Prints the index of the sample if there is an inf value
    if(np.isnan(data).any()):             # Checking for nan values
        print("np.isinf index: ", data_index) # Prints an index of samples with nan values
```

## Analyzing Common Data Processing Problems

### Data Preparation

Common errors you may encounter in the data preparation phase include dataset path and MindRecord file errors when you read or save data from or to a path or when you read or write a MindRecord file.

#### The Dataset Path Contains Chinese Characters

Error log:

```text
RuntimeError: Unexpected error. Failed to open file, file path E:\深度学习\models-master\official\cv\ssd\MindRecord_COCO\test.mindrecord
```

Two solutions are available:

1. Specify the output path of the MindRecord dataset to a path containing only English characters.

2. Upgrade MindSpore to a version later than 1.6.0.

For details, visit the following website:

[MindRecord Data Preparation - Unexpected error. Failed to open file_MindSpore](https://www.hiascend.com/forum/thread-0231107679243990127-1-1.html)

#### MindRecord File Error

* The Duplicate File Is Not Deleted

    Error log:

    ```text
    MRMOpenError: [MRMOpenError]: MindRecord File could not open successfully.
    ```

    Solution:

    1. Add the file deletion logic to the code to ensure that the MindRecord file with the same name in the directory is deleted before the file is saved.

    2. In versions later than MindSpore 1.6.0, when defining the `FileWriter` object, add `overwrite=True` to implement overwriting.

    For details, visit the following website:

    [MindSpore Data Preparation - MindRecord File could not open successfully](https://www.hiascend.com/forum/thread-0231107679243990127-1-1.html)

* The File Is Moved

    Error log:

    ```text
    RuntimeError: Thread ID 1 Unexpected error. Fail to open ./data/cora
    RuntimeError: Unexpected error. Invalid file, DB file can not match file
    ```

    When MindSpore 1.4 or an earlier version is used, in the Windows environment, after a MindRecord dataset file is generated and moved, the file cannot be loaded to MindSpore.

    Solution:

    1. Do not move the MindRecord file generated in the Windows environment.

    2. Upgrade MindSpore to 1.5.0 or a later version and regenerate a MindRecord dataset. Then, the dataset can be copied and moved properly.

    For details, visit the following website:

    [MindSpore Data Preparation - Invalid file,DB file can not match_MindSpore](https://www.hiascend.com/forum/thread-0229106992212728097-1-1.html)

* The User-defined Data Type Is Incorrect

    Error log:

    ```text
    RuntimeError: Unexpected error. Invalid data, the number of schema should be positive but got: 0. Please check the input schema.
    ```

    Solution:

    Modify the input data type to ensure that it is consistent with the type definition in the script.

    For details, visit the following website:

    [MindSpore Data Preparation - Unexpected error. Invalid data](https://www.hiascend.com/forum/thread-0231107678315400125-1-1.html)

### Data Loading

In the data loading phase, errors may be reported in resource configuration, `GeneratorDataset`, and iterators.

#### Resource Configuration

* Incorrect Number of CPU Cores

    Error log:

    ```text
    RuntimeError: Thread ID 140706176251712 Unexpected error. GeneratorDataset's num_workers=8, this value is not within the required range of [1, cpu_thread_cnt=2].
    ```

    Solution:

    1. Add the following code to manually configure the number of CPU cores: `ds.config.set_num_parallel_workers()`

    2. Upgrade to MindSpore 1.6.0, which automatically adapts to the number of CPU cores in the hardware to prevent errors caused by insufficient CPU cores.

    For details, visit the following website:

    [MindSpore Data Loading - Unexpected error. GeneratorDataset's num_workers=8, this value is not within the required range of](https://www.hiascend.com/forum/thread-0215121940801939033-1-1.html)

* Incorrect PageSize Setting

    Error log:

    ```text
    RuntimeError: Syntax error. Invalid data, Page size: 1048576 is too small to save a blob row.
    ```

    Solution:

    Call the set_page_size API to set pagesize to a larger value. The setting method is as follows:

    ```python
    from mindspore.mindrecord import FileWriter
    writer = FileWriter(file_name="test.mindrecord", shard_num=1)
    writer.set_page_size(1 << 26) # 128MB
    ```

    For details, visit the following website:

    [MindSpore Data Loading - Invalid data,Page size is too small"](https://www.hiascend.com/forum/thread-0231107680001698128-1-1.html)

#### `GeneratorDataset`

* Suspended `GeneratorDataset` Thread

    No error log is generated, and the thread is suspended.

    During customized data processing, the `numpy.ndarray` and `mindspore.Tensor` data type are mixed and the `numpy.array(Tensor)` type is incorrectly used for conversion. As a result, the global interpreter lock (GIL) cannot be released and the `GeneratorDataset` cannot work properly.

    Solution:

    1. When defining the first input parameter `source` of `GeneratorDataset`, use the `numpy.ndarray` data type if a Python function needs to be invoked.

    2. Use the `Tensor.asnumpy()` method to convert `Tensor` to `numpy.ndarray`.

    For details, visit the following website:

    [MindSpore Data Loading - Suspended GeneratorDataset Thread](https://www.hiascend.com/forum/thread-0232106992052900089-1-1.html)

* Incorrect User-defined Return Type

    Error log:

    ```python
    Unexpected error. Invalid data type.
    ```

    Error description:

    A user-defined `Dataset` or `map` operation returns data of the dict type, not a numpy array or a tuple consisting of numpy arrays. Data types (such as dict and object) other than numpy array or a tuple consisting of numpy arrays are not controllable and the data storage mode is unclear. As a result, the `Invalid type` error is reported.

    Solution:

    1. Check the return type of the customized data processing. The return type must be numpy array or a tuple consisting of numpy arrays.

    2. Check the return type of the `__getitem__` function during customized data loading. The return type must be a tuple consisting of numpy arrays.

    For details, visit the following website:

    [MindSpore Dataset Loading - Unexpected error. Invalid data type_MindSpore](https://www.hiascend.com/forum/thread-0231107678315400125-1-1.html)

* User-defined Sampler Initialization Error

    Error log:

    ```text
    AttributeError: 'IdentitySampler' object has no attribute 'child_sampler'
    ```

    Solution:

    In the user-defined sampler initialization method '\_\_init\_\_()', use 'super().\_\_init\_\_()' to invoke the constructor of the parent class.

    For details, visit the following website:

    [MindSpore Dataset Loading - 'IdentitySampler' has no attribute child_sampler](https://www.hiascend.com/forum/thread-0229107679386960150-1-1.html)

* Repeated Access Definition

    Error log:

    ```python
    For 'Tensor', the type of "input_data" should be one of ...
    ```

    Solution:

    Select a proper data input method: random access (`__getitem__`) or sequential access (iter, next).

    For details, visit the following website:

    [MindSpore Dataset Loading - the type of `input_data` should be one of](https://www.hiascend.com/forum/thread-0229107683010760153-1-1.html)

* Inconsistency Between the Fields Returned by the User-defined Data and the Defined Fields

    Error log:

    ```text
    RuntimeError: Exception thrown from PyFunc. Invalid python function, the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in column_names
    ```

    Solution:

    Check whether the fields returned by `GeneratorDataset` are the same as those defined in `columns`.

    For details, visit the following website:

    [MindSpore Dataset Loading -Exception thrown from PyFunc](https://www.hiascend.com/forum/thread-0232107680321371137-1-1.html)

* Incorrect User Script

    Error log:

    ```text
    TypeError: parse() missing 1 required positionnal argument: 'self'
    ```

    Solution:

    Debug the code step by step and check the syntax in the script to see whether '()' is missing.

    For details, visit the following website:

    [MindSpore Dataset Loading - parse() missing 1 required positional](https://www.hiascend.com/forum/thread-0235121940704650030-1-1.html)

* Incorrect Use of Tensor Operations or Operators in Custom Datasets

    Error log:

    ```text
    RuntimeError: Exception thrown from PyFunc. RuntimeError: mindspore/ccsrc/pipeline/pynative/pynative_execute.cc:1116 GetOpOutput] : The pointer[cnode] is null.
    ```

    Error description:

    Tensor operations or operators are used in custom datasets. Because data processing is performed in multi-thread parallel mode and tensor operations or operators do not support multi-thread parallel execution, an error is reported.

    Solution:

    In the user-defined Pyfunc, do not use MindSpore tensor operations or operators in `__getitem__` in the dataset. You are advised to convert the input parameters to the Numpy type and then perform Numpy operations to implement related functions.

    For details, visit the following website:

    [MindSpore Dataset Loading - The pointer[cnode] is null](https://www.hiascend.com/forum/thread-0230106992306834091-1-1.html)

* Index Out of Range Due to Incorrect Iteration Initialization

    Error log:

    ```python
    list index out of range
    ```

    Solution:

    Remove unnecessary `index` member variables, or set `index` to 0 before each iteration to perform the reset operation.

    For details, visit the following website:

    [MindSpore Dataset Loading - list index out of range](https://www.hiascend.com/forum/thread-0232107679694236136-1-1.html)

* No Iteration Initialization

    Error log:

    ```python
    Unable to fetch data from GeneratorDataset, try iterate the source function of GeneratorDataset or check value of num_epochs when create iterator.
    ```

    The value of `len` is inconsistent with that of `iter` because iteration initialization is not performed.

    Solution:

    Clear the value of `iter`.

    For details, visit the following website:

    [MindSpore Dataset Loading - Unable to fetch data from GeneratorDataset](https://www.hiascend.com/forum/thread-0215121940606533032-1-1.html)

#### Iterator

* Repeated Iterator Creation

    Error log:

    ```python
    oserror: [errno 24] too many open files
    ```

    Error description:

    If `iter()` is repeatedly called, iterators are repeatedly created. However, because `GeneratorDataset` loads datasets in multi-thread mode by default, the handles opened each time cannot be released before the main process stops. As a result, the number of opened handles keeps increasing.

    Solution:

    Use the dict iterator `create_dict_iterator()` and tuple iterator `create_tuple_iterator()` provided by MindSpore.

    For details, visit the following website:

    [MindSpore Data Loading - too many open files](https://www.hiascend.com/forum/thread-0231107678973789126-1-1.html)

* Improper Data Acquisition from the Iterator

    Error log:

    ```python
    'DictIterator' has no attribute 'get_next'
    ```

    Solution:

    You can obtain the next piece of data from the iterator in either of the following ways:

    ```python
    item = next(ds_test.create_dict_iterator())

    for item in ds_test.create_dict_iterator():
    ```

    For details, visit the following website:

    [MindSpore Dataset Loading - 'DictIterator' has no attribute 'get_next'](https://www.hiascend.com/forum/thread-0230107679565465123-1-1.html)

### Data Augmentation

In the data augmentation phase, the read data is processed. Currently, MindSpore supports common data processing operations, such as shuffle, batch, repeat, and concat. You may encounter the following errors in this phase: data type errors, interface parameter type errors, consumption node conflict, data batch errors, and memory resource errors.

#### Incorrect Data Type for Invoking A Third-party Library API in A User-defined Data Augmentation Operation

Error log:

```text
TypeError: Invalid object with type'<class 'PIL.Image.Image'>' and value'<PIL.Image.Image image mode=RGB size=180x180 at 0xFFFF6132EA58>'.
```

Solution:

Check the data type requirements of the third-party library API used in the user-defined function, and convert the input data type to the data type expected by the API.

For details, visit the following website:

[MindSpore Data Augmentation - TypeError: Invalid with type](https://www.hiascend.com/forum/thread-0229107679078336149-1-1.html)

#### Incorrect Parameter Type in A User-defined Data Augmentation Operation

Error log:

```text
Exception thrown from PyFunc. TypeError: args should be Numpy narray. Got <class 'tuple'>.
```

Solution:

Change the number of input parameters of `call` (except `self`) to the number of parameters in `input_columns` and their type to numpy.ndarray. If `input_columns` is ignored, the number of all data columns is used by default.

For details, visit the following website:

[MindSpore Data Augmentation - args should be Numpy narray](https://www.hiascend.com/forum/thread-0230107678833189122-1-1.html)

#### Consumption Node Conflict in the Dataset

Error log:

```text
ValueError: The data pipeline is not a tree (i.e. one node has 2 consumers)
```

Error description:

A branch occurs in the dataset definition. As a result, the dataset cannot determine the direction.

Solution:

Check the dataset name. Generally, retain the same dataset name.

For details, visit the following website:

[MindSpore Data Augmentation - The data pipeline is not a tree](https://www.hiascend.com/forum/thread-0230107678474985121-1-1.html)

#### Improper Batch Operation Due to Inconsistent Data Shapes

Error log:

```text
RuntimeError: Unexpected error. Inconsistent batch shapes, batch operation expect same shape for each data row, but got inconsistent shape in column 0, expected shape for this column is:, got shape:
```

Solution:

1. Check the shapes of the data that requires the batch operation. If the shapes are inconsistent, cancel the batch operation.

2. If you need to perform the batch operation on the data with inconsistent shapes, sort out the dataset and unify the shapes of the input data by padding.

For details, visit the following website:

[MindSpore Data Augmentation - Unexpected error. Inconsistent batch](https://www.hiascend.com/forum/thread-0254121940499220038-1-1.html)

#### High Memory Usage Due to Data Augmentation

Error description:

If the memory is insufficient when MindSpore performs data augmentation, MindSpore may automatically exit. In MindSpore 1.7 and later versions, an alarm is generated when the memory usage exceeds 80%. When performing large-scale data training, pay attention to the memory usage to prevent direct exit due to high memory usage.

For details, visit the following website:

[MindSpore Data Augmentation - Automatic Exit Due to Insufficient Memory](https://www.hiascend.com/forum/thread-0230107679768460124-1-1.html)
