# Training Process Overview

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/model_train/train_process/overview.md)

With the rapid development of artificial intelligence technology, deep learning has become a core technology in many fields. Deep learning model training is an important part of deep learning, which involves multiple stages and processes.

A complete training process typically consists of four steps, including dataset preprocessing, model creation, defining the loss function and optimizer, and training and saving the model. Normally, defining training and evaluation network and running it directly would be sufficient for basic needs, but when using MindSpore, the network can be encapsulated using `Model`, which simplifies the code to a certain extent and allows easy access to higher-order features such as data sinking and mixed accuracy.

To build a complete training process using MindSpore, you can refer to the following guidelines:

## High-order Encapsulation

For neural networks in simple scenarios, the forward network `network`, the loss function `loss_fn`, the optimizer `optimizer` and the evaluation function `metrics` can be specified when defining the `Model`.

Building a neural network using `Model` is generally divided into four steps as follows:

- **Dataset Preprocessing**: Use ``mindspore.dataset`` to load a dataset and then perform operations such as scaling, normalizing, formatting on the dataset. Refer to the [Dataset Preprocessing](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html) tutorial for more information.
- **Model Building**: Build the neural network using ``mindspore.nn.Cell``, initialize the neural network layers within \ ``__init__``\, and construct the neural network forward execution logic within the ``construct`` function. Refer to the [Model Creation](https://www.mindspore.cn/tutorials/en/master/beginner/model.html) tutorial for more information.
- **Defining Loss Function and Optimizer**: Define the loss function and optimizer function using ``mindspore.nn``. Refer to [Define loss function and optimizer](https://www.mindspore.cn/docs/en/master/model_train/train_process/model.html#defining-loss-function-and-optimizer) tutorial for more information.
- **Training and Saving Models**: Use the ``ModelCheckpoint`` interface to save the network model and parameters, ``model.fit`` for training and evaluating the network, and ``LossMonitor`` to monitor changes in ``loss`` during training. Refer to [Training and saving models](https://www.mindspore.cn/docs/en/master/model_train/train_process/model.html#training-and-saving-model) tutorial for more information.

In addition, during the deep learning training process, we can use the callback mechanism to keep track of the training status of the network model, observe the changes of the parameters of the network model in real time, and implement some user-defined operations during the training process. In addition to MindSpore built-in callback functions, you can also customize your own callback class based on the ``Callback`` base class.

When the training task is over, evaluation functions (Metrics) are often needed to assess how good the model is. Similarly, MindSpore provides evaluation functions for most common tasks, and users can customize Metrics functions by inheriting from the ``mindspore.train.Metric`` base class.

## Performance Optimization Approaches

In the MindSpore deep learning training process, we usually have the following two means to optimize the training performance:

- **Sinking Mode**: MindSpore provides data graph sinking, graph sinking and loop sinking functions, which can greatly reduce the Host-Device interaction overhead and effectively improve the performance of training and inference. Refer to the [Sinking Mode](https://www.mindspore.cn/docs/en/master/model_train/train_process/optimize/sink_mode.html) tutorial for more information.
- **Vectorization Acceleration Interface Vmap**: The ``vmap`` interface converts highly-repetitive arithmetic logic in a model or function into parallel vector arithmetic logic, to gain leaner code logic and more efficient execution performance. Refer to the [Vectorization Acceleration Interface Vmap](https://www.mindspore.cn/docs/en/master/model_train/train_process/optimize/vmap.html) tutorial for more information.

## Advanced Differential Interfaces

MindSpore can use the ``ops.grad`` or ``ops.value_and_grad`` interfaces to perform automatic differentiation, and automatic first-order or higher-order derivation of the function or network to be derived.

## Higher-Order Training Strategies

MindSpore can perform some higher-order training strategies, such as:

- **Gradient Accumulation**: Gradient Accumulation is a way to split the data samples for training a neural network into several small Batches according to the Batch size, and then calculate them sequentially. The purpose is to solve the OOM (Out Of Memory) problem that the neural network can not be trained due to the lack of memory, resulting in the Batch size is too large, or the network model is too large to be loaded. Refer to the [Gradient Accumulation](https://www.mindspore.cn/docs/en/master/model_train/train_process/optimize/gradient_accumulation.html) tutorial for more information.
- **Per-sample-gradients**: per-sample-gradients can help us to better improve the training of the model by more accurately calculating the effect of each sample on the network parameters when training the neural network, and MindSpore provides efficient methods to calculate per-sample-gradients. Refer to the  [Per-sample-gradients](https://www.mindspore.cn/docs/en/master/model_train/train_process/optimize/per_sample_gradients.html) tutorial for more information.