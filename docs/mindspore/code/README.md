# Example of neural network construction and dataset loading

[查看中文](./README_CN.md#)

## LeNet

LeNet was proposed in 1998, a typical convolutional neural network, consisting of two convolutional layers and three fully connected layers.

[Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791)： Y.Lecun, L.Bottou, Y.Bengio, P.Haffner.*Proceedings of the IEEE*.1998.

## MNIST

Dataset used: [MNIST](<http://yann.lecun.com/exdb/mnist/>)

- Dataset size：52.4M，60,000 28*28 in 10 classes
    - Train：60,000 images
    - Test：10,000 images
- Data format：binary files

## Tutorial Catalog

We provides the following two code files that can be directly imported for use:

```bash
.
├── lenet.py // The Network Structure of LeNet5
├── mnist.py // The download, loading, and preprocessing process of MNIST
└── README_CN.md
```

## Quick use of code files

```python
from lenet import LeNet5
from mnist import create_dataset

# Construct a LeNet5 instance
network = LeNet5()

# Construct a dataloader for MNIST
dataloader = create_dataset()
```
