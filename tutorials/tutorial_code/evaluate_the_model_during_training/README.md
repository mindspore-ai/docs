使用数据集: [MNIST](http://yann.lecun.com/exdb/mnist/) 
下载后按照下述结构放置：

```
├─evaluate_the_model_during_training.py
│
└─MNIST_Data
    ├─test
    │      t10k-images.idx3-ubyte
    │      t10k-labels.idx1-ubyte
    │
    └─train
           train-images.idx3-ubyte
           train-labels.idx1-ubyte
```

使用命令`python evaluate_the_model_during_training.py >train.log 2>&1 &`运行（过程较长，大约需要3分钟），运行结果会记录在`log.txt`文件中。