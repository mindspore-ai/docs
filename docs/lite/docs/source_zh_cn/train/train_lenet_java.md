# 基于Java接口实现端侧训练

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/train/train_lenet_java.md)

## 概述

本教程通过构建并部署Java版本的LeNet网络的训练，演示MindSpore Lite端侧训练Java接口的使用。首先指导您在本地成功训练LeNet模型，然后讲解示例代码。

## 准备

### 环境要求

- 系统环境：Linux x86_64，推荐使用Ubuntu 18.04.02LTS

- 软件依赖

    - [Git](https://git-scm.com/downloads) >= 2.28.0

    - [Maven](https://maven.apache.org/download.cgi) >= 3.3

    - [OpenJDK](https://openjdk.java.net/install/) 1.8 到 1.15

### 下载MindSpore并编译端侧训练Java包

首先克隆源码，然后编译MindSpore Lite端侧训练Java包，`Linux`指令如下：

```bash
git clone -b v2.6.0rc1 https://gitee.com/mindspore/mindspore.git
cd mindspore
bash build.sh -I x86_64 -j8
```

编译环境要求以及环境变量设置，请参考[编译MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html)章节。
本教程使用的示例源码在`mindspore/lite/examples/train_lenet_java`目录。

### 下载数据集

示例中的`MNIST`数据集由10类28*28的灰度图片组成，训练数据集包含60000张图片，测试数据集包含10000张图片。

> MNIST数据集官网下载地址：<http://yann.lecun.com/exdb/mnist/>，共4个下载链接，分别是训练数据、训练标签、测试数据和测试标签。

下载并解压到本地，解压后的训练和测试集分别存放于`/PATH/MNIST_Data/train`和`/PATH/MNIST_Data/test`路径下。

目录结构如下：

```text
MNIST_Data/
├── test
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── train
    ├── train-images-idx3-ubyte
    └── train-labels-idx1-ubyte
```

## 部署应用

### 构建与运行

1. 首先进入示例工程所在目录，运行示例程序，命令如下：

    ```bash
    cd /codes/mindspore/mindspore/lite/examples/train_lenet_java
    ./prepare_and_run.sh -D /PATH/MNIST_Data/ -r ../../../../output/mindspore-lite-${version}-linux-x64.tar.gz
    ```

    > ../resources/model/lenet_tod.ms是示例工程中预置的LeNet训练模型，您也可以参考[训练模型转换](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/train/converter_train.html)，自行转换出LeNet模型。
    >
    > /PATH/MNIST_Data/是MNIST数据集所在路径。

    示例运行结果如下：

    ```text
    ==========Loading Model, Create Train Session=============
    Model path is ../model/lenet_tod.ms
    batch_size: 4
    virtual batch multiplier: 16
    ==========Initing DataSet================
    train data cnt: 60000
    test data cnt: 10000
    ==========Training Model===================
    step_500: Loss is 0.05553353 [min=0.010149269] max_acc=0.9543269
    step_1000: Loss is 0.15295759 [min=0.0018140086] max_acc=0.96594554
    step_1500: Loss is 0.018035552 [min=0.0018140086] max_acc=0.9704527
    step_2000: Loss is 0.029250022 [min=0.0010245014] max_acc=0.9765625
    step_2500: Loss is 0.11875624 [min=7.5288175E-4] max_acc=0.9765625
    step_3000: Loss is 0.046675075 [min=7.5288175E-4] max_acc=0.9765625
    step_3500: Loss is 0.034442786 [min=4.3545474E-4] max_acc=0.97686297
    ==========Evaluating The Trained Model============
    accuracy = 0.9770633
    Trained model successfully saved: ./model/lenet_tod_trained.ms
    ```

## 示例程序详细说明  

### 示例程序结构

```text
train_lenet_java
├── lib
├── build.sh
├── model
│   ├── lenet_export.py
│   ├── prepare_model.sh
│   └── train_utils.sh
├── pom.xml
├── prepare_and_run.sh
├── resources
│   └── model
│       └── lenet_tod.ms   # LeNet训练模型
├── src
│   └── main
│       └── java
│           └── com
│               └── mindspore
│                   └── lite
│                       ├── train_lenet
│                       │   ├── DataSet.java      # MNIST数据集处理
│                       │   ├── Main.java         # Main函数
│                       │   └── NetRunner.java    # 整体训练流程

```

### 编写端侧推理代码

详细的Java接口使用请参考<https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/index.html>。

1. 加载并编译MindSpore Lite模型文件，构建会话。

    ```java
        MSContext context = new MSContext();
        // use default param init context
        context.init();
        boolean isSuccess = context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
        TrainCfg trainCfg = new TrainCfg();
        trainCfg.init();
        model = new Model();
        Graph graph = new Graph();
        graph.load(modelPath);
        model.build(graph, context, trainCfg);
        model.setupVirtualBatch(virtualBatch, 0.01f, 1.00f);
    ```

2. 切换为训练模式，循环迭代，训练模型。

    ```java
    model.setTrainMode(true)
    float min_loss = 1000;
    float max_acc = 0;
    for (int i = 0; i < cycles; i++) {
        for (int b = 0; b < virtualBatch; b++) {
            fillInputData(ds.getTrainData(), false);
            model.runStep();
            float loss = getLoss();
            if (min_loss > loss) {
                min_loss = loss;
            }
            if ((b == 0) && ((i + 1) % 500 == 0)) {
                float acc = calculateAccuracy(10); // only test 10 batch size
                if (max_acc < acc) {
                    max_acc = acc;
                }
                System.out.println("step_" + (i + 1) + ": \tLoss is " + loss + " [min=" + min_loss + "]" + " max_acc=" + max_acc);
            }
        }
    }
    ```

3. 切换为推理模式，执行推理，评估模型精度。

    ```java
    model.setTrainMode(false);
    for (long i = 0; i < tests; i++) {
        Vector<Integer> labels = fillInputData(test_set, (maxTests == -1));
        if (labels.size() != batchSize) {
            System.err.println("unexpected labels size: " + labels.size() + " batch_size size: " + batchSize);
            System.exit(1);
        }
        model.predict();
        MSTensor outputsv = searchOutputsForSize((int) (batchSize * numOfClasses));
        if (outputsv == null) {
            System.err.println("can not find output tensor with size: " + batchSize * numOfClasses);
            System.exit(1);
        }
        float[] scores = outputsv.getFloatData();
        for (int b = 0; b < batchSize; b++) {
            int max_idx = 0;
            float max_score = scores[(int) (numOfClasses * b)];
            for (int c = 0; c < numOfClasses; c++) {
                if (scores[(int) (numOfClasses * b + c)] > max_score) {
                    max_score = scores[(int) (numOfClasses * b + c)];
                    max_idx = c;
                }
            }
            if (labels.get(b) == max_idx) {
                accuracy += 1.0;
            }
        }
    }
    ```

    推理完成后，如果需要继续训练，需要切换为训练模式。

4. 保存训练模型。

    ```java
    // arg 0: FileName
    // arg 1: quantization type QT_DEFAULT -> 0
    // arg 2: model type MT_TRAIN -> 0
    // arg 3: use default output tensor names
    model.export(trainedFilePath, 0, false, null);
    ```

    模型训练完成后，保存到指定路径，后续可以继续加载运行。
