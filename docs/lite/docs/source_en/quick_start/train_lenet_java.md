# Implement Device Training Based On Java Interface

`Android` `Java` `Whole Process` `Model Loading` `Model Training` `Data Preparation` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Training a LeNet Model (Java)](#training-a-lenet-model-java)
    - [Overview](#overview)
    - [Preparation](#preparation)
        - [Environment Requirements](#environment-requirements)
        - [Downloading MindSpore and Building the Java Package for On-device Training](#downloading-mindspore-and-building-the-java-package-for-on-device-training)
        - [Downloading the Dataset](#downloading-the-dataset)
    - [Deploying an Application](#deploying-an-application)
        - [Running Dependencies](#running-dependencies)
        - [Building and Running](#building-and-running)
    - [Detailed Demo Description](#detailed-demo-description)
        - [Demo Structure](#demo-structure)
        - [Writing On-Device Inference Code](#writing-on-device-inference-code)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/lite/docs/source_en/quick_start/train_lenet_java.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## Overview

This tutorial demonstrates how to use the Java API on MindSpore Lite by building and deploying LeNet of the Java version. The following describes how to train a LeNet model locally and then explains the sample code.

## Preparation

### Environment Requirements

- System environment: Linux x86_64 (Ubuntu 18.04.02LTS is recommended.)

- Software dependencies

    - [Git](https://git-scm.com/downloads) 2.28.0 or later

    - [Maven](https://maven.apache.org/download.cgi) 3.3 or later

    - [OpenJDK](https://openjdk.java.net/install/) 1.8 or later

### Downloading MindSpore and Building the Java Package for On-device Training

Clone the source code and build the Java package for MindSpore Lite training. The `Linux` command is as follows:

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.5
cd mindspore
bash build.sh -I x86_64 -j8
```

Environment requirements and settings about the build, see [Building MindSpore Lite](https://www.mindspore.cn/lite/docs/en/r1.5/use/build.html).
The sample source code used in this tutorial is in the `mindspore/lite/examples/train_lenet_java` directory.

### Downloading the Dataset

The `MNIST` dataset used in this example consists of 10 classes of 28 x 28 pixels grayscale images, where the training set contains 60,000 images, and the test set contains 10,000 images.

> You can download the MNIST dataset from the official website <http://yann.lecun.com/exdb/mnist/>. The four links are used to download training data, training labels, test data, and test labels.

Download and decompress the package to the local host. The decompressed training and test sets are stored in the `/PATH/MNIST_Data/train` and `/PATH/MNIST_Data/test` directories, respectively.

The directory structure is as follows:

```text
MNIST_Data/
├── test
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── train
    ├── train-images-idx3-ubyte
    └── train-labels-idx1-ubyte
```

## Deploying an Application

### Building and Running

1. Go to the directory where the sample project is located and execute the sample project. The commands are as follows:

    ```shell
    cd /codes/mindspore/mindspore/lite/examples/train_lenet_java
    ./prepare_and_run.sh -D /PATH/MNIST_Data/ -r ../../../../output/mindspore-lite-${version}-linux-x64.tar.gz
    ```

    > ../resources/model/lenet_tod.ms is a LeNet training model preconfigured in the sample project. You can also convert it into a LeNet model by referring to [Creating MindSpore Lite Models](https://www.mindspore.cn/lite/docs/en/r1.5/use/converter_train.html).
    >
    > /PATH/MNIST_Data/ is the path of MNIST dataset.

    The command output is as follows:

    ```text
    MindSpore Lite 1.3.0
    ==========Loading Model, Create Train Session=============
    Model path is ../model/lenet_tod.ms
    batch_size: 4
    virtual batch multiplier: 16
    ==========Initing DataSet================
    train data cnt: 60000
    test data cnt: 10000
    ==========Training Model===================
    step_500: Loss is 0.05553353 [min=0.010149269] max_accc=0.9543269
    step_1000: Loss is 0.15295759 [min=0.0018140086] max_accc=0.96594554
    step_1500: Loss is 0.018035552 [min=0.0018140086] max_accc=0.9704527
    step_2000: Loss is 0.029250022 [min=0.0010245014] max_accc=0.9765625
    step_2500: Loss is 0.11875624 [min=7.5288175E-4] max_accc=0.9765625
    step_3000: Loss is 0.046675075 [min=7.5288175E-4] max_accc=0.9765625
    step_3500: Loss is 0.034442786 [min=4.3545474E-4] max_accc=0.97686297
    ==========Evaluating The Trained Model============
    accuracy = 0.9770633
    Trained model successfully saved: ../model/lenet_tod_trained.ms
    ```

## Detailed Demo Description

### Demo Structure

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
│       └── lenet_tod.ms   # LeNet training model
├── src
│   └── main
│       └── java
│           └── com
│               └── mindspore
│                   └── lite
│                       ├── train_lenet
│                       │   ├── DataSet.java      # MNIST dataset processing
│                       │   ├── Main.java         # Main function
│                       │   └── NetRunner.java    # Overall training process

```

### Writing On-Device Inference Code

For details about how to use Java APIs, visit <https://www.mindspore.cn/lite/api/en/r1.5/index.html>.

1. Load the MindSpore Lite model file and build a session.

    ```java
    // arg 0: DeviceType:DT_CPU -> 0
    // arg 1: ThreadNum -> 2
    // arg 2: cpuBindMode:NO_BIND ->  0
    // arg 3: enable_fp16 -> false
    msConfig.init(0, 2, 0, false);
    session = new LiteSession();
    System.out.println("Model path is " + modelPath);
    session = session.createTrainSession(modelPath, msConfig, false);
    session.setupVirtualBatch(virtualBatch, 0.01f, 1.00f);
    ```

2. Switch to training mode, perform cyclic iteration, and train the model.

    ```java
    session.train();
    float min_loss = 1000;
    float max_acc = 0;
    for (int i = 0; i < cycles; i++) {
        for (int b = 0; b < virtualBatch; b++) {
        fillInputData(ds.getTrainData(), false);
        session.runGraph();
        float loss = getLoss();
        if (min_loss > loss) {
            min_loss = loss;
        }
        if ((b == 0) && ((i + 1) % 500 == 0)) {
            float acc = calculateAccuracy(10); // only test 10 batch size
            if (max_acc < acc) {
                max_acc = acc;
            }
            System.out.println("step_" + (i + 1) + ": \tLoss is " + loss + " [min=" + min_loss + "]" + " max_accc=" + max_acc);
        }
    }
}
    ```

3. Switch to inference mode, perform inference, and evaluate the model accuracy.

    ```java
    session.eval();
    for (long i = 0; i < tests; i++) {
        Vector<Integer> labels = fillInputData(test_set, (maxTests == -1));
        if (labels.size() != batchSize) {
            System.err.println("unexpected labels size: " + labels.size() + " batch_size size: " + batchSize);
            System.exit(1);
        }
        session.runGraph();
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

    After the inference is complete, if you need to continue the training, switch to training mode.

4. Save the trained model.

    ```java
    // arg 0: FileName
    // arg 1: model type MT_TRAIN -> 0
    // arg 2: quantization type QT_DEFAULT -> 0
    session.export(trainedFilePath, 0, 0)
    ```

    After the model training is complete, save the model to the specified path. Then, you can continue to load and run the model.
