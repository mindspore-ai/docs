# Implementing an Image Classification Application of Cross-device Federated Learning (x86)

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_en/image_classification_application.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

Federated learning can be divided into cross-silo federated learning and cross-device federated learning according to different participating clients. In the cross-silo federated learning scenario, the clients participating in federated learning are different organizations (for example, medical or financial) or data centers geographically distributed, that is, training models on multiple data islands. The clients participating in the cross-device federated learning scenario are a large number of mobiles or IoT devices. This framework will introduce how to use the network LeNet to implement an image classification application on the MindSpore cross-silo federated framework, and provides related tutorials for simulating to start multi-client participation in federated learning in the x86 environment.

Before you start, check whether MindSpore has been correctly installed. If not, install MindSpore on your computer by referring to [Install](https://www.mindspore.cn/install/en) on the MindSpore website.

## Preparatory Work

We provide [Federated Learning Image Classification Dataset FEMNIST](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/federated/3500_clients_bin.zip) and the [device-side model file](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/models/lenet_train.ms) of the `.ms` format for users to use directly. Users can also refer to the following tutorials to generate the datasets and models based on actual needs.

### Generating a Device-side Model File

1. Define the network and training process.

    For the definition of the specific network and training process, please refer to [Beginners Getting Started](https://www.mindspore.cn/tutorials/en/master/beginner/quick_start.html).

2. Export a model as a MindIR file.

    The code snippet is as follows:

    ```python
    import argparse
    import numpy as np
    import mindspore as ms
    import mindspore.nn as nn

    def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
        """weight initial for conv layer"""
        weight = weight_variable()
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            weight_init=weight,
            has_bias=False,
            pad_mode="valid",
        )

    def fc_with_initialize(input_channels, out_channels):
        """weight initial for fc layer"""
        weight = weight_variable()
        bias = weight_variable()
        return nn.Dense(input_channels, out_channels, weight, bias)

    def weight_variable():
        """weight initial"""
        return ms.common.initializer.TruncatedNormal(0.02)

    class LeNet5(nn.Cell):
        def __init__(self, num_class=10, channel=3):
            super(LeNet5, self).__init__()
            self.num_class = num_class
            self.conv1 = conv(channel, 6, 5)
            self.conv2 = conv(6, 16, 5)
            self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
            self.fc2 = fc_with_initialize(120, 84)
            self.fc3 = fc_with_initialize(84, self.num_class)
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

    parser = argparse.ArgumentParser(description="export mindir for lenet")
    parser.add_argument("--device_target", type=str, default="CPU")
    parser.add_argument("--mindir_path", type=str,
                        default="lenet_train.mindir")  # the mindir file path of the model to be export

    args, _ = parser.parse_known_args()
    device_target = args.device_target
    mindir_path = args.mindir_path

    ms.set_context(mode=ms.GRAPH_MODE, device_target=device_target)

    if __name__ == "__main__":
        np.random.seed(0)
        network = LeNet5(62)
        criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction="mean")
        net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
        net_with_criterion = nn.WithLossCell(network, criterion)
        train_network = nn.TrainOneStepCell(net_with_criterion, net_opt)
        train_network.set_train()

        data = ms.Tensor(np.random.rand(32, 3, 32, 32).astype(np.float32))
        label = ms.Tensor(np.random.randint(0, 1, (32, 62)).astype(np.float32))
        ms.export(train_network, data, label, file_name=mindir_path,
                    file_format='MINDIR')  # Add the export statement to obtain the model file in MindIR format.
    ```

    The parameter `--mindir_path` is used to set the path of the generated file in MindIR format.

3. Convert the MindIR file into an .ms file that can be used by the federated learning device-side framework.

    For details about model conversion, see [Training Model Conversion Tutorial](https://www.mindspore.cn/lite/docs/en/master/use/converter_train.html).

    The following is an example of model conversion:

    Assume that the model file to be converted is `lenet_train.mindir`. Run the following command:

    ```sh
    ./converter_lite --fmk=MINDIR --trainModel=true --modelFile=lenet_train.mindir --outputFile=lenet_train
    ```

    If the conversion is successful, the following information is displayed:

    ```sh
    CONVERTER RESULT SUCCESS:0
    ```

    This indicates that the MindSpore model is successfully converted to the MindSpore device-side model and the new file `lenet_train.ms` is generated. If the conversion fails, the following information is displayed:

    ```sh
    CONVERT RESULT FAILED:
    ```

    The generated model file in `.ms` format is the model file required by subsequent clients.

## Simulating Multi-client Participation in Federated Learning

1. Prepare a model file for the client.

    This example uses lenet on the device-side to simulate the actual network used, where[device-side model file](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/models/lenet_train.ms) in `.ms` format of lenet. As the real scenario where a client contains only one model file in .ms format, in the simulation scenario, multiple copies of the .ms file need to be copied and named according to the `lenet_train{i}.ms` format, where i represents the client number, since the .ms file has been automatically copied for each client in `run_client_x86.py`.

    See the copy_ms function in [startup script](https://gitee.com/mindspore/federated/tree/master/example/cross_device_lenet_femnist/simulate_x86/run_client_x86.py) for details.

2. Start the cloud side service.

    Users can first refer to [cloud-side deployment tutorial](https://www.mindspore.cn/federated/docs/en/master/deploy_federated_server.html) to deploy the cloud-side environment and start the cloud-side service.

3. Start the client.

    Before starting the client, please refer to the section [Device-side deployment tutotial](https://www.mindspore.cn/federated/docs/en/master/deploy_federated_client.html) for deployment of device environment.

    We provide a reference script [run_client_x86.py](https://gitee.com/mindspore/mindspore/blob/master/tests/st/fl/cross_device_lenet/client/run_client_x86.py), users can set relevant parameters to start different federated learning interfaces.
    After the cloud-side service is successfully started, the script providing run_client_x86.py is used to call the federated learning framework jar package `mindspore-lite-java-flclient.jar` and the corresponding jar package `quick_start_flclient.jar` of the model script, obtaining in [Compiling package Flow in device-side deployment](https://www.mindspore.cn/federated/docs/en/master/deploy_federated_client.html) to simulate starting multiple clients to participate in the federated learning task.

    Taking the LeNet network as an example, some of the input parameters in the `run_client_x86.py` script have the following meanings, and users can set them according to the actual situation:

    - `--fl_jar_path`

        For setting the federated learning jar package path and obtaining x86 environment federated learning jar package, refer to [Compile package process in device-side deployment](https://www.mindspore.cn/federated/docs/en/master/deploy_federated_client.html).

        Please make sure that only the JAR package is included in the path. For example, in the above reference script, `--jarPath` is set to `"libs/jarX86/mindspore-lite-java-flclient.jar"`, you need to make sure that the `jarX86` folder contains only one JAR package `mindspore-lite-java-flclient.jar`.

    - `--case_jar_path`

        For setting the path of jar package `quick_start_flclient.jar` generated by model script and obtaining the JAR package in the x86 environment, see [Compile package process in device-side deployment](https://www.mindspore.cn/federated/docs/en/master/deploy_federated_client.html).

        Please make sure that only the JAR package is included in the path. For example, in the above reference script, `--case_jarPath` is set to `"case_jar/quick_start_flclient.jar"`, and you need to make sure that the `case_jar` folder contains only one JAR package `quick_start_flclient.jar`.

    - `--train_dataset`

        The root path of the training dataset in which the LeNet image classification task is stored is the training data.bin file and label.bin file for each client, e.g. `data/femnist/3500_clients_bin/`.

    - `--flName`

        Specifies the package path of model script used by federated learning. We provide two types of model scripts for your reference ([Supervised sentiment classification task](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert), [Lenet image classification task](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet)). For supervised sentiment classification tasks, this parameter can be set to the package path of the provided script file [AlBertClient.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert/AlbertClient.java), like as `com.mindspore.flclient.demo.albert.AlbertClient`. For Lenet image classification tasks, this parameter can be set to the package path of the provided script file [LenetClient.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet/LenetClient.java), like as `com.mindspore.flclient.demo.lenet.LenetClient`. At the same time, users can refer to these two types of model scripts, define the model script by themselves, and then set the parameter to the package path of the customized model file ModelClient.java (which needs to inherit from the class [Client.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/java/fl_client/src/main/java/com/mindspore/flclient/model/Client.java)).

    - `--train_model_path`

        Specifies the training model path used for federated learning. The path is the directory where multiple .ms files copied in the preceding tutorial are stored, for example, `ms/lenet`. The path must be an absolute path.

    - `--domain_name`

        Used to set the url for device-cloud communication. Currently, https and http communication are supported, and the corresponding formats are like as: https://......, http://....... When `if_use_elb` is set to true, the format must be: https://127.0.0.1:6666 or http://127.0.0.1:6666, where `127.0.0.1` corresponds to the ip of the machine ip providing cloud-side services (corresponding to the cloud-side parameter `--scheduler_ip`), and `6666` corresponds to the cloud-side parameter `--fl_server_port`.

        Note 1: When this parameter is set to `http://......`, it means that HTTP communication is used, and there may be communication security risks.

        Note 2: When this parameter is set to `https://......`, it means the use of HTTPS communication. At this time, SSL certificate authentication must be performed, and the certificate path needs to be set by the parameter `-cert_path`.

    - `--task`

        Specifies the type of the task to be started. `train` indicates that a training task is started. `inference` indicates that multiple data inference tasks are started. `getModel` indicates that the task for obtaining the cloud model is started. Other character strings indicate that the inference task of a single data record is started. The default value is `train`. The initial model file (.ms file) is not trained. Therefore, you are advised to start the training task first. After the training is complete, start the inference task. (Note that the values of client_num in the two startups must be the same to ensure that the model file used by `inference` is the same as that used by `train`.)

    - `--batch_size`

        Specifies the number of single-step training samples used in federated learning training and inference, that is, batch size. It needs to be consistent with the batch size of the input data of the model.

    - `--client_num`

        Specifies the number of clients. The value must be the same as that of `start_fl_job_cnt` when the server is started. This parameter is not required in actual scenarios.

    If you want to know more about the meaning of other parameters in the `run_client_x86.py` script, you can refer to the comments in the script.

    The basic startup instructions of the federated learning interface are as follows:

    ```sh
    python run_client_x86.py --jarPath="libs/jarX86/mindspore-lite-java-flclient.jar" --case_jarPath="case_jar/quick_start_flclient.jar" --train_dataset="data/femnist/3500_clients_bin/" --test_dataset="null" --vocal_file="null" --ids_file="null" --flName="com.mindspore.flclient.demo.lenet.LenetClient" --train_model_path="ms/lenet/" --infer_model_path="ms/lenet/" --train_ms_name="lenet_train"  --infer_ms_name="lenet_train" --domain_name="http://127.0.0.1:6666" --cert_path="certs/https_signature_certificate/client/CARoot.pem" --use_elb="true" --server_num=4 --client_num=8 --thread_num=1 --server_mode="FEDERATED_LEARNING" --batch_size=32 --task="train"
    ```

    Note that the related path in the startup command must give an absolute path.

    The above commands indicate that eight clients are started to participate in federated learning. If the startup is successful, log files corresponding to the eight clients are generated in the current folder. You can view the log files to learn the running status of each client:

    ```text
    ./
    ├── client_0
    │   └── client.log  # Log file of client 0.
    │           ......
    └── client_7
        └── client.log  # Log file of client 7.
    ```

    For different interfaces and scenarios, you only need to modify specific parameter values according to the meaning of the parameters, such as:

    - Start federated learning and training tasks: SyncFLJob.flJobRun()

        When `--task` in `Basic Start Command` is set to `train`, it means to start the task.

        You can use the command `grep -r "average loss:" client_0/client.log` to view the average loss of each epoch of `client_0` during the training process. It will be printed as follows:

        ```sh
        INFO: <FLClient> ----------epoch:0,average loss:4.1258564 ----------
        ......
        ```

        You can also use the command `grep -r "evaluate acc:" client_0/client.log` to view the verification accuracy of the model after the aggregation in each federated learning iteration for `client_0` . It will be printed like the following:

        ```sh
        INFO: <FLClient> [evaluate] evaluate acc: 0.125
        ......
        ```

    - Start the inference task: SyncFLJob.modelInference()

        When `--task` in `Basic Start Command` is set to `inference`, it means to start the task.

        You can view the inference result of `client_0` through the command `grep -r "the predicted labels:" client_0/client.log`:

        ```sh
        INFO: <FLClient> [model inference] the predicted labels: [0, 0, 0, 1, 1, 1, 2, 2, 2]
        ......
        ```

    - Start the task of obtaining the latest model on the cloud side: SyncFLJob.getModel()

        When `--task` in `Basic Start Command` is set to `inference`, it means to start the task.

        If there is the following content in the log file, it means that the latest model on the cloud side is successfully obtained:

        ```sh
        INFO: <FLClient> [getModel] get response from server ok!
        ```

4. Stop the client process.

    For details, see the [finish.py](https://gitee.com/mindspore/federated/tree/master/example/cross_device_lenet_femnist/simulate_x86/finish.py) script. The details are as follows:

    The command of stopping the client process:

    ```sh
    python finish.py --kill_tag=mindspore-lite-java-flclient
    ```

    The parameter `--kill_tag` is used to search for the keyword to kill the client process. You only need to set the special keyword in `--jarPath`. The default value is `mindspore-lite-java-flclient`, that is, the name of the federated learning JAR package. The user can check whether the process still exists through the command `ps -ef |grep "mindspore-lite-java-flclient"`.

5. Experimental results of 50 clients participating in federated learning and training tasks.

    Currently, the `3500_clients_bin` folder contains data of 3500 clients. This script can simulate a maximum of 3500 clients to participate in federated learning.

    The following figure shows the accuracy of the test dataset for federated learning on 50 clients (set `server_num` to 16).

    ![lenet_50_clients_acc](images/lenet_50_clients_acc.png)

    The total number of federated learning iterations is 100, the number of epochs for local training on the client is 20, and the value of batchSize is 32.

    The test accuracy in the figure refers to the accuracy of each client test dataset on the aggregated model on the cloud for each federated learning iteration:

    AVG: average accuracy of 50 clients in the test dataset for each federated learning iteration.

    TOP5: average accuracy of the 5 clients with the highest accuracy in the test dataset for each federated learning iteration.

    LOW5: average accuracy of the 5 clients with the lowest accuracy in the test dataset for each federated learning iteration.
