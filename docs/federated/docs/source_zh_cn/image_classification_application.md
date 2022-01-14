# 实现一个端云联邦的图像分类应用(x86)

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/federated/docs/source_zh_cn/image_classification_application.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

联邦学习根据参与客户的不同可分为云云联邦学习（cross-silo）和端云联邦学习（cross-device）。在云云联邦学习场景中，参与联邦学习的客户是不同的组织（例如，医疗或金融）或地理分布的数据中心，即在多个数据孤岛上训练模型。而在端云联邦学习场景中参与的客户为大量的移动或物联网设备。本框架将介绍如何在MindSpore端云联邦框架上使用网络LeNet实现一个图片分类应用，并提供在x86环境中模拟启动多客户端参与联邦学习的相关教程。

在动手进行实践之前，确保你已经正确安装了MindSpore。如果没有，可以参考[MindSpore安装页面](https://www.mindspore.cn/install)完成安装。

## 准备工作

我们提供了可供用户直接使用的[联邦学习图像分类数据集FEMNIST](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/federated/3500_clients_bin.zip)，以及`.ms`格式的[端侧模型文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/models/lenet_train.ms)。用户也可以根据实际需求，参考以下教程自行生成数据集和模型。

### 数据处理

本示例采用`leaf`数据集中的联邦学习数据集`FEMNIST`， 数据集的具体获取方式可参考文档[端云联邦学习图像分类数据集处理](https://gitee.com/mindspore/mindspore/blob/r1.6/tests/st/fl/cross_device_lenet/client/image_classfication_dataset_process.md)。

用户也可自行定义数据集，注意，数据集必须为`.bin`格式文件，且文件中数据维度必须与网络的输入维度保持一致。

### 生成端侧模型文件

1. 定义网络和训练过程

    具体网络和训练过程的定义可参考[初学入门](https://www.mindspore.cn/tutorials/zh-CN/r1.6/quick_start.html#%E5%88%9B%E5%BB%BA%E6%A8%A1%E5%9E%8B)。

    我们提供了网络定义文件[model.py文件](https://gitee.com/mindspore/mindspore/blob/r1.6/tests/st/fl/mobile/src/model.py)和训练过程定义文件[run_export_lenet](https://gitee.com/mindspore/mindspore/blob/r1.6/tests/st/fl/cross_device_lenet/cloud/run_export_lenet.py)供大家参考。

2. 将模型导出为MindIR格式文件。

    运行脚本`run_export_lenet`获取MindIR格式模型文件，其中代码片段如下：

    ```python
    from mindspore import export
    ...

    parser = argparse.ArgumentParser(description="export mindir for lenet")
    parser.add_argument("--device_target", type=str, default="CPU")
    parser.add_argument("--mindir_path", type=str, default="lenet_train.mindir")  # MindIR格式文件路径
    ...

    for _ in range(epoch):
            data = Tensor(np.random.rand(32, 3, 32, 32).astype(np.float32))
            label = Tensor(np.random.randint(0, 61, (32)).astype(np.int32))
            loss = train_network(data, label).asnumpy()
            losses.append(loss)
            export(train_network, data, label, file_name= mindir_path, file_format='MINDIR')  # 在训练过程中添加export语句获取MindIR格式模型文件
        print(losses)
    ```

    具体运行指令如下：

    ```sh
    python export_lenet_mindir.py --mindir_path="ms/lenet/lenet_train.mindir"
    ```

    参数`--mindir_path`用于设置生成的MindIR格式文件路径。

3. 将MindIR文件转化为联邦学习端侧框架可用的ms文件。

    模型转换可参考[训练模型转换教程](https://www.mindspore.cn/lite/docs/zh-CN/r1.6/use/converter_train.html )。

    模型转换示例如下：

    假设待转换的模型文件为`lenet_train.mindir`，执行如下转换命令：

    ```sh
    ./converter_lite --fmk=MINDIR --trainModel=true --modelFile=lenet_train.mindir --outputFile=lenet_train
    ```

    转换成功输出如下：

    ```sh
    CONVERTER RESULT SUCCESS:0
    ```

    这表明MindSpore模型成功转换为MindSpore端侧模型，并生成了新文件`lenet_train.ms`。如果转换失败输出如下：

    ```sh
    CONVERT RESULT FAILED:
    ```

    将生成的`.ms`格式的模型文件放在某个路径上，在调用联邦学习接口时可设置`FLParameter.trainModelPath`为该模型文件的路径。

## 模拟启动多客户端参与联邦学习

1. 为客户端准备好模型文件。

    由于真实场景一个客户端只包含一个.ms格式的模型文件，在模拟场景中，需要拷贝多份.ms文件，并按照`lenet_train{i}.ms`格式进行命名。其中i代表客户端编号，由于`run_client_x86.py`中代码逻辑，i需要设置为`0, 1, 2, 3, 4, 5 .....`等数字。每个客户端各使用一份.ms文件。

    可参考下面脚本，对原始.ms文件进行拷贝和命名：

    ```python
    import shutil
    import os

    def copy_file(raw_path,new_path,copy_num):
        # Copy the specified number of files from the raw path to the new path
        for i in range(copy_num):
            file_name = "lenet_train" + str(i) + ".ms"
            new_file_path = os.path.join(new_path, file_name)
            shutil.copy(raw_path ,new_file_path)
            print('====== copying ',i, ' file ======')
        print("the number of copy .ms files: ", len(os.listdir(new_path)))

    if __name__ == "__main__":
        raw_path = "lenet_train.ms"
        new_path = "ms/lenet"
        num = 5
        copy_file(raw_path, new_path, num)
    ```

    其中`raw_path`设置原始.ms文件路径，`new_path`设置拷贝的.ms文件需要放置的路径，`num`设置拷贝的份数，一般需要模拟启动客户端的数量。

    比如以上脚本中设置，在路径`ms/lenet`中生成了供5个客户端使用的.ms文件，其目录结构如下：

    ```sh
    ms/lenet
    ├── lenet_train0.ms  # 客户端0使用的.ms文件
    ├── lenet_train1.ms  # 客户端1使用的.ms文件
    ├── lenet_train2.ms  # 客户端2使用的.ms文件
    ├── lenet_train3.ms  # 客户端3使用的.ms文件
    └── lenet_train4.ms  # 客户端4使用的.ms文件
    ```

2. 启动云侧服务

    用户可先参考[云侧部署教程](https://www.mindspore.cn/federated/docs/zh-CN/r1.6/deploy_federated_server.html)部署云侧环境，并启动云侧服务。

3. 启动客户端。

    启动客户端之前请先参照端侧部署教程中[x86环境部分](https://www.mindspore.cn/federated/docs/zh-CN/r1.6/deploy_federated_client.html)进行端侧环境部署。

    我们框架提供了三个类型的联邦学习接口供用户调用，具体的接口介绍可参考[API文件](https://www.mindspore.cn/federated/docs/zh-CN/r1.6/java_api_syncfljob.html)：

    - `SyncFLJob.flJobRun()`

        用于启动客户端参与到联邦学习训练任务中，并获取最终训练好的聚合模型。

    - `SyncFLJob.modelInference()`

        用于获取给定数据集的推理结果。

    - `SyncFLJob.getModel()`

        用于获取云侧最新的模型。

    待云侧服务启动成功之后，可编写一个Python脚本，调用联邦学习框架jar包`mindspore-lite-java-flclient.jar` 和模型脚本对应的jar包`quick_start_flclient.jar`（可参考[端侧部署中编译出包流程](https://www.mindspore.cn/federated/docs/zh-CN/r1.6/deploy_federated_client.html)获取）来模拟启动多客户端参与联邦学习任务。

    我们提供了参考脚本[run_client_x86.py](https://gitee.com/mindspore/mindspore/blob/r1.6/tests/st/fl/cross_device_lenet/client/run_client_x86.py)，可通过相关参数的设置，来启动不同的联邦学习接口。

    以LeNet网络为例，`run_client_x86.py`脚本中部分入参含义如下，用户可根据实际情况进行设置：

    - `--jarPath`

        设置联邦学习jar包路径，x86环境联邦学习jar包获取可参考[端侧部署中编译出包流程](https://www.mindspore.cn/federated/docs/zh-CN/r1.6/deploy_federated_client.html)。

        注意，请确保该路径下仅包含该jar包。例如，在上面示例代码中，`--jarPath`设置为`"jarX86/mindspore-lite-java-flclient.jar"`，则需确保`jarX86`文件夹下仅包含一个jar包`mindspore-lite-java-flclient.jar`。

    - `--case_jarPath`

        设置模型脚本所生成的jar包`quick_start_flclient.jar`的路径，x86环境联邦学习jar包获取可参考[端侧部署中编译出包流程](https://www.mindspore.cn/federated/docs/zh-CN/r1.6/deploy_federated_client.html)。

        注意，请确保该路径下仅包含该jar包。例如，在上面示例代码中，`--case_jarPath`设置为`"case_jar/quick_start_flclient.jar"`，则需确保`case_jar`文件夹下仅包含一个jar包`quick_start_flclient.jar`。

    - `--train_dataset`

        训练数据集root路径，LeNet图片分类任务在该root路径中存放的是每个客户端的训练data.bin文件与label.bin文件，例如`leaf/data/femnist/3500_clients_bin/`。

    - `--flName`

        联邦学习使用的模型脚本包路径。我们提供了两个类型的模型脚本供大家参考（[有监督情感分类任务](https://gitee.com/mindspore/mindspore/tree/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert)、[LeNet图片分类任务](https://gitee.com/mindspore/mindspore/tree/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet)），对于有监督情感分类任务，该参数可设置为所提供的脚本文件[AlBertClient.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert/AlbertClient.java) 的包路径`com.mindspore.flclient.demo.albert.AlbertClient`；对于LeNet图片分类任务，该参数可设置为所提供的脚本文件[LenetClient.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet/LenetClient.java) 的包路径`com.mindspore.flclient.demo.lenet.LenetClient`。同时，用户可参考这两个类型的模型脚本，自定义模型脚本，然后将该参数设置为自定义的模型文件ModelClient.java（需继承于类[Client.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/java/java/fl_client/src/main/java/com/mindspore/flclient/model/Client.java)）的包路径即可。

    - `--train_model_path`

        设置联邦学习使用的训练模型路径，为上面教程中拷贝的多份.ms文件所存放的目录，比如`ms/lenet`，必须为绝对路径。

    - `--train_ms_name`

        设置多客户端训练模型文件名称相同部分，模型文件名需为格式`{train_ms_name}1.ms`，`{train_ms_name}2.ms`， `{train_ms_name}3.ms`  等。

    - `--domain_name`

        用于设置端云通信url，目前，可支持https和http通信，对应格式分别为：https://......、http://......，当`if_use_elb`设置为true时，格式必须为：https://127.0.0.0:6666 或者http://127.0.0.0:6666 ，其中`127.0.0.0`对应提供云侧服务的机器ip（即云侧参数`--scheduler_ip`），`6666`对应云侧参数`--fl_server_port`。

        注意1，当该参数设置为`http://......`时代表使用HTTP通信，可能会存在通信安全风险，请知悉。

        注意2，当该参数设置为`https://......`代表使用HTTPS通信。此时必须进行SSL证书认证，需要通过参数`--cert_path`设置证书路径。

    - `--task`

        用于设置本此启动的任务类型，为`train`代表启动训练任务，为`inference`代表启动多条数据推理任务，为`getModel`代表启动获取云侧模型的任务，设置其他字符串代表启动单条数据推理任务。默认为`train`。由于初始的模型文件(.ms文件)是未训练过的，建议先启动训练任务，待训练完成之后，再启动推理任务（注意两次启动的`client_num`保持一致，以保证`inference`使用的模型文件与`train`保持一致）。

    - `--batch_size`

        设置联邦学习训练和推理时使用的单步训练样本数，即batch size。需与模型的输入数据的batch size保持一致。

    - `--client_num`

        设置client数量， 与启动server端时的`start_fl_job_cnt`保持一致，真实场景不需要此参数。

    若想进一步了解`run_client_x86.py`脚本中其他参数含义，可参考脚本中注释部分。

    联邦学习接口基本启动指令如下：

    ```sh
    python run.py --jarPath="libs/jarX86/mindspore-lite-java-flclient.jar" --case_jarPath="case_jar/quick_start_flclient.jar" --train_dataset="data/femnist/3500_clients_bin/"  --flName="com.mindspore.flclient.demo.lenet.LenetClient" --train_model_path="ms/lenet/ms/"  --train_ms_name="lenet_train.mindir"  --domain_name="http://127.0.0.0:6666"  --client_num=5  --batch_size=32 --task="train"
    ```

    注意，启动指令中涉及路径的必须给出绝对路径。

    以上指令代表启动5个客户端参与联邦学习训练任务，若启动成功，会在当前文件夹生成5个客户端对应的日志文件，查看日志文件内容可了解每个客户端的运行情况：

    ```text
    ./
    ├── client_0
    │   └── client.log  # 客户端0的日志文件
    │           ......
    └── client_4
        └── client.log  # 客户端4的日志文件
    ```

    针对不同的接口和场景，只需根据参数含义，修改特定参数值即可，比如：

    - 启动联邦学习训练任务SyncFLJob.flJobRun()

        当`基本启动指令`中 `--task`设置为`train`时代表启动该任务。

        可通过指令`grep -r "average loss:" client_0/client.log`查看`client_0`在训练过程中每个epoch的平均loss，会有类似如下打印：

        ```sh
        INFO: <FLClient> ----------epoch:0,average loss:4.1258564 ----------
        ......
        ```

        也可通过指令`grep -r "evaluate acc:" client_0/client.log`查看`client_0`在每个联邦学习迭代中聚合后模型的验证精度，会有类似如下打印：

        ```sh
        INFO: <FLClient> [evaluate] evaluate acc: 0.125
        ......
        ```

    - 启动推理任务SyncFLJob.modelInference()

        当`基本启动指令`中 `--task`设置为`inference`时代表启动该任务。

        可通过指令`grep -r "the predicted labels:" client_0/client.log`查看`client_0`的推理结果：

        ```sh
        INFO: <FLClient> [model inference] the predicted labels: [0, 0, 0, 1, 1, 1, 2, 2, 2]
        ......
        ```

    - 启动获取云侧最新模型任务SyncFLJob.getModel()

        当`基本启动指令`中 `--task`设置为`getModel`时代表启动该任务。

        在日志文件中若有如下内容代表获取云侧最新模型成功：

        ```sh
        INFO: <FLClient> [getModel] get response from server ok!
        ```

4. 关闭客户端进程。

    可参考[finish.py](https://gitee.com/mindspore/mindspore/blob/r1.6/tests/st/fl/cross_device_lenet/client/finish.py)脚本，具体如下：

    ```python
    import argparse
    import subprocess
    parser = argparse.ArgumentParser(description="Finish client process")
    # The parameter `--kill_tag` is used to search for the keyword to kill the client process.
    parser.add_argument("--kill_tag", type=str, default="mindspore-lite-java-flclient")
    args, _ = parser.parse_known_args()
    kill_tag = args.kill_tag
    cmd = "pid=`ps -ef|grep " + kill_tag
    cmd += " |grep -v \"grep\" | grep -v \"finish\" |awk '{print $2}'` && "
    cmd += "for id in $pid; do kill -9 $id && echo \"killed $id\"; done"
    subprocess.call(['bash', '-c', cmd])
    ```

    关闭客户端指令如下：

    ```sh
    python finish.py --kill_tag=mindspore-lite-java-flclient
    ```

    其中参数`--kill_tag`用于搜索该关键字对客户端进程进行kill，只需要设置`--jarPath`中的特殊关键字即可。默认为`mindspore-lite-java-flclient`，即联邦学习jar包名。
    用户可通过指令`ps -ef |grep "mindspore-lite-java-flclient"`查看进程是否还存在。

5. 50个客户端参与联邦学习训练任务实验结果。

    目前`3500_clients_bin`文件夹中包含3500个客户端的数据，本脚本最多可模拟3500个客户端参与联邦学习。

    下图给出了50个客户端(设置`server_num`为16)进行联邦学习的测试集精度：

    ![lenet_50_clients_acc](images/lenet_50_clients_acc.png)

    其中联邦学习总迭代数为100，客户端本地训练epoch数为20，batchSize设置为32。

    图中测试精度指对于每个联邦学习迭代，各客户端测试集在云侧聚合后的模型上的精度。

    AVG：对于每个联邦学习迭代，50个客户端测试集精度的平均值。

    TOP5：对于每个联邦学习迭代，测试集精度最高的5个客户端的精度平均值。

    LOW5：对于每个联邦学习迭代，测试集精度最低的5个客户端的精度平均值。
