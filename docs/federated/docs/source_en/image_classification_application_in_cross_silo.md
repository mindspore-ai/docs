# Implementing a Cloud-Slio Federated Image Classification Application (x86)

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_en/image_classification_application_in_cross_silo.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

Based on the type of participating clients, federated learning can be classified into cross-silo federated learning and cross-device federated learning. In a cross-silo federated learning scenario, the clients involved in federated learning are different organizations (e.g., healthcare or finance) or geographically distributed data centers, i.e., training models on multiple data silos. In the cross-device federated learning scenario, the participating clients are a large number of mobile or IoT devices. This framework will describe how to implement an image classification application by using the network LeNet on the MindSpore Federated cross-silo federated framework.

The full script to launch cross-silo federated image classification application can be found [here](https://gitee.com/mindspore/federated/tree/master/example/cross_silo_femnist).

## Downloading the Dataset

This example uses the federated learning dataset `FEMNIST` from [leaf dataset](https://github.com/TalwalkarLab/leaf), which contains 62 different categories of handwritten numbers and letters (numbers 0 to 9, 26 lowercase letters, 26 uppercase letters) with an image size of `28 x 28` pixels . The dataset contains handwritten digits and letters from 3500 users (up to 3500 clients can be simulated to participate in federated learning). The total data volume is 805263, the average amount of data contained per user is 226.83, and the variance of the data volume for all users is 88.94.

You can refer to [Image classfication dataset process](https://www.mindspore.cn/federated/docs/en/master/image_classfication_dataset_process.html ) in steps 1 to 7 to obtain the 3500 user datasets `3500_client_img` in the form of images.

Due to the relatively small amount of data per user in the original 3500 user dataset, it will converge too fast in the cross-silo federated task to obviously reflect the convergence effect of the cross-silo federated framework. The following provides a reference script to merge the specified number of user data into one user to increase the amount of individual user data participating in the cross-silo federated task and better simulate the cross-silo federated framework experiment.

```python
import os
import shutil


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def combine_users(root_data_path, new_data_path, raw_user_num, new_user_num):
    mkdir(new_data_path)
    user_list = os.listdir(root_data_path)
    num_per_user = int(raw_user_num / new_user_num)
    for i in range(new_user_num):
        print(
            "========================== combine the raw {}~{} users to the new user: dataset_{} ==========================".format(
                i * num_per_user, i * num_per_user + num_per_user - 1, i))
        new_user = "dataset_" + str(i)
        new_user_path = os.path.join(new_data_path, new_user)
        mkdir(new_user_path)
        for j in range(num_per_user):
            index = i * new_user_num + j
            user = user_list[index]
            user_path = os.path.join(root_data_path, user)
            tags = os.listdir(user_path)
            print("------------- process the raw user: {} -------------".format(user))
            for t in tags:
                tag_path = os.path.join(user_path, t)
                label_list = os.listdir(tag_path)
                new_tag_path = os.path.join(new_user_path, t)
                mkdir(new_tag_path)
                for label in label_list:
                    label_path = os.path.join(tag_path, label)
                    img_list = os.listdir(label_path)
                    new_label_path = os.path.join(new_tag_path, label)
                    mkdir(new_label_path)

                    for img in img_list:
                        img_path = os.path.join(label_path, img)
                        new_img_name = user + "_" + img
                        new_img_path = os.path.join(new_label_path, new_img_name)
                        shutil.copy(img_path, new_img_path)

if __name__ == "__main__":
    root_data_path = "cross_silo_femnist/femnist/3500_clients_img"
    new_data_path = "cross_silo_femnist/femnist/35_7_client_img"
    raw_user_num = 35
    new_user_num = 7
    combine_users(root_data_path, new_data_path, raw_user_num, new_user_num)
```

where `root_data_path` is the path to the original 3500 user datasets, `new_data_path` is the path to the merged dataset, `raw_user_num` specifies the total number of user datasets to be merged (needs to be <= 3500), and `new_user_num` is used to set the number of users merged by the original datasets. For example, the sample code will select the first 35 users from `cross_silo_femnist/femnist/3500_clients_img`, merge them into 7 user datasets and store them in the path `cross_silo_femnist/femnist/35_7_client_img` (the merged 7 users each contains the original 5 user dataset).

The following print represents a successful merge of the data sets.

```sh
========================== combine the raw 0~4 users to the new user: dataset_0 ==========================
------------- process the raw user: f1798_42 -------------
------------- process the raw user: f2149_81 -------------
------------- process the raw user: f4046_46 -------------
------------- process the raw user: f1093_13 -------------
------------- process the raw user: f1124_24 -------------
========================== combine the raw 5~9 users to the new user: dataset_1 ==========================
------------- process the raw user: f0586_11 -------------
------------- process the raw user: f0721_31 -------------
------------- process the raw user: f3527_33 -------------
------------- process the raw user: f0146_33 -------------
------------- process the raw user: f1272_09 -------------
========================== combine the raw 10~14 users to the new user: dataset_2 ==========================
------------- process the raw user: f0245_40 -------------
------------- process the raw user: f2363_77 -------------
------------- process the raw user: f3596_19 -------------
------------- process the raw user: f2418_82 -------------
------------- process the raw user: f2288_58 -------------
========================== combine the raw 15~19 users to the new user: dataset_3 ==========================
------------- process the raw user: f2249_75 -------------
------------- process the raw user: f3681_31 -------------
------------- process the raw user: f3766_48 -------------
------------- process the raw user: f0537_35 -------------
------------- process the raw user: f0614_14 -------------
========================== combine the raw 20~24 users to the new user: dataset_4 ==========================
------------- process the raw user: f2302_58 -------------
------------- process the raw user: f3472_19 -------------
------------- process the raw user: f3327_11 -------------
------------- process the raw user: f1892_07 -------------
------------- process the raw user: f3184_11 -------------
========================== combine the raw 25~29 users to the new user: dataset_5 ==========================
------------- process the raw user: f1692_18 -------------
------------- process the raw user: f1473_30 -------------
------------- process the raw user: f0909_04 -------------
------------- process the raw user: f1956_19 -------------
------------- process the raw user: f1234_26 -------------
========================== combine the raw 30~34 users to the new user: dataset_6 ==========================
------------- process the raw user: f0031_02 -------------
------------- process the raw user: f0300_24 -------------
------------- process the raw user: f4064_46 -------------
------------- process the raw user: f2439_77 -------------
------------- process the raw user: f1717_16 -------------
```

The following directory structure of the folder `cross_silo_femnist/femnist/35_7_client_img` is as follows:

```text
35_7_client_img  # Merge the 35 users in the FeMnist dataset into 7 client data (each containing 5 user data)
├── dataset_0  # The dataset of Client 0
│   ├── train   # Training dataset
│   │   ├── 0  # Store image data corresponding to category 0
│   │   ├── 1  # Store image data corresponding to category 1
│   │   │        ......
│   │   └── 61  # Store image data corresponding to category 61
│   └── test  # Test dataset，with the same directory structure as train
│              ......
│
└── dataset_6  # The dataset of Client 6
    ├── train   # Training dataset
    │   ├── 0  # Store image data corresponding to category 0
    │   ├── 1  # Store image data corresponding to category 1
    │   │        ......
    │   └── 61  # Store image data corresponding to category 61
    └── test  # Test dataset，with the same directory structure as train
```

## Defining the Network

We choose the relatively simple LeNet network, which has seven layers without the input layer: two convolutional layers, two downsampling layers (pooling layers), and three fully connected layers. Each layer contains a different number of training parameters, as shown in the following figure:

![LeNet5](https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_zh_cn/images/LeNet_5.jpg)

> More information about LeNet network is not described herein. For more details, please refer to <http://yann.lecun.com/exdb/lenet/>.

The network used for this task can be found in the script [test_cross_silo_femnist.py](https://gitee.com/mindspore/federated/tree/master/example/cross_silo_femnist/test_cross_ silo_femnist.py).

For a specific understanding of the network definition process in MindSpore, please refer to [quick start](https://www.mindspore.cn/tutorials/en/master/beginner/quick_start.html#building-network).

## Launching the Cross-Silo Federated Task

### Installing MindSpore and Mindspore Federated

Both source code and downloadable distribution are included. Support CPU and GPU hardware platforms, just choose to install according to the hardware platforms. The installation steps can be found in [MindSpore Installation Guide](https://www.mindspore.cn/install), [Mindspore Federated Installation Guide](https://www.mindspore.cn/federated/docs/en/master/federated_install.html).

Currently the federated learning framework is only supported for deployment in Linux environments. Cross-silo federated learning framework requires MindSpore version number >= 1.5.0.

### Launching the Task

Refer to [Example](https://gitee.com/mindspore/federated/tree/master/example/cross_silo_femnist) to launch cluster. The reference example directory structure is as follows.

```text
cross_silo_femnist/
├── config.json # Configuration file
├── finish_cross_silo_femnist.py # Close the cross-silo federated task script
├── run_cross_silo_femnist_sched.py # Start cross-silo federated scheduler script
├── run_cross_silo_femnist_server.py # Start cross-silo federated server script
├── run_cross_silo_femnist_worker.py # Start cross-silo federated worker script
└── test_cross_silo_femnist.py # Training scripts used by the client
```

1. Start Scheduler

   `run_cross_silo_femnist_sched.py` is a Python script provided for the user to start the `Scheduler` and supports modifying the configuration via argument passing `argparse`. The following command is executed, representing the `Scheduler` that starts this federated learning task with TCP port `5554`.

   ```sh
   python run_cross_silo_femnist_sched.py --scheduler_manage_address=127.0.0.1:5554
   ```

   The following print represents a successful start-up:

   ```sh
   [INFO] FEDERATED(35566,7f4275895740,python):2022-10-09-15:23:22.450.205 [mindspore_federated/fl_arch/ccsrc/scheduler/scheduler.cc:35] Run] Scheduler started successfully.
   [INFO] FEDERATED(35566,7f41f259d700,python):2022-10-09-15:23:22.450.357 [mindspore_federated/fl_arch/ccsrc/common/communicator/http_request_handler.cc:90] Run] Start http server!
   ```

2. Start Server

   `run_cross_silo_femnist_server.py` is a Python script for the user to start a number of `Server`, and supports modify the configuration via argument passing `argparse`. The following command is executed, representing the `Server` that starts this federated learning task, with an http start port of `5555` and a number of `servers` of `4`.

   ```sh
    python run_cross_silo_femnist_server.py --local_server_num=4 --http_server_address=10.113.216.40:5555
   ```

   The above command is equivalent to starting four `Server` processes, each with a federal learning service port of `5555`, `5556`, `5557` and `5558`.

3. Start Worker

   `run_cross_silo_femnist_worker.py` is a Python script for the user to start a number of `worker`, and supports modify the configuration via argument passing `argparse`. The following command is executed, representing the `worker` that starts this federated learning task, with an http start port of `5555` and a number of `worker` of `4`.

   ```sh
   python run_cross_silo_femnist_worker.py --dataset_path=/data_nfs/code/fed_user_doc/federated/tests/st/cross_silo_femnist/35_7_client_img/ --http_server_address=10.113.216.40:5555
   ```

After executing the above three commands, go to the `worker_0` folder in the current directory and check the `worker_0` log with the command `grep -rn "test acc" *` and you will see a print similar to the following:

```sh
local epoch: 0, loss: 3.787421340711655, trian acc: 0.05342741935483871, test acc: 0.075
```

Then it means that cross-silo federated learning is started successfully and `worker_0` is training, other workers can be viewed in a similar way.

Please refer to [yaml configuration notes](https://gitee.com/mindspore/federated/blob/master/docs/federated_server_yaml.md) for the description of parameter configuration in the above script.

### Viewing Log

After successfully starting the task, the corresponding log file will be generated under the current directory `cross_silo_femnist` with the following log file directory structure:

```text
cross_silo_femnist
├── scheduler
│   └── scheduler.log     # Print the log during running scheduler
├── server_0
│   └── server.log        # Print the log during running server_0
├── server_1
│   └── server.log        # Print the log during running server_1
├── server_2
│   └── server.log        # Print the log during running server_2
├── server_3
│   └── server.log        # Print the log during running server_3
├── worker_0
│   ├── ckpt              # Store the aggregated model ckpt obtained by worker_0 at the end of each federation learning iteration
│   │   ├── 0-fl-ms-bs32-0epoch.ckpt
│   │   ├── 0-fl-ms-bs32-1epoch.ckpt
│   │   │
│   │   │              ......
│   │   │
│   │   └── 0-fl-ms-bs32-19epoch.ckpt
│   └── worker.log        # Record the output logs when worker_0 participates in the federated learning task
└── worker_1
    ├── ckpt              # Store the aggregated model ckpt obtained by worker_1 at the end of each federation learning iteration
    │  ├── 1-fl-ms-bs32-0epoch.ckpt
    │  ├── 1-fl-ms-bs32-1epoch.ckpt
    │  │
    │  │                     ......
    │  │
    │  └── 1-fl-ms-bs32-19epoch.ckpt
    └── worker.log        # Record the output logs when worker_1 participates in the federated learning task
```

### Closing the Task

If you want to exit in the middle, the following command is available:

```sh
python finish_cross_silo_femnist.py --redis_port=2345
```

Or wait until the training task is finished and then the cluster will exit automatically, no need to close it manually.

### Results

- Used data:

  The `35_7_client_img/` dataset generated in the `download dataset` section above

- The number of client-side local training epochs: 20

- The total number of cross-silo federated learning iterations: 20

- Results (accuracy of the model on the client's test set after each iteration aggregation)

`worker_0` result:

```sh
worker_0/worker.log:7409:local epoch: 0, loss: 3.787421340711655, trian acc: 0.05342741935483871, test acc: 0.075
worker_0/worker.log:14419:local epoch: 1, loss: 3.725699281115686, trian acc: 0.05342741935483871, test acc: 0.075
worker_0/worker.log:21429:local epoch: 2, loss: 3.5285709657335795, trian acc: 0.19556451612903225, test acc: 0.16875
worker_0/worker.log:28439:local epoch: 3, loss: 3.0393165519160608, trian acc: 0.4889112903225806, test acc: 0.4875
worker_0/worker.log:35449:local epoch: 4, loss: 2.575952764115026, trian acc: 0.6854838709677419, test acc: 0.60625
worker_0/worker.log:42459:local epoch: 5, loss: 2.2081101375296512, trian acc: 0.7782258064516129, test acc: 0.6875
worker_0/worker.log:49470:local epoch: 6, loss: 1.9229739431736557, trian acc: 0.8054435483870968, test acc: 0.69375
worker_0/worker.log:56480:local epoch: 7, loss: 1.7005576549999293, trian acc: 0.8296370967741935, test acc: 0.65625
worker_0/worker.log:63490:local epoch: 8, loss: 1.5248727620766704, trian acc: 0.8407258064516129, test acc: 0.6375
worker_0/worker.log:70500:local epoch: 9, loss: 1.3838803705352127, trian acc: 0.8568548387096774, test acc: 0.7
worker_0/worker.log:77510:local epoch: 10, loss: 1.265225578921041, trian acc: 0.8679435483870968, test acc: 0.7125
worker_0/worker.log:84520:local epoch: 11, loss: 1.167484122101638, trian acc: 0.8659274193548387, test acc: 0.70625
worker_0/worker.log:91530:local epoch: 12, loss: 1.082880981700859, trian acc: 0.8770161290322581, test acc: 0.65625
worker_0/worker.log:98540:local epoch: 13, loss: 1.0097520119572772, trian acc: 0.8840725806451613, test acc: 0.64375
worker_0/worker.log:105550:local epoch: 14, loss: 0.9469810053708015, trian acc: 0.9022177419354839, test acc: 0.7
worker_0/worker.log:112560:local epoch: 15, loss: 0.8907848935604703, trian acc: 0.9022177419354839, test acc: 0.6625
worker_0/worker.log:119570:local epoch: 16, loss: 0.8416629644123349, trian acc: 0.9082661290322581, test acc: 0.70625
worker_0/worker.log:126580:local epoch: 17, loss: 0.798475691030866, trian acc: 0.9122983870967742, test acc: 0.70625
worker_0/worker.log:133591:local epoch: 18, loss: 0.7599438544427897, trian acc: 0.9243951612903226, test acc: 0.6875
worker_0/worker.log:140599:local epoch: 19, loss: 0.7250227383907605, trian acc: 0.9294354838709677, test acc: 0.7125
```

The test results of other clients are basically the same, so the details are not listed herein.
