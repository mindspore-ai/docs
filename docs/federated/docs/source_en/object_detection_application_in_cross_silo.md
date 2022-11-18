# Implementing a Cross-Silo Federated Target Detection Application (x86)

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_en/object_detection_application_in_cross_silo.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

Based on the type of participating clients, federated learning can be classified into cross-silo federated learning and cross-device federated learning. In a cross-silo federated learning scenario, the clients involved in federated learning are different organizations (e.g., healthcare or finance) or geographically distributed data centers, i.e., training models on multiple data silos. In the cross-device federated learning scenario, the participating clients are a large number of mobile or IoT devices. This framework will describe how to implement a target detection application by using network Fast R-CNN on MindSpore Federated cross-silo federated framework.

The full script to launch cross-silo federated target detection application can be found [here](https://gitee.com/mindspore/federated/tree/master/example/cross_silo_faster_rcnn).

## Preparation

This tutorial deploy the cross-silo federated target detection task based on the faster_rcnn network provided in MindSpore model_zoo. Please first follow the official [faster_rcnn tutorial and code](https://gitee.com/mindspore/models/tree/master/official/cv/FasterRCNN) to understand the COCO dataset, faster_rcnn network structure, training process and evaluation process first. Since the COCO dataset is open source, please refer to its [official website](https://cocodataset.org/#home) guidelines to download a dataset by yourself and perform dataset slicing (for example, suppose there are 100 clients, the dataset can be sliced into 100 copies, each representing the data held by one client).

Since the original COCO dataset is in json file format, the target detection script provided by cross-silo federated learning framework only supports input data in MindRecord format. You can convert the json file to MindRecord format file according to the following steps.

- Configure the following parameters in the configuration file[default_config.yaml](https://gitee.com/mindspore/federated/blob/master/example/cross_silo_faster_rcnn/default_config.yaml):

    - `mindrecord_dir`

        Used to set the generated MindRecord format file save path. The folder name must be mindrecord_{num} format, and the number num represents the client label number 0, 1, 2, 3, ......

        ```sh
        mindrecord_dir:"./datasets/coco_split/split_100/mindrecord_0"
        ```

    - `instance_set`

        Used to set original json file path.

        ```sh
        instance_set: "./datasets/coco_split/split_100/train_0.json"
        ```

- Run the script [generate_mindrecord.py](https://gitee.com/mindspore/federated/blob/master/example/cross_silo_faster_rcnn/generate_mindrecord.py) to generate MindRecord file according to `train_0.json`, saved in the `mindrecord_dir` path.

## Starting the Cross-Silo Federated Mission

### Installing MindSpore and Mindspore Federated

Including both downloading source code and downloading release version, supporting CPU, GPU, Ascend hardware platforms, just choose to install according to the hardware platforms.  For the installing step, refer to [MindSpore installation](https://www.mindspore.cn/install)， [Mindspore Federated installation](https://www.mindspore.cn/federated/docs/en/master/index.html).

Currently the federated learning framework is only supported for deployment in Linux environments, and cross-silo federated learning framework requires MindSpore version number >= 1.5.0.

## Starting Mission

Refer to [example](https://gitee.com/mindspore/federated/tree/master/example/cross_silo_faster_rcnn) to start the cluster. The reference example directory structure is as follows:

```text
cross_silo_faster_rcnn
├── src
│   ├── FasterRcnn
│   │   ├── __init__.py                  // init file
│   │   ├── anchor_generator.py          // Anchor generator
│   │   ├── bbox_assign_sample.py        // Phase I Sampler
│   │   ├── bbox_assign_sample_stage2.py // Phase II Sampler
│   │   ├── faster_rcnn_resnet.py        // Faster R-CNN network
│   │   ├── faster_rcnn_resnet50v1.py    // Faster R-CNN network taking Resnet50v1.0 as backbone
│   │   ├── fpn_neck.py                  // Feature Pyramid Network
│   │   ├── proposal_generator.py        // Candidate generator
│   │   ├── rcnn.py                      // R-CNN network
│   │   ├── resnet.py                    // Backbone network
│   │   ├── resnet50v1.py                // Resnet50v1.0 backbone network
│   │   ├── roi_align.py                 // ROI aligning network
│   │   └── rpn.py                       // Regional candidate network
│   ├── dataset.py                     // Create and process datasets
│   ├── lr_schedule.py                 // Learning rate generator
│   ├── network_define.py              // Faster R-CNN network definition
│   ├── util.py                        // Routine operation
│   └── model_utils
│           ├── __init__.py                  // init file
│           ├── config.py                    // Obtain .yaml configuration parameter
│           ├── device_adapter.py            // Obtain on-cloud id
│           ├── local_adapter.py             // Get local id
│           └── moxing_adapter.py            // On-cloud data preparation
├── requirements.txt
├── mindspore_hub_conf.py
├── generate_mindrecord.py              // Convert annotations files in .json format to MindRecord format for reading datasets
├── default_config.yaml                 // Network structure, dataset address, configuration file required by fl_plan
├── default.yaml                         // Configuration file required for federated training
├── config.json                         // Configuration file required for disaster recovery
├── run_cross_silo_fasterrcnn_worker.py // Starting cross-silo federated worker script
└── test_fl_fasterrcnn.py               // Training scripts used on the client side
```

1. Note that you can choose whether to record the loss value for each step by setting the parameter `dataset_sink_mode` in the `test_fl_fasterrcnn.py` file.

    ```python
    model.train(config.client_epoch_num, dataset, callbacks=cb)  # Not setting dataset_sink_mode means that only the loss value of the last step in each epoch is recorded, which is the default mode in the code
    model.train(config.client_epoch_num, dataset, callbacks=cb, dataset_sink_mode=False)   # Set dataset_sink_mode=False to record the loss value of each step
    ```

2. Set the following parameters in configuration file [default_config.yaml](https://gitee.com/mindspore/federated/blob/master/example/cross_silo_faster_rcnn/default_config.yaml):

    - `pre_trained`

        Used to set the pre-trained model path (.ckpt format).

        The pre-trained model experimented in this tutorial is a ResNet-50 checkpoint trained on ImageNet 2012. You can use the [resnet50](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) script in ModelZoo to train, and then use src/convert_checkpoint.py to convert the trained resnet50 weight file into a loadable weight file.

3. Start redis

    ```sh
    redis-server --port 2345 --save ""
    ```

4. Start Scheduler

    `run_sched.py` is the Python script used to start `Scheduler` and supports modifying the configuration by passing argument `argparse`. Execute the following command, which represents the `Scheduler` that starts this federated learning task. `--yaml_config` is used to set the yaml file path, and its management ip:port is `10.113.216.40:18019`.

    ```sh
    python run_sched.py --yaml_config="default.yaml" --scheduler_manage_address="10.113.216.40:18019"
    ```

    For the detailed implementation, see [run_sched.py](https://gitee.com/mindspore/federated/blob/master/tests/st/cross_device_cloud/run_sched.py).

    The following print represents a successful starting:

    ```sh
    [INFO] FEDERATED(3944,2b280497ed00,python):2022-10-10-17:11:08.154.878 [mindspore_federated/fl_arch/ccsrc/scheduler/scheduler.cc:35] Run] Scheduler started successfully.
    [INFO] FEDERATED(3944,2b28c5ada700,python):2022-10-10-17:11:08.155.056 [mindspore_federated/fl_arch/ccsrc/common/communicator/http_request_handler.cc:90] Run] Start http server!
    ```

5. Start Server

    `run_server.py` is a Python script for starting a number of `Server`s, and supports modifying the configuration by the passing argument `argparse`. Execute the following command, representing the `Server` that starts this Federated Learning task with a TCP address of `10.113.216.40`, a Federated Learning HTTP service starting port of `6668`, and a number of `Server`s of `4`.

    ```sh
    python run_server.py --yaml_config="default.yaml" --tcp_server_ip="10.113.216.40" --checkpoint_dir="fl_ckpt" --local_server_num=4 --http_server_address="10.113.216.40:6668"
    ```

    The above command is equivalent to starting four `Server` processes, each with a federated learning service port of `6668`, `6669`, `6670` and `6671`, as detailed in [run_server.py](https://gitee.com/mindspore/federated/blob/master/example/cross_device_lenet_femnist/run_server.py).

    The following print represents a successful starting:

    ```sh
    [INFO] FEDERATED(3944,2b280497ed00,python):2022-10-10-17:11:08.154.645 [mindspore_federated/fl_arch/ccsrc/common/communicator/http_server.cc:122] Start] Start http server!
    [INFO] FEDERATED(3944,2b280497ed00,python):2022-10-10-17:11:08.154.725 [mindspore_federated/fl_arch/ccsrc/common/communicator/http_request_handler.cc:85] Initialize] Ev http register handle of: [/d    isableFLS, /enableFLS, /state, /queryInstance, /newInstance] success.
    [INFO] FEDERATED(3944,2b280497ed00,python):2022-10-10-17:11:08.154.878 [mindspore_federated/fl_arch/ccsrc/scheduler/scheduler.cc:35] Run] Scheduler started successfully.
    [INFO] FEDERATED(3944,2b28c5ada700,python):2022-10-10-17:11:08.155.056 [mindspore_federated/fl_arch/ccsrc/common/communicator/http_request_handler.cc:90] Run] Start http server!
    ```

6. Start Worker

    `run_cross_silo_femnist_worker.py` is a Python script for starting a number of `worker`s, and supports modifying the configuration by the passing argument `argparse`. The following instruction is executed, representing the `worker` that starts this federated learning task, and the number of `workers` needed for the federated learning task to proceed properly is `2`.

    ```sh
    python run_cross_silo_fasterrcnn_worker.py --worker_num=2 --dataset_path datasets/coco_split/split_100 --http_server_address=10.113.216.40:6668
    ```

    For the detailed implementation, see [run_cross_silo_femnist_worker.py](https://gitee.com/mindspore/federated/blob/master/example/cross_silo_faster_rcnn/run_cross_silo_fasterrcnn_worker.py).

    As the above command, `--worker_num=2` means starting two clients, and the datasets used by the two clients are `datasets/coco_split/split_100/mindrecord_0` and `datasets/coco_split/split_100/mindrecord_1`. Please prepare the required datasets for the corresponding clients according to the `pre-task preparation` tutorial.

    After executing the above three commands and waiting for a while, go to the `worker_0` folder in the current directory and check the `worker_0` log with the command `grep -rn "\epoch:" *` and you will see a log message similar to the following:

    ```sh
    epoch: 1 step: 1 total_loss: 0.6060338
    ```

    Then it means that cross-silo federated is started successfully and `worker_0` is training. Other workers can be viewed in a similar way.

    Please refer to [yaml configuration notes](https://www.mindspore.cn/federated/docs/en/master/horizontal/federated_server_yaml.html) for the description of parameter configuration in the above script.

### Viewing the Log

After successfully starting the task, the corresponding log file will be generated under the current directory `cross_silo_faster_rcnn`. The log file directory structure is as follows:

```text
cross_silo_faster_rcnn
├── scheduler
│   └── scheduler.log     # Print logs during running scheduler
├── server_0
│   └── server.log        # Print logs during running server_0
├── server_1
│   └── server.log        # Print logs during running server_1
├── server_2
│   └── server.log        # Print logs during running server_2
├── server_3
│   └── server.log        # Print logs during running server_3
├── worker_0
│   ├── ckpt              # Store the aggregated model ckpt obtained by worker_0 at the end of each federated learning iteration
│   │  └── mindrecord_0
│   │      ├── mindrecord_0-fast-rcnn-0epoch.ckpt
│   │      ├── mindrecord_0-fast-rcnn-1epoch.ckpt
│   │      │
│   │      │              ......
│   │      │
│   │      └── mindrecord_0-fast-rcnn-29epoch.ckpt
│   ├──loss_0.log         # Record the loss value of each step in the training process of worker_0
│   └── worker.log        # Record the output logs during worker_0 participation in the federal learning task
└── worker_1
    ├── ckpt              # Store the aggregated model ckpt obtained by worker_1 at the end of each federated learning iteration
    │  └── mindrecord_1
    │      ├── mindrecord_1-fast-rcnn-0epoch.ckpt
    │      ├── mindrecord_1-fast-rcnn-1epoch.ckpt
    │      │
    │      │                     ......
    │      │
    │      └── mindrecord_1-fast-rcnn-29epoch.ckpt
    ├──loss_0.log         # Record the loss value of each step in the training process of worker_1
    └── worker.log        # Record the output logs during worker_1 participation in the federal learning task
```

### Closing the Mission

If you want to exit in the middle, the following command is available:

```sh
python finish_cloud.py --redis_port=2345
```

For the detailed implementation, see [finish_cloud.py](https://gitee.com/mindspore/federated/blob/master/tests/st/cross_device_cloud/finish_cloud.py).

Or when the training task is finished, the cluster exits automatically, no need to close it manually.

### Results

- Use data：

  COCO dataset is split into 100 copies, and the first two copies are taken as two worker datasets respectively

- The number of client-side local training epochs: 1

- Total number of cross-silo federated learning iterations: 30

- Results (recording the loss values during the client-side local training):

  Go to the `worker_0` folder in the current directory, and check the `worker_0` log with the command `grep -rn "\]epoch:" *` to see the loss values output in each step:

  ```sh
  epoch: 1 step: 1 total_loss: 5.249325
  epoch: 1 step: 2 total_loss: 4.0856013
  epoch: 1 step: 3 total_loss: 2.6916502
  epoch: 1 step: 4 total_loss: 1.3917351
  epoch: 1 step: 5 total_loss: 0.8109232
  epoch: 1 step: 6 total_loss: 0.99101084
  epoch: 1 step: 7 total_loss: 1.7741735
  epoch: 1 step: 8 total_loss: 0.9517553
  epoch: 1 step: 9 total_loss: 1.7988946
  epoch: 1 step: 10 total_loss: 1.0213892
  epoch: 1 step: 11 total_loss: 1.1700443
                    .
                    .
                    .
  ```

The histograms of the training loss transformations in each step of worker_1 and worker_2 during the 30 iterations training are as follows, [1] and [2]:

The polygrams of the average loss (the sum of the losses of all the steps in an epoch divided by the number of steps) in each step of worker_1 and worker_2 during the 30 iterations training are as follows, [3] and [4]:

![cross-silo_fastrcnn-2workers-loss.png](https://gitee.com/mindspore/docs/raw/master/docs/federated/docs/source_zh_cn/images/cross-silo_fastrcnn-2workers-loss.png)
