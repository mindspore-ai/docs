# Using MindSpore on the Cloud

`Ascend` `GPU` `CPU` `Model Development` `Model Running` `Model Evaluation`

<!-- TOC -->

- [Using MindSpore on the Cloud](#using-mindspore-on-the-cloud)
    - [Overview](#overview)
    - [Preparations](#preparations)
        - [Preparing ModelArts](#preparing-modelarts)
        - [Accessing Ascend AI Processor Resources on HUAWEI CLOUD](#accessing-ascend-ai-processor-resources-on-huawei-cloud)
        - [Preparing Data](#preparing-data)
        - [Preparing for Script Execution](#preparing-for-script-execution)
    - [Running the MindSpore Script on ModelArts After Simple Adaptation](#running-the-mindspore-script-on-modelarts-after-simple-adaptation)
        - [Adapting to Script Arguments](#adapting-to-script-arguments)
        - [Adapting to OBS Data](#adapting-to-obs-data)
        - [Adapting to 8-Device Training Jobs](#adapting-to-8-device-training-jobs)
        - [Sample Code](#sample-code)
    - [Creating a Training Job](#creating-a-training-job)
        - [Opening the ModelArts Console](#opening-the-modelarts-console)
        - [Using a Common Framework to Create a Training Job](#using-a-common-framework-to-create-a-training-job)
        - [Using MindSpore as a Common Framework to Create a Training Job](#using-mindspore-as-a-common-framework-to-create-a-training-job)
    - [Viewing the Execution Result](#viewing-the-execution-result)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/use_on_the_cloud.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

ModelArts is a one-stop AI development platform provided by HUAWEI CLOUD. It integrates the Ascend AI Processor resource pool. Developers can experience MindSpore on this platform.

ResNet-50 is used as an example to describe how to use MindSpore to complete a training task on ModelArts.

## Preparations

### Preparing ModelArts

Create an account, configure ModelArts, and create an Object Storage Service (OBS) bucket by referring to the "Preparations" section in the ModelArts tutorial.
> For more information about ModelArts, visit <https://support.huaweicloud.com/wtsnew-modelarts/index.html>. Prepare ModelArts by referring to the "Preparations" section.

### Accessing Ascend AI Processor Resources on HUAWEI CLOUD

You can click [here](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/dashboard/applyModelArtsAscend910Beta) to join the beta testing program of the ModelArts Ascend Compute Service.

### Preparing Data

ModelArts uses OBS to store data. Therefore, before starting a training job, you need to upload the data to OBS. The CIFAR-10 dataset in binary format is used as an example.

1. Download and decompress the CIFAR-10 dataset.

    > Download the CIFAR-10 dataset at <http://www.cs.toronto.edu/~kriz/cifar.html>. Among the three dataset versions provided on the page, select CIFAR-10 binary version.

2. Create an OBS bucket (for example, ms-dataset), create a data directory (for example, cifar-10) in the bucket, and upload the CIFAR-10 data to the data directory according to the following structure.

    ```text
    └─Object storage/ms-dataset/cifar-10
        ├─train
        │      data_batch_1.bin
        │      data_batch_2.bin
        │      data_batch_3.bin
        │      data_batch_4.bin
        │      data_batch_5.bin
        │
        └─eval
            test_batch.bin
    ```

### Preparing for Script Execution

Create an OBS bucket (for example, `resnet50-train`), create a code directory (for example, `resnet50_cifar10_train`) in the bucket, and upload all scripts in the following directories to the code directory:
> ResNet-50 is used in scripts in <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/sample_for_cloud/> to train the CIFAR-10 dataset and validate the accuracy after training is complete. `1*Ascend` or `8*Ascend` can be used in scripts on ModelArts for training.
>
> Note that the script version must be the same as the MindSpore version selected in "Creating a Training Task." For example, if you use scripts provided for MindSpore 1.1, you need to select MindSpore 1.1 when creating a training job.

To facilitate subsequent training job creation, you need to create a training output directory and a log output directory. The directory structure created in this example is as follows:

```text
└─Object storage/resnet50-train
    ├─resnet50_cifar10_train
    │      dataset.py
    │      resnet.py
    │      resnet50_train.py
    │
    ├─output
    └─log
```

## Running the MindSpore Script on ModelArts After Simple Adaptation

Scripts provided in section "Preparing for Script Execution" can directly run on ModelArts. If you want to experience how to use ResNet-50 to train CIFAR-10, skip this section. If you need to run customized MindSpore scripts or more MindSpore sample code on ModelArts, perform simple adaptation on the MindSpore code as follows:

### Adapting to Script Arguments

1. Set `data_url` and `train_url`. They are necessary for running the script on ModelArts, corresponding to the data storage path (an OBS path) and training output path (an OBS path), respectively.

    ``` python
    import argparse

    parser = argparse.ArgumentParser(description='ResNet-50 train.')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    ```

2. ModelArts allows you to pass arguments to the configuration options in the script. For details, see "Creating a Training Job."

    ``` python
    parser.add_argument('--epoch_size', type=int, default=90, help='Train epoch size.')
    ```

### Adapting to OBS Data

MindSpore does not provide APIs for directly accessing OBS data. You need to use APIs provided by MoXing to interact with OBS. ModelArts training scripts are executed in containers. Generally, the `/cache` directory is used to store the container data.
> HUAWEI CLOUD MoXing provides various APIs for users: <https://github.com/huaweicloud/ModelArts-Lab/tree/master/docs/moxing_api_doc>. In this example, only the `copy_parallel` API is used.

1. Download the data stored in OBS to an execution container.

    ```python
    import moxing as mox
    mox.file.copy_parallel(src_url='s3://dataset_url/', dst_url='/cache/data_path')
    ```

2. Upload the training output from the container to OBS.

    ```python
    import moxing as mox
    mox.file.copy_parallel(src_url='/cache/output_path', dst_url='s3://output_url/')
    ```

### Adapting to 8-Device Training Jobs

To run scripts in the `8*Ascend` environment, you need to adapt dataset creation code and a local data path, and configure a distributed policy. By obtaining the environment variables `DEVICE_ID` and `RANK_SIZE`, you can build training scripts applicable to `1*Ascend` and `8*Ascend`.

1. Adapt a local path.

    ```python
    import os

    device_num = int(os.getenv('RANK_SIZE'))
    device_id = int(os.getenv('DEVICE_ID'))
    # define local data path
    local_data_path = '/cache/data'

    if device_num > 1:
        # define distributed local data path
        local_data_path = os.path.join(local_data_path, str(device_id))
    ```

2. Adapt datasets.

    ```python
    import os
    import mindspore.dataset as ds

    device_id = int(os.getenv('DEVICE_ID'))
    device_num = int(os.getenv('RANK_SIZE'))
    if device_num == 1:
        # create train data for 1 Ascend situation
        dataset = ds.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        # create train data for 1 Ascend situation, split train data for 8 Ascend situation
        dataset = ds.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                    num_shards=device_num, shard_id=device_id)
    ```

3. Configure a distributed policy.

    ```python
    import os
    from mindspore import context
    from mindspore.context import ParallelMode

    device_num = int(os.getenv('RANK_SIZE'))
    if device_num > 1:
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    ```

### Sample Code

Perform simple adaptation on the MindSpore script based on the preceding three points. The following pseudocode is used as an example:

Original MindSpore script:

``` python
import os
import argparse
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.dataset as ds

device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))

def create_dataset(dataset_path):
    if device_num == 1:
        dataset = ds.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        dataset = ds.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                    num_shards=device_num, shard_id=device_id)
    return dataset

def resnet50_train(args):
    if device_num > 1:
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    train_dataset = create_dataset(local_data_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet-50 train.')
    parser.add_argument('--local_data_path', required=True, default=None, help='Location of data.')
    parser.add_argument('--epoch_size', type=int, default=90, help='Train epoch size.')

    args_opt, unknown = parser.parse_known_args()

    resnet50_train(args_opt)
```

Adapted MindSpore script:

``` python
import os
import argparse
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.dataset as ds

# adapt to cloud: used for downloading data
import moxing as mox

device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))

def create_dataset(dataset_path):
    if device_num == 1:
        dataset = ds.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        dataset = ds.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                    num_shards=device_num, shard_id=device_id)
    return dataset

def resnet50_train(args):
    # adapt to cloud: define local data path
    local_data_path = '/cache/data'

    if device_num > 1:
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        # adapt to cloud: define distributed local data path
        local_data_path = os.path.join(local_data_path, str(device_id))

    # adapt to cloud: download data from obs to local location
    print('Download data.')
    mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_path)

    train_dataset = create_dataset(local_data_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet-50 train.')
    # adapt to cloud: get obs data path
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    # adapt to cloud: get obs output path
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    parser.add_argument('--epoch_size', type=int, default=90, help='Train epoch size.')
    args_opt, unknown = parser.parse_known_args()

    resnet50_train(args_opt)
```

## Creating a Training Job

Create a training job to run the MindSpore script. The following provides step-by-step instructions for creating a training job on ModelArts.

### Opening the ModelArts Console

Click Console on the HUAWEI CLOUD ModelArts home page at <https://www.huaweicloud.com/product/modelarts.html>.

### Using a Common Framework to Create a Training Job

ModelArts Tutorial <https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0238.html> shows how to use a common framework to create a training job.

### Using MindSpore as a Common Framework to Create a Training Job

Training scripts and data in this tutorial are used as an example to describe how to configure arguments on the training job creation page.

1. `Algorithm Source`: Click `Frameworks`, and then select `Ascend-Powered-Engine` and the required MindSpore version. (`Mindspore-0.5-python3.7-aarch64` is used as an example here. Use scripts corresponding to the selected version.)

2. `Code Directory`: Select a code directory created in an OBS bucket. Set `Startup File` to a startup script in the code directory.

3. `Data Source`: Click `Data Storage Path` and enter the CIFAR-10 dataset path in OBS.

4. `Argument`: Set `data_url` and `train_url` to the values of `Data Storage Path` and `Training Output Path`, respectively. Click the add icon to pass values to other arguments in the script, for example, `epoch_size`.

5. `Resource Pool`: Click `Public Resource Pool > Ascend`.

6. `Specification`: Select `Ascend: 1 * Ascend 910 CPU: 24-core 96 GiB` or `Ascend: 8 * Ascend 910 CPU: 192-core 768 GiB`, which indicate single-node single-device and single-node 8-device specifications, respectively.

## Viewing the Execution Result

1. You can view run logs on the Training Jobs page.

    The `8*Ascend` specification is used to execute the ResNet-50 training job. The total number of epochs is 92, the accuracy is about 92%, and the number of images trained per second is about 12,000.

    The `1*Ascend` specification is used to execute the ResNet-50 training job. The total number of epochs is 92, the accuracy is about 95%, and the number of images trained per second is about 1800.

2. If you specify a log path when creating a training job, you can download log files from OBS and view them.
