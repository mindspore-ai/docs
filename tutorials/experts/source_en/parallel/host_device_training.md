# Host&Device Heterogeneous

`Ascend` `GPU` `Distributed Parallel` `Whole Process`

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/apply_host_device_training.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

In deep learning, one usually has to deal with the huge model problem, in which the total size of parameters in the model is beyond the device memory capacity. To efficiently train a huge model, one solution is to employ homogeneous accelerators (*e.g.*, Ascend 910 AI Accelerator and GPU) for distributed training. When the size of a model is hundreds of GBs or several TBs,
the number of required accelerators is too overwhelming for people to access, resulting in this solution inapplicable.  One alternative is Host+Device hybrid training. This solution simultaneously leveraging the huge memory in hosts and fast computation in accelerators, is a promisingly
efficient method for addressing huge model problem.

In MindSpore, users can easily implement hybrid training by configuring trainable parameters and necessary operators to run on hosts, and other operators to run on accelerators.
This tutorial introduces how to train [Wide&Deep](https://gitee.com/mindspore/models/tree/master/official/recommend/wide_and_deep) in the Host+Ascend 910 AI Accelerator mode.

## Preliminaries

1. Prepare the model. The Wide&Deep code can be found at: <https://gitee.com/mindspore/models/tree/master/official/recommend/wide_and_deep>, in which `train_and_eval_auto_parallel.py` is the main function for training, `src/` directory contains the model definition, data processing and configuration files, `script/` directory contains the launch scripts in different modes.

2. Prepare the dataset. Please refer the link in [1] to download the dataset, and use the script `src/preprocess_data.py` to transform dataset into MindRecord format.

3. Configure the device information. When performing training in the bare-metal environment, the network information file needs to be configured. This example only employs one accelerator, thus `rank_table_1p_0.json` containing #0 accelerator is configured (about the rank table file, you can refer to [HCCL_TOOL](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)).

## Configuring for Hybrid Training

1. Configure the flag of hybrid training. In the file `default_config.yaml`, change the default value of `host_device_mix` to be `1`:

    ```python
    host_device_mix: 1
    ```

2. Check the deployment of necessary operators and optimizers. In class `WideDeepModel` of file `src/wide_and_deep.py`, check the execution of `EmbeddingLookup` is at host:

    ```python
    self.deep_embeddinglookup = nn.EmbeddingLookup()
    self.wide_embeddinglookup = nn.EmbeddingLookup()
    ```

   In `class TrainStepWrap(nn.Cell)` of file `src/wide_and_deep.py`, check two optimizers are also executed at host:

    ```python
    self.optimizer_w.target = "CPU"
    self.optimizer_d.target = "CPU"
    ```

## Training the Model

In order to save enough log information, use the command `export GLOG_v=1` to set the log level to INFO before executing the script, and add the `-p on` option when compiling MindSpore. For the details about compiling MindSpore, refer to [Compiling MindSpore](https://www.mindspore.cn/install/detail/en?path=install/master/mindspore_ascend_install_source_en.md&highlight=%E7%BC%96%E8%AF%91mindspore).

Use the script `script/run_auto_parallel_train.sh`. Run the command `bash run_auto_parallel_train.sh 1 1 <DATASET_PATH> <RANK_TABLE_FILE>`,
where the first `1` is the number of accelerators, the second `1` is the number of epochs, `DATASET_PATH` is the path of dataset,
and `RANK_TABLE_FILE` is the path of the above `rank_table_1p_0.json` file.

The running log is in the directory of `device_0`, where `loss.log` contains every loss value of every step in the epoch. Here is an example:

```text
epoch: 1 step: 1, wide_loss is 0.6873926, deep_loss is 0.8878349
epoch: 1 step: 2, wide_loss is 0.6442529, deep_loss is 0.8342661
epoch: 1 step: 3, wide_loss is 0.6227323, deep_loss is 0.80273706
epoch: 1 step: 4, wide_loss is 0.6107221, deep_loss is 0.7813441
epoch: 1 step: 5, wide_loss is 0.5937832, deep_loss is 0.75526017
epoch: 1 step: 6, wide_loss is 0.5875453, deep_loss is 0.74038756
epoch: 1 step: 7, wide_loss is 0.5798845, deep_loss is 0.7245408
epoch: 1 step: 8, wide_loss is 0.57553077, deep_loss is 0.7123517
epoch: 1 step: 9, wide_loss is 0.5733629, deep_loss is 0.70278376
epoch: 1 step: 10, wide_loss is 0.566089, deep_loss is 0.6884129
...
```

`test_deep0.log` contains the runtime log.
Search `EmbeddingLookup` in `test_deep0.log`, the following can be found:

```text
[INFO] DEVICE(109904,python3.7):2020-06-27-12:42:34.928.275 [mindspore/ccsrc/device/cpu/cpu_kernel_runtime.cc:324] Run] cpu kernel: Default/network-VirtualDatasetCellTriple/_backbone-NetWithLossClass/network-WideDeepModel/EmbeddingLookup-op297 costs 3066 us.
[INFO] DEVICE(109904,python3.7):2020-06-27-12:42:34.943.896 [mindspore/ccsrc/device/cpu/cpu_kernel_runtime.cc:324] Run] cpu kernel: Default/network-VirtualDatasetCellTriple/_backbone-NetWithLossClass/network-WideDeepModel/EmbeddingLookup-op298 costs 15521 us.
```

The above shows the running time of `EmbeddingLookup` on the host.

Search `FusedSparseFtrl` and `FusedSparseLazyAdam` in `test_deep0.log`, the following can be found:

```text
[INFO] DEVICE(109904,python3.7):2020-06-27-12:42:35.422.963 [mindspore/ccsrc/device/cpu/cpu_kernel_runtime.cc:324] Run] cpu kernel: Default/optimizer_w-FTRL/FusedSparseFtrl-op299 costs 54492 us.
[INFO] DEVICE(109904,python3.7):2020-06-27-12:42:35.565.953 [mindspore/ccsrc/device/cpu/cpu_kernel_runtime.cc:324] Run] cpu kernel: Default/optimizer_d-LazyAdam/FusedSparseLazyAdam-op300 costs 142865 us.
```

The above shows the running time of two optimizers on the host.

## Reference

[1] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He. [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction.](https://doi.org/10.24963/ijcai.2017/239) IJCAI 2017.
