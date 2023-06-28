# Host+Device Hybrid Training

<a href="https://gitee.com/mindspore/docs/blob/r0.6/tutorials/source_en/advanced_use/host_device_training.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

In deep learning, one usually has to deal with the huge model problem, in which the total size of parameters in the model is beyond the device memory capacity. To efficiently train a huge model, one solution is to employ homogenous accelerators (*e.g.*, Ascend 910 AI Accelerator and GPU) for [distributed training](https://www.mindspore.cn/tutorial/en/r0.6/advanced_use/distributed_training.html). When the size of a model is hundreds of GBs or several TBs,
the number of required accelerators is too overwhelming for people to access, resulting in this solution inapplicable.  One alternative is Host+Device hybrid training. This solution simultaneously leveraging the huge memory in hosts and fast computation in accelerators, is a promisingly
efficient method for addressing huge model problem. 

In MindSpore, users can easily implement hybrid training by configuring trainable parameters and necessary operators to run on hosts, and other operators to run on accelerators.
This tutorial introduces how to train [Wide&Deep](https://gitee.com/mindspore/mindspore/tree/r0.6/model_zoo/official/recommend/wide_and_deep) in the Host+Ascend 910 AI Accelerator mode.
## Preliminaries

1. Prepare the model. The Wide&Deep code can be found at: <https://gitee.com/mindspore/mindspore/tree/r0.6/model_zoo/official/recommend/wide_and_deep>, in which `train_and_eval_auto_parallel.py` is the main function for training, 
`src/` directory contains the model definition, data processing and configuration files, `script/` directory contains the launch scripts in different modes.

2. Prepare the dataset. The dataset can be found at: <https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz>. Use the script `src/preprocess_data.py` to transform dataset into MindRecord format.

3. Configure the device information. When performing training in the bare-metal environment, the network information file needs to be configured. This example only employs one accelerator, thus `rank_table_1p_0.json` containing #0 accelerator is configured as follows (you need to check the server's IP first):

    ```json
    {
         "version": "1.0",
         "server_count": "1",
         "server_list": [
             {
                 "server_id":"10.155.170.16",
                 "device": [
                          {"device_id":"0","device_ip":"192.1.113.246","rank_id":"0"}],
                 "host_nic_ip":"reserve"
             }
         ],
         "status": "completed"
     }
    ```

## Configuring for Hybrid Training

1. Configure the flag of hybrid training. In the function `argparse_init` of file `src/config.py`, change the default value of `host_device_mix` to be `1`; change `self.host_device_mix` in function `__init__` of `class WideDeepConfig` to be `1`:
    ```python
    self.host_device_mix = 1
    ```

2. Check placement of necessary operators and optimizers. In class `WideDeepModel` of file `src/wide_and_deep.py`, check the placement of `EmbeddingLookup` is at host:
    ```python
    self.deep_embeddinglookup = nn.EmbeddingLookup()
    self.wide_embeddinglookup = nn.EmbeddingLookup()
    ```
   In `class TrainStepWrap(nn.Cell)` of file `src/wide_and_deep.py`, check two optimizer are also at host:
    ```python
    self.optimizer_w.sparse_opt.add_prim_attr("primitive_target", "CPU")
    self.optimizer_d.sparse_opt.add_prim_attr("primitive_target", "CPU")
    ```

## Training the Model

Use the script `script/run_auto_parallel_train.sh`. Run the command `bash run_auto_parallel_train.sh 1 1 DATASET RANK_TABLE_FILE`,
where the first `1` is the number of accelerators, the second `1` is the number of epochs, `DATASET` is the path of dataset,
and `RANK_TABLE_FILE` is the path of the above `rank_table_1p_0.json` file.

The running log is in the directory of `device_0`, where `loss.log` contains every loss value of every step in the epoch. Here is an example:

```
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

`test_deep0.log` contains the runtime log (This needs to adjust the log level to INFO, and add the `-p on` option when compiling MindSpore).
Search `EmbeddingLookup` in `test_deep0.log`, the following can be found:

```
[INFO] DEVICE(109904,python3.7):2020-06-27-12:42:34.928.275 [mindspore/ccsrc/device/cpu/cpu_kernel_runtime.cc:324] Run] cpu kernel: Default/network-VirtualDatasetCellTriple/_backbone-NetWithLossClass/network-WideDeepModel/EmbeddingLookup-op297 costs 3066 us.
[INFO] DEVICE(109904,python3.7):2020-06-27-12:42:34.943.896 [mindspore/ccsrc/device/cpu/cpu_kernel_runtime.cc:324] Run] cpu kernel: Default/network-VirtualDatasetCellTriple/_backbone-NetWithLossClass/network-WideDeepModel/EmbeddingLookup-op298 costs 15521 us.
```

showing the running time of `EmbeddingLookup` on the host.

Search `FusedSparseFtrl` and `FusedSparseLazyAdam` in `test_deep0.log`, the following can be found:

```
[INFO] DEVICE(109904,python3.7):2020-06-27-12:42:35.422.963 [mindspore/ccsrc/device/cpu/cpu_kernel_runtime.cc:324] Run] cpu kernel: Default/optimizer_w-FTRL/FusedSparseFtrl-op299 costs 54492 us.
[INFO] DEVICE(109904,python3.7):2020-06-27-12:42:35.565.953 [mindspore/ccsrc/device/cpu/cpu_kernel_runtime.cc:324] Run] cpu kernel: Default/optimizer_d-LazyAdam/FusedSparseLazyAdam-op300 costs 142865 us.
```

showing the running time of two optimizers on the host.
