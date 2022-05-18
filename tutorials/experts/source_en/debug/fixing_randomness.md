# Fixed Randomness to Reproduce Run Results of Script

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/debug/fixing_randomness.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

The purpose of fixed randomness is to reproduce the run results of the script and assist in locating the problem. After fixing the randomness, the loss curve produced by the two trainings under the same conditions should be basically the same, and you can perform debugging multiple times to easily find the cause of the loss curve abnormality without worrying about the problem phenomenon of the last debugging and no longer appearing in this run.

Note that even after all fixable randomities have been fixed, it may not be possible to accurately reproduce the results of the run on MindSpore. Especially when the MindSpore version (commit id) is used at the same time, or the machine executing the script is not the same machine, or the AI training accelerator executing the script is not the same physical device, even using the same seed may not be able to reproduce the running results.

After fixing the randomness, there may be a decrease in running performance, so it is recommended that after the problem is fixed, the randomness is unsettled and the relevant script changes are removed so as not to affect the normal running performance of the script.

This document applies to Graph mode on Ascend.

## Steps to Fix the Randomness of the MindSpore Script

1. Insert code at the beginning of the script you want to execute to pin the global random number of seeds.

    Random number of seeds that need to be fixed include MindSpore global random number seeds [mindspore.set_seed(1)](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_seed.html#mindspore.set_seed), numpy and other tripartite libraries of global random number of seeds `numpy.random.seed(1)` , and Python random number of seeds `random.seed(1)` etc. The sample code is as follows:

    ```python
    import random

     import numpy

     import mindspore

     mindspore.set_seed(1)
     numpy.random.seed(1)
     random.seed(1)
    ```

2. Fix hyperparameter.

    It is recommended to specify each hyperparameter with a clear value, and if it involves dynamic learning rate, please ensure that the parameters of the generated dynamic learning rate are determined. Avoid using superparameters with randomness.

3. Fix initialization weights.

    It is recommended to fix the initialization weights by loading a fixed checkpoint file. When loading checkpoint, make sure that the file is fully loaded, and cannot pop out some keys before loading.

4. Fix data processing methods and data order.

    (1) Remove or replace all random data processing operators (for example removing [RandomHorizontalFlip](https://mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.RandomHorizontalFlip.html#mindspore.dataset.vision.c_transforms.RandomHorizontalFlip) and replace [RandomCrop](https://mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.RandomCrop.html#mindspore.dataset.vision.c_transforms.RandomCrop) with [Crop](https://mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.c_transforms.Crop.html#mindspore.dataset.vision.c_transforms.Crop)). Random operators refer to data processing operators with Random in all names.

    (2) Set `shuffle=False` to turn off shuffle. Do not use the sampler of the dataset.

    (3) Set the `num_parallel_workers` parameter to 1 to avoid the effect of parallel data processing on the data order.

    (4) If you need to start training from an iteration, you can use the `dataset.skip()` interface to skip the data from the previous iteration.
    The sample code is as follows:

    ```python
    import mindspore.dataset as ds

     data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=1, shuffle=False)
     data_set.map(operations=trans, input_columns="image", num_parallel_workers=1)
    ```

5. Fix network.

    Remove operators with randomness in the network, such as DropOut operators and operators with Random in the name. If some random operators really can't be removed, you should set a fixed random number seed (random number seed is recommended to choose a number other than 0). DropOut operator randomness is difficult to fix in some scenarios, and it is recommended to always delete it. The known random operators include: [Random Operators](https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html#random-generation-operator), all the DropOut operators, such as [Dropout](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Dropout.html#mindspore.ops.Dropout), [Dropout2D](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Dropout2D.html#mindspore.ops.Dropout2D), [Dropout3D](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Dropout3D.html#mindspore.ops.Dropout3D), [DropoutDoMask](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.DropoutDoMask.html#mindspore.ops.DropoutDoMask), [DropoutGenMask](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.DropoutGenMask.html#mindspore.ops.DropoutGenMask).

    In addition, there are special operators on the Ascend backend that carry a slight randomness when calculating, which does not cause errors in the calculation results, but only causes the calculation results to produce a slight difference between the two calculations that enter the same. For networks containing these special operators, the difference in loss values between the two runs of the script due to error accumulation will increase significantly, and the criteria for judging whether the loss values provided in this article are consistent are not applicable. A list of special operators on the Ascend backend is found at the end of this article.

6. Confirm that the randomness was successfully fixed.

    Run the training script twice in the same environment and check the loss curve to determine whether the randomness is successfully fixed. It is recommended to run the script in non-sink mode to get the loss value for each iteration of the script, and then you can compare the loss value of the first two iterations. The reason why it is not recommended to use the sinking mode is that the loss value of each epoch can generally only be obtained in the sinking mode, because there are many iterations experienced in an epoch, and the accumulation of randomness may make there is a significant gap in the loss value of the epoch granularity of the two runs, which cannot be used as the basis for whether the randomness is fixed or not.

    Successful fixing of randomness requires the following two conditions:

    (1) Run the script twice, and the loss value of the first iteration satisfies atol=1e-3, and [numpy.allclose()](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html) is True. This indicates that the randomness of the forward propagation of the network has been fixed.

    (2) Run the script twice, and the loss value of the second iteration satisfies atol=1e-3, and [numpy.allclose()](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html) is True. This indicates that the randomness of forward and backpropagation of the network has been fixed.

    If the above two conditions cannot be met at the same time, it should be checked whether the above fixed randomness steps are in place. If the operation of fixing randomness has been done, but the script is run twice, and the loss values of the first two iterations are still inconsistent, please [Set New issue to ask MindSpore for help](https://gitee.com/mindspore/mindspore/issues/new).

    We provide a [sample code](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/mindinsight/fix_randomness/fix_randomness.py) that successfully fixed randomness, which performs 2 iterations of training. As can be seen by running this code twice, the loss value of the first iteration of the two trainings satisfies the numpy.allclose() function, and the loss value of the second iteration of the two trainings satisfies the numpy.allclose() function, indicating that the randomness of the network is fixed.

## Notes

1. This document is primarily for the `GRAPH_MODE` training script on the Ascend backend.

2. The list of special operators on the Ascend backend is as follows, which have a slight randomness when calculated:
   - [DynamicGRUV2](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.DynamicGRUV2.html#mindspore.ops.DynamicGRUV2)
   - [DynamicRNN](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.DynamicRNN.html#mindspore.ops.DynamicRNN)
   - [LayerNorm](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.LayerNorm.html#mindspore.ops.LayerNorm)
   - [NLLLoss](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.NLLLoss.html#mindspore.ops.NLLLoss)
   - BNTrainingReduce: When you use the [BatchNorm](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.BatchNorm.html #mindspore.ops.BatchNorm) class operator, the BNTrainingReduce operator is used in forward calculations.
   - BNTrainingReduceGrad: When you use the [BatchNorm](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.BatchNorm.html #mindspore.ops.BatchNorm) class operator, the BNTrainingReduceGrad operator is used in the reverse calculation.
   - Conv2DBackFilter: When you use the [Conv2d](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Conv2D.html#mindspore.ops.Conv2D) operator in the network, the reverse calculation uses the Conv2DBackFilter operator.
   - Conv3DBackFilter: When you use the [Conv3d](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Conv3D.html#mindspore.ops.Conv3D) operator in your network, the reverse calculation uses the Conv3DBackFilter operator.
   - HcomAllreduce: When you use the [AllReduce](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.AllReduce.html #mindspore.ops.AllReduce) operator in your network, the HcomAllreduce operator may be used in forward computations.
   - MaxPool3dGrad: When you use the [MaxPool3D](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.MaxPool3D.html#mindspore.ops.MaxPool3D) operator in the network, the MaxPool3dGrad operator is used in the reverse calculation.
   - ReduceAllD: When you use the [ReduceAll](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ReduceAll.html #mindspore.ops.ReduceAll) operator in your network, the ReduceAllD operator may be used in forward computations.
   - ReduceAnyD: When you use the [ReduceAny](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.ReduceAny.html #mindspore.ops.ReduceAny) operator, the ReduceAnyD operator may be used in forward calculations.
   - ReduceMaxD: When you use the [ReduceMax](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ReduceMax.html #mindspore.ops.ReduceMax) operator in your network, the ReduceMaxD operator may be used in forward computations.
   - ReduceMeanD: When you use the [ReduceMean](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ReduceMean.html #mindspore.ops.ReduceMean) operator in your network, the ReduceMeanD operator may be used in forward calculations.
   - ReduceMinD: When you use the [ReduceMin](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ReduceMin.html #mindspore.ops.ReduceMin) operator in your network, the ReduceMinD operator may be used in forward computations.
   - ReduceProdD: When you use the [ReduceProd](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ReduceProd.html #mindspore.ops.ReduceProd) operator, the ReduceProdD operator may be used in forward calculations.
   - ReduceSum and ReduceSumD: When you use the [ReduceSum](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ReduceSum.html #mindspore.ops.ReduceSum) operator in your network, the ReduceSum or ReduceSum operator may be used in forward calculations.
   - RoiAlignGrad: When you use the [ROIAlign](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ROIAlign.html #mindspore.ops.ROIAlign) operator, the StrikeDSliceGrad operator is used in the reverse calculation.
   - SquareSum: When you use the [SquareSumAll](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.SquareSumAll.html #mindspore.ops.SquareSumAll) operator, the SquareSum operator is used in forward computations.
   - StridedSliceGrad: When you use the [StridedSlice](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.StridedSlice.html#mindspore.ops.StridedSlice) operator, the TridEdSliceGrad operator is used in the reverse calculation.