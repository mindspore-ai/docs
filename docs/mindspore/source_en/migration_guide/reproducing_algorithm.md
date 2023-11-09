# Reproducing Algorithm Implementation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/migration_guide/reproducing_algorithm.md)

## Obtaining Sample Code

When you obtain a paper to implement migration on MindSpore, you need to find the reference code that has been implemented in other frameworks. In principle, the reference code must meet at least one of the following requirements:

1. The author opens the paper to the public.
2. The implementation is starred and forked by many developers, which means it is widely recognized.
3. The code is new and maintained by developers.
4. The PyTorch reference code is preferred.

If the results are not reproducible in the reference project or the version information is missing, check the project issue for information.

If a new paper has no reference implementation, you can refer to [Constructing MindSpore Network](https://www.mindspore.cn/docs/en/r2.3/migration_guide/model_development/model_development.html).

## Analyzing Algorithm and Network Structure

First, when reading the paper and reference code, you need to analyze the network structure to organize the code writing. The following shows the general network structure of YOLOX.

| Module| Implementation|
| ---- | ---- |
| backbone | CSPDarknet (s, m, l, x)|
| neck | FPN |
| head | Decoupled Head |

Second, analyze the innovative points of the migration algorithm and record the tricks used during the training, for example, data augmentation added during data processing, shuffle, optimizer, learning rate attenuation policy, and parameter initialization. You can prepare a checklist and fill in the corresponding items during analysis.

For example, the following records some tricks used by the YOLOX network during training.

<table>
    <tr>
        <th>Trick</th>
        <th>Record</th>
   </tr>
    <tr>
        <td rowspan="2">Data augmentation</td>
        <td >Mosaic, including random scaling, crop, and layout</td>
    </tr>
    <tr>
        <td >MixUp</td>
    </tr>
    <tr>
        <td >Learning rate attenuation policy</td>
        <td >Multiple attenuation modes are available. By default, the COS learning rate attenuation is used. </td>
    </tr>
    <tr>
        <td >Optimizer parameters</td>
        <td >SGD momentum=0.9, nesterov=True, and no weight decay</td>
    </tr>
    <tr>
        <td >Training parameters</td>
        <td >epoch: 300; batchsize: 8</td>
    </tr>
    <tr>
        <td >Network structure optimization points</td>
        <td >Decoupled Head; Anchor Free; SimOTA</td>
    </tr>
    <tr>
        <td >Training process optimization points </td>
        <td >EMA; Data augmentation is not performed for the last 15 epochs; mixed precision </td>
    </tr>
</table>

**Note that the tricks used in the code are mainly reproduced. The tricks mentioned in some papers may not be useful.**

In addition, you need to determine whether the paper can be implemented by modifying the existing MindSpore model. If yes, you can greatly reduce the development workload. For example, WGAN-PG can be developed based on WGAN.
[MindSpore models](https://gitee.com/mindspore/models) is a model repository. It covers mainstream models in multiple fields, such as machine vision, natural language processing, voice, and recommendation system. You can check whether there are required models from the repository.

## Reproducing Paper Implementation

After obtaining the reference code, you need to reproduce the accuracy of the reference implementation and obtain the performance data of the reference implementation. This has the following advantages:

1. Identify some issues in advance.

    - Check whether the third-party repository used by the reference code depends on a version to identify version adaptation problems in advance.
    - Check whether the dataset can be obtained. Some datasets are private or the author adds some datasets to the public dataset. This problem can be found at the reproduction reference implementation stage.
    - Check whether the reference implementation can reproduce the accuracy of the paper. Some official reference implementations may not reproduce the accuracy of the paper. In this case, detect the problem in time, replace the reference implementation, or adjust the accuracy baseline.

2. Obtain some reference data for the MindSpore migration process.

    - Obtain the loss decrease trend to check whether the training convergence trend on MindSpore is normal.
    - Obtain the parameter file for conversion and inference verification. For details, see [Inference and Training Process](https://www.mindspore.cn/docs/en/r2.3/migration_guide/model_development/training_and_evaluation.html).
    - Obtain the performance baseline for performance tuning. For details, see [Debugging and Tuning](https://www.mindspore.cn/docs/en/r2.3/migration_guide/debug_and_tune.html).
