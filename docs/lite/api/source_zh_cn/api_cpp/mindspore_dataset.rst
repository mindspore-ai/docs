:gitee_url: https://gitee.com/mindspore/docs


.. _namespace_mindspore__dataset:

mindspore::dataset
============================

该模块提供了加载和处理各种常见数据集的API，目前支持Album、MNIST数据集。

此外，该模块还提供了在加载时对数据进行采样的API。

请注意，Windows平台上还不支持缓存。
请不要在Windows平台上加载和处理数据时使用。

Dataset函数
-----------------

Dataset函数提供各种函数来加载和处理数据集。

Vision
^^^^^^

.. list-table::
    :widths: 15 15 15
    :header-rows: 1

    * - API
      - 描述
      - 重载的API（重载的Sampler）

    * - :doc:`Function mindspore::dataset::Album <../generate/function_mindspore_dataset_Album-1>`
      - 读取和解析Album数据集的源数据集，
        并返回一个 :doc:`Class AlbumDataset <.../generate/classmindspore_dataset_AlbumDataset>` 的对象。
      - :doc:`Function mindspore::dataset::Album (raw ptr Sampler) <../generate/function_mindspore_dataset_Album-2>`

        :doc:`Function mindspore::dataset::Album (reference Sampler) <../generate/function_mindspore_dataset_Album-3>`

    * - :doc:`Function mindspore::dataset::Mnist <../generate/function_mindspore_dataset_Mnist-1>`
      - 读取和解析MNIST数据集的源数据集，
        并返回一个 :doc:`Class MnistDataset <.../generate/classmindspore_dataset_MnistDataset>` 的对象。
      - :doc:`Function mindspore::dataset::Mnist (raw ptr Sampler) <../generate/function_mindspore_dataset_Mnist-2>`

        :doc:`Function mindspore::dataset::Mnist (reference Sampler) <../generate/function_mindspore_dataset_Mnist-3>`

Dataset类
---------------

Dataset类提供了数据集基类的定义。
以及数据集的常见转换操作，如map、shuffle和batch。
还提供了用于获取数据的Iterator的定义。

- :doc:`../generate/classmindspore_dataset_Dataset`
- :doc:`../generate/classmindspore_dataset_Iterator`
- :doc:`../generate/classmindspore_dataset_PullIterator`

Sampler类
---------------

Sampler类提供了采样器的定义，
用来从数据集中选择样本。

- :doc:`../generate/classmindspore_dataset_Sampler`
- :doc:`../generate/classmindspore_dataset_DistributedSampler`
- :doc:`../generate/classmindspore_dataset_PKSampler`
- :doc:`../generate/classmindspore_dataset_RandomSampler`
- :doc:`../generate/classmindspore_dataset_SequentialSampler`
- :doc:`../generate/classmindspore_dataset_SubsetRandomSampler`
- :doc:`../generate/classmindspore_dataset_SubsetSampler`
- :doc:`../generate/classmindspore_dataset_WeightedRandomSampler`

Eager类
-------------

Eager类提供了Execute类的定义。
该类用于在eager模式下对输入张量进行变换（例如视觉/文本）。

- :doc:`../generate/classmindspore_dataset_Execute`

常量
---------

常量提供了一些常用的枚举和常量变量。

- :doc:`../generate/enum_mindspore_dataset_BorderType-1`
- :doc:`../generate/enum_mindspore_dataset_ImageBatchFormat-1`
- :doc:`../generate/enum_mindspore_dataset_ImageFormat-1`
- :doc:`../generate/enum_mindspore_dataset_InterpolationMode-1`
- :doc:`../generate/enum_mindspore_dataset_JiebaMode-1`
- :doc:`../generate/enum_mindspore_dataset_MapTargetDevice-1`
- :doc:`../generate/enum_mindspore_dataset_NormalizeForm-1`
- :doc:`../generate/enum_mindspore_dataset_RelationalOp-1`
- :doc:`../generate/enum_mindspore_dataset_SamplingStrategy-1`
- :doc:`../generate/enum_mindspore_dataset_SentencePieceModel-1`
- :doc:`../generate/enum_mindspore_dataset_ShuffleMode-1`
- :doc:`../generate/enum_mindspore_dataset_SPieceTokenizerLoadType-1`
- :doc:`../generate/enum_mindspore_dataset_SPieceTokenizerOutType-1`
- :doc:`../generate/enum_mindspore_dataset_TensorImpl-1`
- :doc:`../generate/variable_mindspore_dataset_kCfgCallbackTimeout-1`
- :doc:`../generate/variable_mindspore_dataset_kCfgDefaultCacheHost-1`
- :doc:`../generate/variable_mindspore_dataset_kCfgDefaultCachePort-1`
- :doc:`../generate/variable_mindspore_dataset_kCfgDefaultRankId-1`
- :doc:`../generate/variable_mindspore_dataset_kCfgDefaultSeed-1`
- :doc:`../generate/variable_mindspore_dataset_kCfgMonitorSamplingInterval-1`
- :doc:`../generate/variable_mindspore_dataset_kCfgOpConnectorSize-1`
- :doc:`../generate/variable_mindspore_dataset_kCfgParallelWorkers-1`
- :doc:`../generate/variable_mindspore_dataset_kCfgRowsPerBuffer-1`
- :doc:`../generate/variable_mindspore_dataset_kCfgWorkerConnectorSize-1`
- :doc:`../generate/variable_mindspore_dataset_kCVInvalidType-1`
- :doc:`../generate/variable_mindspore_dataset_kDecimal-1`
- :doc:`../generate/variable_mindspore_dataset_kDeMaxDim-1`
- :doc:`../generate/variable_mindspore_dataset_kDeMaxFreq-1`
- :doc:`../generate/variable_mindspore_dataset_kDeMaxRank-1`
- :doc:`../generate/variable_mindspore_dataset_kDeMaxTopk-1`
- :doc:`../generate/variable_mindspore_dataset_kDftAutoNumWorkers-1`
- :doc:`../generate/variable_mindspore_dataset_kDftMetaColumnPrefix-1`
- :doc:`../generate/variable_mindspore_dataset_kDftNumConnections-1`
- :doc:`../generate/variable_mindspore_dataset_kMaxLegalPort-1`
- :doc:`../generate/variable_mindspore_dataset_kMinLegalPort-1`


其他
------

本节包含一些与数据集操作相关的预定义类、工具函数和一些类型定义。

类
^^^^^^^

- :doc:`../generate/classmindspore_dataset_SentencePieceVocab`
- :doc:`../generate/classmindspore_dataset_Slice`
- :doc:`../generate/classmindspore_dataset_SliceOption`
- :doc:`../generate/classmindspore_dataset_TensorTransform`
- :doc:`../generate/classmindspore_dataset_Vocab`

函数
^^^^^^^^^

- :doc:`../generate/function_mindspore_dataset_Schema-1`
- :doc:`../generate/function_mindspore_dataset_SchemaCharIF-1`

定义类型
^^^^^^^^

- :doc:`../generate/typedef_mindspore_dataset_connection_id_type-1`
- :doc:`../generate/typedef_mindspore_dataset_dsize_t-1`
- :doc:`../generate/typedef_mindspore_dataset_MSTensorMap-1`
- :doc:`../generate/typedef_mindspore_dataset_MSTensorMapChar-1`
- :doc:`../generate/typedef_mindspore_dataset_MSTensorVec-1`
- :doc:`../generate/typedef_mindspore_dataset_row_id_type-1`
- :doc:`../generate/typedef_mindspore_dataset_session_id_type-1`
- :doc:`../generate/typedef_mindspore_dataset_uchar-1`


Lite-CV
-------

Lite-CV是一个特殊的库，限制了没有OpenCV实现的图像转换方法。
需要注意的是，这个库只能在MindSpore的lite模式下通过选项 :py:obj:`-n lite_cv` 编译。
有了这个库，与其他链接到OpenCV的库相比，lite包的大小将更小。

类
^^^^^

- :doc:`../generate/classmindspore_dataset_LiteMat`
- :doc:`../generate/classmindspore_dataset_LDataType`

函数
^^^^^^^^^

- :doc:`../generate/function_mindspore_dataset_Affine-1`
- :doc:`../generate/function_mindspore_dataset_Affine-2`
- :doc:`../generate/function_mindspore_dataset_ApplyNms-1`
- :doc:`../generate/function_mindspore_dataset_Canny-1`
- :doc:`../generate/function_mindspore_dataset_Conv2D-1`
- :doc:`../generate/function_mindspore_dataset_ConvertRgbToGray-1`
- :doc:`../generate/function_mindspore_dataset_ConvertTo-1`
- :doc:`../generate/function_mindspore_dataset_Crop-1`
- :doc:`../generate/function_mindspore_dataset_Divide-1`
- :doc:`../generate/function_mindspore_dataset_ExtractChannel-1`
- :doc:`../generate/function_mindspore_dataset_GaussianBlur-1`
- :doc:`../generate/function_mindspore_dataset_GetAffineTransform-1`
- :doc:`../generate/function_mindspore_dataset_GetPerspectiveTransform-1`
- :doc:`../generate/function_mindspore_dataset_GetRotationMatrix2D-1`
- :doc:`../generate/function_mindspore_dataset_HWC2CHW-1`
- :doc:`../generate/function_mindspore_dataset_InitFromPixel-1`
- :doc:`../generate/function_mindspore_dataset_Merge-1`
- :doc:`../generate/function_mindspore_dataset_Multiply-1`
- :doc:`../generate/function_mindspore_dataset_Pad-1`
- :doc:`../generate/function_mindspore_dataset_ResizeBilinear-1`
- :doc:`../generate/function_mindspore_dataset_ResizePreserveARWithFiller-1`
- :doc:`../generate/function_mindspore_dataset_Sobel-1`
- :doc:`../generate/function_mindspore_dataset_Split-1`
- :doc:`../generate/function_mindspore_dataset_SubStractMeanNormalize-1`
- :doc:`../generate/function_mindspore_dataset_Subtract-1`
- :doc:`../generate/function_mindspore_dataset_Transpose-1`
- :doc:`../generate/function_mindspore_dataset_WarpAffineBilinear-1`
- :doc:`../generate/function_mindspore_dataset_WarpPerspectiveBilinear-1`

常量 / 结构体
^^^^^^^^^^^^^^^^^^^^^

- :doc:`../generate/enum_mindspore_dataset_PaddBorderType-1`
- :doc:`../generate/structmindspore_dataset_Point`
