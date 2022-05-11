:gitee_url: https://gitee.com/mindspore/docs


.. _namespace_mindspore__dataset:

mindspore::dataset
============================


Dataset函数
-----------------

Vision
^^^^^^

.. list-table::
    :widths: 15 15
    :header-rows: 1

    * - API
      - 重载的API（重载的Sampler）

    * - :doc:`Function mindspore::dataset::Album <../generate/function_mindspore_dataset_Album-1>`
      - :doc:`Function mindspore::dataset::Album (raw ptr Sampler) <../generate/function_mindspore_dataset_Album-2>`

        :doc:`Function mindspore::dataset::Album (reference Sampler) <../generate/function_mindspore_dataset_Album-3>`

    * - :doc:`Function mindspore::dataset::Mnist <../generate/function_mindspore_dataset_Mnist-1>`
      - :doc:`Function mindspore::dataset::Mnist (raw ptr Sampler) <../generate/function_mindspore_dataset_Mnist-2>`

        :doc:`Function mindspore::dataset::Mnist (reference Sampler) <../generate/function_mindspore_dataset_Mnist-3>`

Dataset类
---------------

- :doc:`../generate/classmindspore_dataset_Dataset`
- :doc:`../generate/classmindspore_dataset_Iterator`
- :doc:`../generate/classmindspore_dataset_PullIterator`

Sampler类
---------------

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

- :doc:`../generate/classmindspore_dataset_Execute`

常量
---------

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
