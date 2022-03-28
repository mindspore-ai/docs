:gitee_url: https://gitee.com/mindspore/docs


.. _namespace_mindspore__dataset:

mindspore::dataset
============================

This module provides APIs to load and process various common datasets such as MNIST, CIFAR-10,
CIFAR-100, VOC, COCO, ImageNet, CelebA, CLUE, etc. It also supports datasets in standard format,
including MindRecord, TFRecord, Manifest, etc. Users can also define their own datasets with this module.

Besides, this module provides APIs to sample data while loading.

Notice that cache is not supported on Windows platform yet.
Please do not use it while loading and processing data on Windows.


Dataset Functions
-----------------

**Dataset Functions** provides various functions to load and process datasets.

Vision
^^^^^^

.. list-table::
    :widths: 15 15 15
    :header-rows: 1

    * - API
      - Descriptions
      - Extra overload APIs (Parameter Sampler is overloaded)

    * - :doc:`Function mindspore::dataset::Album <../generate/function_mindspore_dataset_Album-1>`
      - A source dataset for reading and parsing Album dataset
        and return an object of :doc:`Class AlbumDataset <../generate/classmindspore_dataset_AlbumDataset>`.
      - :doc:`Function mindspore::dataset::Album (raw ptr Sampler) <../generate/function_mindspore_dataset_Album-2>`

        :doc:`Function mindspore::dataset::Album (reference Sampler) <../generate/function_mindspore_dataset_Album-3>`

    * - :doc:`Function mindspore::dataset::CelebA <../generate/function_mindspore_dataset_CelebA-1>`
      - A source dataset for reading and parsing CelebA dataset
        and return an object of :doc:`Class CelebADataset <../generate/classmindspore_dataset_CelebADataset>`.
      - :doc:`Function mindspore::dataset::CelebA (raw ptr Sampler) <../generate/function_mindspore_dataset_CelebA-2>`

        :doc:`Function mindspore::dataset::CelebA (reference Sampler) <../generate/function_mindspore_dataset_CelebA-3>`

    * - :doc:`Function mindspore::dataset::Cifar100 <../generate/function_mindspore_dataset_Cifar100-1>`
      - A source dataset for reading and parsing Cifar100 dataset
        and return an object of :doc:`Class Cifar100Dataset <../generate/classmindspore_dataset_Cifar100Dataset>`.
      - :doc:`Function mindspore::dataset::Cifar100 (raw ptr Sampler) <../generate/function_mindspore_dataset_Cifar100-2>`

        :doc:`Function mindspore::dataset::Cifar100 (reference Sampler) <../generate/function_mindspore_dataset_Cifar100-3>`
    
    * - :doc:`Function mindspore::dataset::Cifar10 <../generate/function_mindspore_dataset_Cifar10-1>`
      - A source dataset for reading and parsing Cifar10 dataset
        and return an object of :doc:`Class Cifar10Dataset <../generate/classmindspore_dataset_Cifar10Dataset>`.
      - :doc:`Function mindspore::dataset::Cifar10 (raw ptr Sampler) <../generate/function_mindspore_dataset_Cifar10-2>`

        :doc:`Function mindspore::dataset::Cifar10 (reference Sampler) <../generate/function_mindspore_dataset_Cifar10-3>`

    * - :doc:`Function mindspore::dataset::Coco <../generate/function_mindspore_dataset_Coco-1>`
      - A source dataset for reading and parsing Coco dataset
        and return an object of :doc:`Class CocoDataset <../generate/classmindspore_dataset_CocoDataset>`.
      - :doc:`Function mindspore::dataset::Coco (raw ptr Sampler) <../generate/function_mindspore_dataset_Coco-2>`

        :doc:`Function mindspore::dataset::Coco (reference Sampler) <../generate/function_mindspore_dataset_Coco-3>`

    * - :doc:`Function mindspore::dataset::ImageFolder <../generate/function_mindspore_dataset_ImageFolder-1>`
      - A source dataset for reading images from a tree of directories
        and return an object of :doc:`Class ImageFolderDataset <../generate/classmindspore_dataset_ImageFolderDataset>`.
      - :doc:`Function mindspore::dataset::ImageFolder (raw ptr Sampler) <../generate/function_mindspore_dataset_ImageFolder-2>`

        :doc:`Function mindspore::dataset::ImageFolder (reference Sampler) <../generate/function_mindspore_dataset_ImageFolder-3>`

    * - :doc:`Function mindspore::dataset::Mnist <../generate/function_mindspore_dataset_Mnist-1>`
      - A source dataset for reading and parsing the MNIST dataset
        and return an object of :doc:`Class MnistDataset <../generate/classmindspore_dataset_MnistDataset>`.
      - :doc:`Function mindspore::dataset::Mnist (raw ptr Sampler) <../generate/function_mindspore_dataset_Mnist-2>`

        :doc:`Function mindspore::dataset::Mnist (reference Sampler) <../generate/function_mindspore_dataset_Mnist-3>`

    * - :doc:`Function mindspore::dataset::VOC <../generate/function_mindspore_dataset_VOC-1>`
      - A source dataset for reading and parsing the VOC dataset
        and return an object of :doc:`Class VOCDataset <../generate/classmindspore_dataset_VOCDataset>`.
      - :doc:`Function mindspore::dataset::VOC (raw ptr Sampler) <../generate/function_mindspore_dataset_VOC-2>`

        :doc:`Function mindspore::dataset::VOC (reference Sampler) <../generate/function_mindspore_dataset_VOC-3>`

Text
^^^^

.. list-table::
    :widths: 15 15 15
    :header-rows: 1

    * - API
      - Descriptions
      - Extra overload APIs

    * - :doc:`Function mindspore::dataset::CLUE <../generate/function_mindspore_dataset_CLUE-1>`
      - A source dataset for reading and parsing CLUE dataset
        and return an object of :doc:`Class CLUEDataset <../generate/classmindspore_dataset_CLUEDataset>`.
      - None

Standard Format
^^^^^^^^^^^^^^^

.. list-table::
    :widths: 15 15 15
    :header-rows: 1

    * - API
      - Descriptions
      - Extra overload APIs (Parameter Sampler is overloaded)

    * - :doc:`Function mindspore::dataset::CSV <../generate/function_mindspore_dataset_CSV-1>`
      - A source dataset for reading and parsing comma-separated values (CSV) datasets
        and return an object of :doc:`Class CSVDataset <../generate/classmindspore_dataset_CSVDataset>`.
      - None

    * - :doc:`Function mindspore::dataset::Manifest <../generate/function_mindspore_dataset_Manifest-1>`
      - A source dataset for reading images from a Manifest file
        and return an object of :doc:`Class ManifestDataset <../generate/classmindspore_dataset_ManifestDataset>`.
      - :doc:`Function mindspore::dataset::Manifest (raw ptr Sampler) <../generate/function_mindspore_dataset_Manifest-2>`

        :doc:`Function mindspore::dataset::Manifest (reference Sampler) <../generate/function_mindspore_dataset_Manifest-3>`

    * - :doc:`Function mindspore::dataset::MindData <../generate/function_mindspore_dataset_MindData-1>`
      - A source dataset for reading and parsing single MindRecord file
        and return an object of :doc:`Class MindDataDataset <../generate/classmindspore_dataset_MindDataDataset>`.
      - :doc:`Function mindspore::dataset::MindData (raw ptr Sampler) <../generate/function_mindspore_dataset_MindData-2>`

        :doc:`Function mindspore::dataset::MindData (reference Sampler) <../generate/function_mindspore_dataset_MindData-3>`

    * - :doc:`Function mindspore::dataset::MindData <../generate/function_mindspore_dataset_MindData-4>`
      - A source dataset for reading and parsing multiple MindRecord files
        and return an object of :doc:`Class MindDataDataset <../generate/classmindspore_dataset_MindDataDataset>`.
      - :doc:`Function mindspore::dataset::MindData (raw ptr Sampler) <../generate/function_mindspore_dataset_MindData-5>`

        :doc:`Function mindspore::dataset::MindData (reference Sampler) <../generate/function_mindspore_dataset_MindData-6>`
        
    * - :doc:`Function mindspore::dataset::RandomData <../generate/function_mindspore_dataset_RandomData-1>`
      - A source dataset for generating random data
        and return an object of :doc:`Class RandomDataDataset <../generate/classmindspore_dataset_RandomDataDataset>`.
      - None

    * - :doc:`Function mindspore::dataset::TextFile <../generate/function_mindspore_dataset_TextFile-1>`
      - A source dataset for reading and parsing datasets stored on disk in text format
        and return an object of :doc:`Class TextFileDataset <../generate/classmindspore_dataset_TextFileDataset>`.
      - None

    * - :doc:`Function mindspore::dataset::TFRecord <../generate/function_mindspore_dataset_TFRecord-1>`
      - A source dataset for reading and parsing datasets stored on disk in TFData format
        and return an object of :doc:`Class TFRecordDataset <../generate/classmindspore_dataset_TFRecordDataset>`.
      - None

Dataset Classes
---------------

**Dataset Classes** provides the definition of base class of dataset
and common transform operations of dataset such as map, shuffle and batch.
It also provides the definition of Iterator for fetching data.

- :doc:`../generate/classmindspore_dataset_Dataset`
- :doc:`../generate/classmindspore_dataset_Iterator`
- :doc:`../generate/classmindspore_dataset_PullIterator`

Sampler Classes
---------------

**Sampler Classes** provides the definitions of samplers,
which are used to choose samples from the dataset.

- :doc:`../generate/classmindspore_dataset_Sampler`
- :doc:`../generate/classmindspore_dataset_DistributedSampler`
- :doc:`../generate/classmindspore_dataset_PKSampler`
- :doc:`../generate/classmindspore_dataset_RandomSampler`
- :doc:`../generate/classmindspore_dataset_SequentialSampler`
- :doc:`../generate/classmindspore_dataset_SubsetRandomSampler`
- :doc:`../generate/classmindspore_dataset_SubsetSampler`
- :doc:`../generate/classmindspore_dataset_WeightedRandomSampler`

Eager Classes
-------------

**Eager Classes** provides the definitions of Execute class,
which is used to apply transforms (e.g. vision/text) on input tensor in eager mode.

- :doc:`../generate/classmindspore_dataset_Execute`

Constants
---------

**Constants** provides some common enums and const variables.

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


Others
------

This section contains some predefined classes related to Dataset operations, tool functions, and some Typedefs.

Classes
^^^^^^^

- :doc:`../generate/classmindspore_dataset_SentencePieceVocab`
- :doc:`../generate/classmindspore_dataset_Slice`
- :doc:`../generate/classmindspore_dataset_SliceOption`
- :doc:`../generate/classmindspore_dataset_TensorTransform`
- :doc:`../generate/classmindspore_dataset_Vocab`

Functions
^^^^^^^^^

- :doc:`../generate/function_mindspore_dataset_Schema-1`
- :doc:`../generate/function_mindspore_dataset_SchemaCharIF-1`
- :doc:`../generate/function_mindspore_dataset_CreateDatasetCache-1`
- :doc:`../generate/function_mindspore_dataset_CreateDatasetCacheCharIF-1`

Typedefs
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

**Lite-CV** is a special library contrains image transform methods which are implemented without OpenCV.
Note that this library will only be compiled in lite mode of MindSpore with option :py:obj:`-n lite_cv`.
With this library, the size of lite package will be smaller compared to other libraries links to OpenCV.

Class
^^^^^

- :doc:`../generate/classmindspore_dataset_LiteMat`
- :doc:`../generate/classmindspore_dataset_LDataType`

Functions
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

Constants / Structure
^^^^^^^^^^^^^^^^^^^^^

- :doc:`../generate/enum_mindspore_dataset_PaddBorderType-1`
- :doc:`../generate/structmindspore_dataset_Point`
