MindPandas Documentation
==============================

Data processing and analysis is an important part of the AI training process, in which tabular data type is a common form of data representation. The most commonly used data analysis framework in the industry is Pandas, which provides easy-to-use and rich interfaces. However, due to its single-threaded execution mode, Pandas performs poorly when handling large amounts of data. Moreover, because it does not support distribution, it is unable to handle large amounts of data beyond the memory of a single machine. In addition, as the commonly used data analysis framework in the industry is independent of the AI framework such as MindSpore, data needs to go through steps such as disk dropping and format conversion before it can be trained, which greatly affects the use efficiency.

MindSpore is a data analysis framework that is compatible with the Pandas interface and provides a data analysis framework with distributed processing capabilities. MindSpore is dedicated to providing high performance tabular data processing capabilities for large volumes of data. MindSpore can be seamlessly integrated into the training process, enabling MindSpore to support the entire training process of a complete AI model.

The architecture diagram of MindPandas is shown below:

.. raw:: html

    <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/docs/mindpandas/docs/source_en/images/mindpandas_architecture.png" width="700px" alt="" >

1. The top layer provides an API that is compatible with Pandas. You can switch to MindPandas for distributed execution by modifying a small amount of code based on the existing Pandas script.

2. The Distributed Query Compiler converts the API into a distributed combination of basic normal forms (MAP/Reduce/Injective_map, etc.) to ensure the stability of the back-end logic. When new operators are implemented, it can be converted into the existing combination of general computing normal forms.

3. Parallel Execution layer provides two execution modes: multi-threaded mode and multi-process mode. Users can choose the mode according to their actual scenarios.

4. MindPandas slices the original data into multiple internal partitions. Each Partition executes the corresponding operator logic in different threads or processes to achieve parallel data processing.

5. Plug-in operator execution logic is provided at the lowest level. Currently, it mainly supports Pandas operators, and more types of operator logic will be supported in the form of plug-ins later.

Design Features
------------------

1. MindPandas can utilize all cores on the machine

   Compared to the single-threaded implementation of the native Pandas, which can only use one CPU core at any given time. MindPandas can effectively use all cores on a single machine or all cores in a cluster that has multiple machines. The usages are as below:

   .. raw:: html

       <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/docs/mindpandas/docs/source_en/images/mindpandas_multicore.png" width="700px" alt="" >

   MindPandas can be extended to the entire cluster, utilizing the memory as well as CPU resources of the entire cluster. The usages are as below:

   .. raw:: html

       <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/docs/mindpandas/docs/source_en/images/cluster.png" width="700px" alt="" >

2. MindPandas is consistent with the native Pandas API in terms of interface usage. You can run scripts by setting the back-end running mode of MindPandas. Just replace the import of pandas with:

   .. code-block:: python

       # import pandas as pd
       import mindpandas as pd

   Easy to use, to_pandas interface is also compatible with existing Pandas code, in order to achieve excellent performance with small code changes.

MindPandas Performance
------------------------

MindPandas dramatically reduces computation time by slicing the raw data and performing distributed parallel computation base on slicing.

Using read_CSV as an example, an 8-core CPU is used to read a 900MB CSV file. The result is as follows:

Test scenarios:

- CPU: I7-8565U (4 cores, 8 threads)
- memory: 16GB
- data size: 900MB CSV file

======== ====== ==========
API      pandas mindpandas
======== ====== ==========
read_csv 11.53s 5.62s
======== ====== ==========

.. code-block:: python

   import pandas as pd
   import mindpandas as mpd

   # pandas
   df = pd.read_csv("data.csv")

   # MindPandas
   mdf = mpd.read_csv("data.csv")

Other commonly used APIs, such as FillNA, use MindPandas to obtain speedups ranging from several to tens of times.

Test scenarios:

- CPU: I7-8565U (4 cores, 8 threads)
- memory: 16GB
- data size: 800MB (2,000,000 rows \* 48 columns)

====== ====== ==========
API    pandas mindpandas
====== ====== ==========
fillna 0.77s  0.13s
====== ====== ==========

.. code:: python

   import pandas as pd
   import mindpandas as mpd

   df = df.fillna(1)

   # The number of slices can be set according to the actual situation.
   mpd.set_partition_shape((4, 2))
   mdf = mdf.fillna(1)

Common statistical class APIs are also substantially improved in MindPandas by parallelizing performance, such as max, min, sum, all, and any.

Test scenarios:

- CPU: I7-8565U (4 cores, 8 threads)
- memory: 16GB
- Data size: 2GB (10,000,000 rows \* 48 columns)

.. raw:: html

    <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/docs/mindpandas/docs/source_en/images/performance_compare.png" width="700px" alt="" >

With the increase of data size, MindPandas provides more advantages for distributed parallel processing. The following figure shows the performance comparison for different data volumes:

.. raw:: html

    <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/docs/mindpandas/docs/source_en/images/mindpandas_fillna.png" width="700px" alt="" >

Note: MindPandas is set to multiprocess mode and uses a 32-core CPU.

Future Goals
----------------

The initial version of MindPandas contains 100+ APIs，such as DataFrame, Series, Groupby and Other classes, and will add support for more APIs and efficient data flow.

Typical scenarios using MindPandas
---------------------------------------

- Data Processing using MindPandas

  Since MindPandas provides the same interfaces as Pandas, distributed parallel processing of raw data can be performed by replacing the referenced package.

.. toctree::
   :maxdepth: 1
   :caption: Deployment

   mindpandas_install

.. toctree::
   :maxdepth: 1
   :caption: Guide

   mindpandas_quick_start
   mindpandas_configuration

.. toctree::
   :maxdepth: 1
   :caption: API References

   mindpandas.config
   mindpandas.DataFrame
   mindpandas.Series
   mindpandas.Groupby
   mindpandas.Others

.. toctree::
   :maxdepth: 1
   :caption: REFERENCES

   faq

.. toctree::
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE
