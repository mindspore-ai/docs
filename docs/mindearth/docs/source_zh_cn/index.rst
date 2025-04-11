MindSpore Earth介绍
=====================

天气现象与人类的生产生活、社会经济、军事活动等方方面面都密切相关，准确的天气预报能够在灾害天气事件中减轻影响、避免经济损失，还能创造持续不断地财政收入，例如能源、农业、交通和娱乐行业。目前，天气预报主要采用数值天气预报模式，通过处理由气象卫星、观测台站、雷达等收集到的观测资料，求解描写天气演变的大气动力学方程组，进而提供天气气候的预测信息。数值预报模式的预测过程涉及大量计算，耗费较长时间与较大的计算资源。相较于数值预报模式，数据驱动的深度学习模型能够有效地将计算成本降低数个量级。

`MindSpore Earth <https://gitee.com/mindspore/mindscience/tree/master/MindEarth>`_ 是基于昇思MindSpore开发的地球科学领域套件，支持短临、中期、长期等多时空尺度以及降水、台风等灾害性天气的AI气象预测，旨在于为广大的工业界科研工程人员、高校老师及学生提供高效易用的AI气象预测软件。

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindearth/docs/source_zh_cn/images/mindearth_archi_cn.png" width="1200px" alt="" style="display: inline-block">

代码仓地址: <https://gitee.com/mindspore/mindscience/tree/master/MindEarth>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 安装部署

   mindearth_install

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 高分辨率数字高程模型

   dem-super-resolution/DEM-SRNet

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 中期预报

   medium-range/FourCastNet
   medium-range/graphcast
   medium-range/vit_kno
   medium-range/fuxi
   medium-range/graphcast_tp

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 短临预报

   nowcasting/DgmrNet
   nowcasting/Nowcastnet
   nowcasting/prediffnet

.. toctree::
   :maxdepth: 1
   :caption: API参考

   mindearth.cell
   mindearth.core
   mindearth.data
   mindearth.module
   mindearth.utils

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE
