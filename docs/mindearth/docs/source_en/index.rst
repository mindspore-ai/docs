MindSpore Earth Introduction
=============================

Weather phenomena are closely related to human production and life, social economy, military activities and other aspects. Accurate weather forecasts can mitigate the impact of disaster weather events, avoid economic losses, and generate continuous fiscal revenue, such as energy, agriculture, transportation and entertainment industries. At present, the weather forecast mainly adopts numerical weather prediction models, which processes the observation data collected by meteorological satellites, observation stations and radars, solves the atmospheric dynamic equations describing weather evolution, and then provides weather and climate prediction information. The prediction process of numerical prediction model involves a lot of computation, which consumes a long time and a large amount of computation resources. Compared with the numerical prediction model, the data-driven deep learning model can effectively reduce the computational cost by several orders of magnitude.

`MindSpore Earth <https://gitee.com/mindspore/mindscience/tree/master/MindEarth>`_ is an earth science suite developed based on MindSpore. It supports AI meteorological prediction of short-term, medium-term, and long-term weather and catastrophic weather such as precipitation and typhoon. The aim is to provide efficient and easy-to-use AI meteorological prediction software for industrial scientific research engineers, college teachers and students.

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindearth/docs/source_en/images/mindearth_archi_en.png" width="1200px" alt="" style="display: inline-block">

Code repository address: <https://gitee.com/mindspore/mindscience/tree/master/MindEarth>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Installation Deployment

   mindearth_install

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Dem-Super-resolution

   dem-super-resolution/DEM-SRNet

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Medium-range

   medium-range/FourCastNet
   medium-range/graphcast
   medium-range/vit_kno
   medium-range/fuxi
   medium-range/graphcast_tp

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: NowCasting

   nowcasting/DgmrNet
   nowcasting/Nowcastnet
   nowcasting/prediffnet

.. toctree::
   :maxdepth: 1
   :caption: API Reference

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
