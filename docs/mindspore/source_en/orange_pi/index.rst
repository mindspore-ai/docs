Orange Pi Development
===============================

   The development of OrangePi AIpro requires knowledge of the following:

   - `MindSpore <https://www.mindspore.cn/>`_ 
   - `Linux <https://www.runoob.com/linux/linux-tutorial.html>`_ 
   - `Jupyter Notebook <https://jupyter.org/documentation>`_ 

`OrangePi AIpro <http://www.orangepi.org/>`_ adopts the route of Ascend AI technology, specifically 4-core 64-bit processor and AI processor, integrated graph processor.

At present, the system image of OrangePi AIpro development board has been realized with the Ascend MindSpore AI framework pre-installed, and continues to evolve in subsequent version iterations, and currently supports all network models covered by the tutorials on the MindSpore official website. The OrangePi AIpro development board provides developers with the openEuler version and the ubuntu version, both of which are preconfigured with Ascend MindSpore, allowing users to experience the efficient development experience brought by the synergistic optimization of hardware and software. Meanwhile, developers are welcome to customize MindSpore and CANN running environment.

The next tutorials will demonstrate how to build a customized environment based on OrangePi AIpro, how to start Jupyter Lab in OrangePi AIpro, and use handwritten digit recognition as an example of the operations that need to be done to perform online inference based on MindSpore OrangePi AIpro.

The following operations are based on the OrangePi AIpro 8-12TOPS 16G development board, and the 20TOPS development board operates in the same way.

For more examples of OrangePi AIpro development boards based on MindSpore, please refer to: `GitHub link <https://github.com/mindspore-courses/orange-pi-mindspore>`_ 

.. toctree::
   :glob:
   :maxdepth: 1

   environment_setup
   model_infer
   dev_start
