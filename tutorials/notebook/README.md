# MindSpore的教程体验

## 环境配置
### Windows和Linux系统配置方法

- 系统版本：Windows 10，Ubuntu 16.04及以上

- 软件配置：[Anaconda](https://www.anaconda.com/products/individual)，Jupyter Notebook

- 语言环境：Python3.7.X 推荐 Python3.7.5

- MindSpore 下载地址：[MindSpore官网下载](https://www.mindspore.cn/versions)，使用Windows系统用户选择Windows-X86版本，使用Linux系统用户选择Ubuntu-X86版本

> MindSpore的[具体安装教程](https://www.mindspore.cn/install/) 


### Jupyter Notebook切换conda环境（Kernel Change）的配置方法

- 首先，增加Jupyter Notebook切换conda环境功能（Kernel Change）

  启动Anaconda Prompt，输入命令：
    ```
    conda install nb_conda
    ```
    > 建议在base环境操作上述命令。

  执行完毕，重启Jupyter Notebook即可完成功能添加。

- 然后，添加conda环境到Jypyter Notebook的Kernel Change中。

  1. 新建一个conda环境，启动Anaconda Prompt，输入命令：
      ```
      conda create -n {env_name} python=3.7.5
      ```
      > env_name可以按照自己想要的环境名称自行命名。
  
  2. 激活新环境，输入命令：
      ```
      conda activate {env_name}
      ```
  3. 安装ipykernel，输入命令：
      ```
      conda install -n {env_name} ipykernel
      ```
      > 如果添加已有环境，只需执行安装ipykernel操作即可。

  执行完毕后，刷新Jupyter notebook页面点击Kernel下拉，选择Kernel Change，就能选择新添加的conda环境。

## notebook说明

| 教程名称                                         |  内容描述
| :-----------                                    |:------   
| [quick_start.ipynb](./quick_start.ipynb)                               | - CPU平台下从数据集到模型验证的全过程解读 <br/> - 体验教程中各功能模块的使用说明 <br/> - 数据集图形化展示 <br/> - 了解LeNet5具体结构和参数作用 <br/> - 学习使用自定义回调函数 <br/> - loss值与训练步数的变化图 <br/> - 模型精度与训练步数的变化图 <br/> -  使用模型应用到手写图片的预测与分类上
| [debugging_in_pynative_mode.ipynb](./debugging_in_pynative_mode.ipynb)                               | - GPU平台下从数据集获取单个数据进行单个step训练的数据变化全过程解读 <br/> - 了解PyNative模式下的调试方法 <br/> - 图片数据在训练过程中的变化情况的图形展示 <br/> - 了解构建权重梯度计算函数的方法 <br/> - 展示1个step过程中权重的变化及数据展示
| [mindinsight_model_lineage_and_data_lineage.ipynb](./mindinsight/debugging_in_pynative_mode.ipynb)                               | - 了解MindSpore中训练数据的采集及展示 <br/> - 学习使用SummaryRecord记录数据 <br/> - 学习使用回调函数SummaryCollector进行数据采集 <br/> - 使用MindInsight进行数据可视化 <br/> - 了解数据溯源和模型溯源的使用方法
| [calculate_and_datagraphic.ipynb](https://gitee.com/mindspore/docs/blob/r0.6/tutorials/notebook/mindinsight/calculate_and_datagraphic.ipynb)                               | - 了解MindSpore中新增可视化功能 <br/> - 学习使用MindInsight可视化看板<br/> - 学习使用查看计算图可视化图的信息的方法<br/> - 学习使用查看数据图中展示的信息的方法 
| [data_loading_enhancement.ipynb](https://gitee.com/mindspore/docs/blob/r0.6/tutorials/notebook/data_loading_enhance/data_loading_enhancement.ipynb)                               | - 学习MindSpore中数据处理和增强的方法 <br/> - 展示数据处理、增强方法的实际操作 <br/> - 对比展示数据处理前和处理后的效果<br/> - 表述在数据处理、增强后的意义
| [loading_dataset.ipynb](https://gitee.com/mindspore/docs/blob/r0.6/tutorials/notebook/loading_dataset.ipynb) | - 学习MindSpore中加载数据集的方法 <br/> - 展示加载常用数据集的方法<br/> - 展示加载MindRecord格式数据集的方法<br/> - 展示加载自定义格式数据集的方法 
| [nlp_application.ipynb](https://gitee.com/mindspore/docs/blob/r0.6/tutorials/notebook/nlp_application.ipynb) | - 展示MindSpore在自然语言处理的应用<br/> - 展示自然语言处理中数据集特定的预处理方法<br/> - 展示如何定义基于LSTM的SentimentNet网络 
| [mindinsight_image_histogram_scalar_tensor.ipynb](https://gitee.com/mindspore/docs/blob/r0.6/tutorials/notebook/mindinsight/mindinsight_image_histogram_scalar_tensor.ipynb)   | - 了解完整的MindSpore深度学习及MindInsight可视化展示的过程 <br/> - 学习使用MindInsight对训练过程中标量、直方图、图像和张量信息进行可视化展示<br/> - 学习使用Summary算子记录标量、直方图、图像和张量信息<br/> - 学习单独对标量、直方图、图像和张量信息进行记录并可视化展示的方法
