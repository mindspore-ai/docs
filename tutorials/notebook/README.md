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

    ```bash
    conda install nb_conda
    ```

    > 建议在base环境操作上述命令。

  执行完毕，重启Jupyter Notebook即可完成功能添加。

- 然后，添加conda环境到Jypyter Notebook的Kernel Change中。

  1. 新建一个conda环境，启动Anaconda Prompt，输入命令：

      ```bash
      conda create -n {env_name} python=3.7.5
      ```

      > env_name可以按照自己想要的环境名称自行命名。

  2. 激活新环境，输入命令：

      ```bash
      conda activate {env_name}
      ```

  3. 安装ipykernel，输入命令：

      ```bash
      conda install -n {env_name} ipykernel
      ```

      > 如果添加已有环境，只需执行安装ipykernel操作即可。

  执行完毕后，刷新Jupyter notebook页面点击Kernel下拉，选择Kernel Change，就能选择新添加的conda环境。

## notebook说明

| 教&nbsp;&nbsp;程&nbsp;&nbsp;类&nbsp;&nbsp;别               | 教&nbsp;&nbsp;程&nbsp;&nbsp;名&nbsp;&nbsp;称                    | 文&nbsp;&nbsp;件&nbsp;&nbsp;名&nbsp;&nbsp;称       |  内&nbsp;&nbsp;容&nbsp;&nbsp;描&nbsp;&nbsp;述
| :-----------               | :-----------   | :-------              |:------
|  快速入门        | 手写数字分类识别入门体验教程           |   [quick_start.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/quick_start.ipynb)     | - CPU平台下从数据集到模型验证的全过程解读 <br/> - 体验教程中各功能模块的使用说明 <br/> - 数据集图形化展示 <br/> - 了解LeNet5具体结构和参数作用 <br/> - 学习使用自定义回调函数 <br/> - loss值与训练步数的变化图 <br/> - 模型精度与训练步数的变化图 <br/> -  使用模型应用到手写图片的预测与分类上
| 快速入门         | 线性拟合         | [linear_regression.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/linear_regression.ipynb)       | - 了解线性拟合的算法原理<br/> - 了解在MindSpore中如何实现线性拟合的算法原理 <br/> - 学习使用MindSpore实现AI训练中的正向传播和方向传播<br/> - 可视化线性函数拟合数据的全过程。
| 基础使用         | 加载数据集        | [loading_dataset.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/loading_dataset.ipynb)           | - 学习MindSpore中加载数据集的方法 <br/> - 展示加载常用数据集的方法<br/> - 展示加载MindRecord格式数据集的方法<br/> - 展示加载自定义格式数据集的方法
| 基础使用         | 加载图像数据集        | [loading_image_dataset.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/loading_image_dataset.ipynb)           | - 学习加载图像数据集 <br/> - 学习处理图像数据集<br/> - 学习增强图像数据集
| 基础使用         | 加载文本数据集        | [loading_text_dataset.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/loading_text_dataset.ipynb)           | - 学习加载文本数据集 <br/> - 学习处理文本数据集<br/> - 学习文本数据集分词
| 基础使用         |  保存模型   |     [save_model.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/save_model.ipynb)          | - 了解不同平台用于训练的模型类型<br/> - 学习如何用不同策略保存训练模型<br/> - 学习如何将模型导出为不同的文件类型，用于不同平台上的训练
| 基础使用         | 加载模型用于推理或迁移学习         | [load_model_for_inference_and_transfer.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/load_model/load_model_for_inference_and_transfer.ipynb)       | - 了解预训练模型的方法<br/> - 学习在本地加载已有的模型进行推理的方法<br/> - 学习在本地加载模型并进行迁移学习的方法
| 处理数据         | 转换数据集为MindRecord        | [convert_dataset_to_mindrecord.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/convert_dataset_to_mindrecord/convert_dataset_to_mindrecord.ipynb)           | - 展示将数据集转换为MindRecord <br/> - 展示读取MindRecord数据集
| 处理数据        | 数据处理与数据增强      |  [data_loading_enhancement.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/data_loading_enhance/data_loading_enhancement.ipynb)            | - 学习MindSpore中数据处理和增强的方法 <br/> - 展示数据处理、增强方法的实际操作 <br/> - 对比展示数据处理前和处理后的效果<br/> - 表述在数据处理、增强后的意义
| 数据处理        | 优化数据准备的性能         | [optimize_the_performance_of_data_preparation.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/optimize_the_performance_of_data_preparation/optimize_the_performance_of_data_preparation.ipynb)       | - 数据加载性能优化<br/> - shuffle性能优化<br/> - 数据增强性能优化<br/> - 性能优化方案总结
| 应用实践        | 自然语言处理应用         |  [nlp_application.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/nlp_application.ipynb)         | - 展示MindSpore在自然语言处理的应用<br/> - 展示自然语言处理中数据集特定的预处理方法<br/> - 展示如何定义基于LSTM的SentimentNet网络
| 应用实践         | 计算机视觉应用     | [computer_vision_application.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/computer_vision_application.ipynb)       | - 学习MindSpore卷积神经网络在计算机视觉应用的过程 <br/> - 学习下载CIFAR-10数据集，搭建运行环境<br/>- 学习使用ResNet-50构建卷积神经网络<br/> - 学习使用Momentum和SoftmaxCrossEntropyWithLogits构建优化器和损失函数<br/> - 学习调试参数训练模型，判断模型精度
| 调试网络         | 模型的训练及验证同步方法         | [evaluate_the_model_during_training.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/evaluate_the_model_during_training.ipynb)       | - 了解模型训练和验证同步进行的方法<br/> - 学习同步训练和验证中参数设置方法<br/> - 利用绘图函数从保存的模型中挑选出最优模型
| 调试网络         | 使用PyNative进行神经网络的训练调试体验          | [debugging_in_pynative_mode.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/debugging_in_pynative_mode.ipynb)      | - GPU平台下从数据集获取单个数据进行单个step训练的数据变化全过程解读 <br/> - 了解PyNative模式下的调试方法 <br/> - 图片数据在训练过程中的变化情况的图形展示 <br/> - 了解构建权重梯度计算函数的方法 <br/> - 展示1个step过程中权重的变化及数据展示
| 调试网络         | 自定义调试信息体验文档         | [custom_debugging_info.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/custom_debugging_info.ipynb)       | - 了解MindSpore的自定义调试算子 <br/> - 学习使用自定义调试算子Callback设置定时训练<br/>- 学习设置metrics算子输出相对应的模型精度信息<br/> - 学习设置日志环境变量来控制glog输出日志
| 调试网络        |  MindInsight的溯源分析和对比分析            |  [lineage_and_scalars_comparision.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/mindinsight/lineage_and_scalars_comparision.ipynb)       | - 了解MindSpore中训练数据的采集及展示  <br/> - 学习使用回调函数SummaryCollector进行数据采集 <br/> - 使用MindInsight进行数据可视化 <br/> - 了解数据溯源和模型溯源的使用方法 <br/> - 了解对比分析的使用方法
| 调试网络          | MindInsight训练看板         | [mindinsight_dashboard.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/mindinsight/mindinsight_dashboard.ipynb)       | - 了解完整的MindSpore深度学习及MindInsight可视化展示的过程 <br/> - 学习使用MindInsight对训练过程中标量、直方图、图像、计算图、数据图和张量信息进行可视化展示<br/> - 学习使用Summary算子记录标量、直方图、图像、计算图、数据图和张量信息
| 优化训练性能           | 混合精度         | [mixed_precision.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/mixed_precision.ipynb)       | - 了解混合精度训练的原理  <br/> - 学习在MindSpore中使用混合精度训练 <br/> - 对比单精度训练和混合精度训练的对模型训练的影响
| 模型安全和隐私        | 模型安全         | [model_security.ipynb](https://gitee.com/mindspore/docs/blob/master/tutorials/notebook/model_security.ipynb)       | - 了解AI算法的安全威胁的概念和影响<br/> - 介绍MindArmour提供的模型安全防护手段<br/> - 学习如何模拟攻击训练模型<br/> - 学习针对被攻击模型进行对抗性防御

