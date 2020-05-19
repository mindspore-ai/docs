# MindSpore的教程体验

## 环境配置
### Windows系统配置方法

- 系统版本：Windows 10

- 软件配置：[Anaconda](https://www.anaconda.com/products/individual)，Jupyter Notebook

- 语言环境：Python3.7.X 推荐 Python3.7.5

- MindSpore 下载地址：[MindSpore官网下载](https://www.mindspore.cn/versions)选择Windows版本

> Windows系统MindSpore的[具体安装教程](https://www.mindspore.cn/install/) 

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
| [quick_start.ipynb](./quick_start.ipynb)                               |  通过该文件，你可更容易地理解各个功能模块的具体作用，学习到数据集查看及数据集图形展示方法，了解到数据集是如何通过训练生成模型；也可以通过LeNet计算图的展示，了解具体结构和参数作用；可以学习使用自定义回调函数来了解训练过程模型的变化，通过训练过程loss值与训练步数的变化图，模型精度与训练步数的变化图，更容易的理解训练对机器学习产生的意义，还能学习将训练出来的模型应用到手写图片的预测与分类上。