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
| [quick_start.ipynb](./quick_start.ipynb)                               | - 从数据集到模型验证的全过程解读 <br/> - 体验教程中各功能模块的使用说明 <br/> - 数据集图形化展示 <br/> - 了解LeNet5具体结构和参数作用 <br/> - 学习使用自定义回调函数 <br/> - loss值与训练步数的变化图 <br/> - 模型精度与训练步数的变化图 <br/> -  使用模型应用到手写图片的预测与分类上