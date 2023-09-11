# 脚本启动模型

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/docs/sciai/docs/source_zh_cn/launch_with_scripts.md)&nbsp;&nbsp;

MindSpore SciAI中的模型为用户提供了训练与评估的脚本文件。

通过模型的脚本文件，用户可以直接启动某个模型的训练与评估，并通过修改配置文件或是传入命令行参数的方式调整模型参数。[该目录](https://gitee.com/mindspore/mindscience/SciAI/sciai/model)中包含了所有支持脚本启动的模型。

下面使用模型Conservative Physics-Informed Neural Networks(CPINNs)介绍使用脚本训练、评估模型的基本通用流程。CPINNs模型相关代码请参考[链接](https://gitee.com/mindspore/mindscience/SciAI/sciai/model/cpinns)。

更多关于该模型的信息，请参考[论文](https://www.sciencedirect.com/science/article/abs/pii/S0045782520302127)。

## 仓库下载

使用如下命令直接克隆整个仓库，并初始化环境变量`PYTHONPATH`。

```bash
git clone https://gitee.com/mindspore/mindscience
source ./mindscience/SciAI/.env
```

克隆完成后，用户可以按照模型[README_CN.md](https://gitee.com/mindspore/mindscience/blob/master/SciAI/sciai/model/cpinns/README_CN.md)（以模型CPINNs为例）中的`快速开始`章节，使用脚本进行训练与推理。

```bash
cd ./mindscience/SciAI/sciai/model/cpinns/
```

## 训练、微调模型

用户可以使用训练脚本[train.py](https://gitee.com/mindspore/mindscience/blob/master/SciAI/sciai/model/cpinns/train.py)进行网络模型训练。

```bash
python ./train.py [--parameters]
# expected output
...
step: 0, loss1: 2.1404986, loss2: 8.205103, loss3: 37.23588, loss4: 3.56359, interval: 50.85803508758545s, total: 50.85803508758545s
step: 10, loss1: 2.6560388, loss2: 3.869413, loss3: 9.323585, loss4: 2.1194165, interval: 5.159524917602539s, total: 56.01756000518799s
step: 20, loss1: 1.7885156, loss2: 4.470225, loss3: 3.3072894, loss4: 1.5674783, interval: 1.8615927696228027s, total: 57.87915277481079s
...
```

使用已有的`.ckpt`文件进行模型微调。

```bash
python ./train.py --load_ckpt true --load_ckpt_path {your_file}.ckpt [--parameters]
```

使用可选参数`[--parameters]`可以配置模型的训练过程，包括学习率、训练周期、数据读取保存路径等。

具体可配置的参数列表请参考[README_CN.md](https://gitee.com/mindspore/mindscience/blob/master/SciAI/sciai/model/cpinns/README_CN.md)中`脚本参数`章节。

## 评估模型

用户可以使用脚本`eval.py`对已完成训练的网络模型进行评估。

```bash
python ./eval.py [--parameters]
# expected output
...
error_u:  0.024803562642018585
Total time running eval: 20.625872135162354 seconds
```

使用可选参数`[--parameters]`可以配置模型的评估过程，包括数据读取保存路径、checkpoints文件加载路径等。
