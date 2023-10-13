# Launching Model with Scripts

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/sciai/docs/source_en/launch_with_scripts.md)&nbsp;&nbsp;

The models in MindSpore SciAI provides users with scripts for training and evaluation.

User can train or evaluate any model by running scripts, and the model parameters can be adjusted either through editing the config file or passing parameters in the command line. [This folder](https://gitee.com/mindspore/mindscience/tree/r0.5/SciAI/sciai/model) contains all the models that support launching with scripts.

The following content introduces the general process of training, evaluating models with scripts, taking Conservative Physics-Informed Neural Networks(CPINNs) as an example. For the codes of CPINNs model, please refer to the [link](https://gitee.com/mindspore/mindscience/tree/r0.5/SciAI/sciai/model/cpinns).

The fundamental idea about this model can be found in this [paper](https://www.sciencedirect.com/science/article/abs/pii/S0045782520302127).

## Downloading the Repository

User can clone the whole repository and initialize the environment variable `PYTHONPATH` with the following commands.

```bash
git clone https://gitee.com/mindspore/mindscience
source ./mindscience/SciAI/.env
```

After a successful clone, user can start training or evaluating according to the `Quick Start` section in the [README.md](https://gitee.com/mindspore/mindscience/blob/r0.5/SciAI/sciai/model/cpinns/README.md)(In case of CPINNs).

```bash
cd ./mindscience/SciAI/sciai/model/cpinns/
source ./mindscience/SciAI/.env
```

## Training and Fine-tuning the Model

User can run script [train.py](https://gitee.com/mindspore/mindscience/blob/r0.5/SciAI/sciai/model/cpinns/train.py) in each model directory to train the models.

```bash
python ./train.py [--parameters]
# expected output
...
step: 0, loss1: 2.1404986, loss2: 8.205103, loss3: 37.23588, loss4: 3.56359, interval: 50.85803508758545s, total: 50.85803508758545s
step: 10, loss1: 2.6560388, loss2: 3.869413, loss3: 9.323585, loss4: 2.1194165, interval: 5.159524917602539s, total: 56.01756000518799s
step: 20, loss1: 1.7885156, loss2: 4.470225, loss3: 3.3072894, loss4: 1.5674783, interval: 1.8615927696228027s, total: 57.87915277481079s
...
```

Use the `.ckpt` file to finetune the network:

```bash
python ./train.py --load_ckpt true --load_ckpt_path {your_file}.ckpt [--parameters]
```

Using the optional parameter `[--parameters]`, user can configure the training process of the model, including learning rate, training epochs, data saving and loading paths and so on.

For details about the configurable parameters in each model, see the `Script Parameters` section in the [README.md](https://gitee.com/mindspore/mindscience/blob/r0.5/SciAI/sciai/model/cpinns/README.md).

## Evaluating the Model

User can run script `eval.py` in each model directory to evaluate the trained networks.

```bash
python ./eval.py [--parameters]
# expected output
...
error_u:  0.024803562642018585
Total time running eval: 20.625872135162354 seconds
```

Using the optional parameter `[--parameters]`, user can configure the evaluation process of the model, including the data read and save paths, checkpoints file loading paths, and so on.