# 评测

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_zh_cn/usage/evaluation.md)

## Harness评测

### 基本介绍

[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)是一个开源语言模型评测框架，提供60多种标准学术数据集的评测，支持HuggingFace模型评测、PEFT适配器评测、vLLM推理评测等多种评测方式，支持自定义prompt和评测指标，包含loglikelihood、generate_until、loglikelihood_rolling三种类型的评测任务。基于Harness评测框架对MindSpore Transformers进行适配后，支持加载MindSpore Transformers模型进行评测。

目前已验证过的模型和支持的评测任务如下表所示（其余模型和评测任务正在积极验证和适配中，请关注版本更新）：

| 已验证的模型   | 支持的评测任务                |
|----------|------------------------|
| Llama3   | gsm8k、ceval-valid、mmlu、cmmlu、race、lambada |
| Llama3.1 | gsm8k、ceval-valid、mmlu、cmmlu、race、lambada |
| Qwen2    | gsm8k、ceval-valid、mmlu、cmmlu、race、lambada |

### 安装

Harness支持pip安装和源码编译安装两种方式。pip安装更简单快捷，源码编译安装更便于调试分析，用户可以根据需要选择合适的安装方式。

#### pip安装

用户可以执行如下命令安装Harness（推荐使用0.4.4版本）：

```shell
pip install lm_eval==0.4.4
```

#### 源码编译安装

用户可以执行如下命令编译并安装Harness：

```bash
git clone --depth 1 -b v0.4.4 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

### 使用方式

#### 评测前准备

  1. 创建一个新目录，例如名称为`model_dir`，用于存储模型yaml文件。
  2. 在上个步骤创建的目录中，放置模型推理yaml配置文件（predict_xxx_.yaml）。不同模型的推理yaml配置文件所在目录位置，请参考[模型库](../start/models.md)。
  3. 配置yaml文件。如果yaml中模型类、模型Config类、模型Tokenzier类使用了外挂代码，即代码文件在[research](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research)目录或其他外部目录下，需要修改yaml文件：在相应类的`type`字段下，添加`auto_register`字段，格式为“module.class”（其中“module”为类所在脚本的文件名，“class”为类名。如果已存在，则不需要修改）。

      以[predict_llama3_1_8b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3_1/llama3_1_8b/predict_llama3_1_8b.yaml)配置为例，对其中的部分配置项进行如下修改：

        ```yaml
        run_mode: 'predict'       # 设置推理模式
        load_checkpoint: 'model.ckpt'   # 权重路径
        processor:
          tokenizer:
            vocab_file: "tokenizer.model"     # tokenizer路径
            type: Llama3Tokenizer
            auto_register: llama3_tokenizer.Llama3Tokenizer
        ```

        关于每个配置项的详细说明请参考[配置文件说明](../appendix/conf_files.md)。
  4. 如果使用`ceval-valid`、`mmlu`、`cmmlu`、`race`、`lambada`数据集进行评测，需要将`use_flash_attention`设置为`False`，以`predict_llama3_1_8b.yaml`为例，修改yaml如下：

       ```yaml
       model:
         model_config:
           # ...
           use_flash_attention: False  # 设置为False
           # ...
       ```

#### 评测样例

执行脚本[run_harness.sh](https://gitee.com/mindspore/mindformers/blob/r1.5.0/toolkit/benchmarks/run_harness.sh)进行评测。

run_harness.sh脚本参数配置如下表：

| 参数               | 类型  | 参数介绍                                                                                           | 是否必须 |
|------------------|-----|------------------------------------------------------------------------------------------------|------|
| `--register_path`| str | 外挂代码所在目录的绝对路径。比如[research](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research)目录下的模型目录 | 否（外挂代码必填）    |
| `--model`        | str | 需设置为 `mf` ，对应为MindSpore Transformers评估策略                                                                  | 是    |
| `--model_args`   | str | 模型及评估相关参数，见下方模型参数介绍                                                                            | 是    |
| `--tasks`        | str | 数据集名称。可传入多个数据集，使用逗号（，）分隔                                                                         | 是    |
| `--batch_size`   | int | 批处理样本数                                                                                         | 否    |

其中，model_args参数配置如下表：

| 参数             | 类型      | 参数介绍               | 是否必须 |
|----------------|---------|--------------------|------|
| `pretrained`   | str     | 模型目录路径             | 是    |
| `max_length`   | int     | 模型生成的最大长度          | 否    |
| `use_parallel` | bool | 开启并行策略(执行多卡评测必须开启) | 否    |
| `tp`           | int     | 张量并行数               | 否    |
| `dp`           | int     | 数据并行数               | 否    |

Harness评测支持单机单卡、单机多卡、多机多卡场景，每种场景的评测样例如下：

1. 单卡评测样例

   ```shell
      source toolkit/benchmarks/run_harness.sh \
       --register_path mindformers/research/llama3_1 \
       --model mf \
       --model_args pretrained=model_dir \
       --tasks gsm8k
   ```

2. 多卡评测样例

   ```shell
      source toolkit/benchmarks/run_harness.sh \
       --register_path mindformers/research/llama3_1 \
       --model mf \
       --model_args pretrained=model_dir,use_parallel=True,tp=4,dp=1 \
       --tasks ceval-valid \
       --batch_size BATCH_SIZE WORKER_NUM
   ```

   - `BATCH_SIZE`为模型批处理样本数；
   - `WORKER_NUM`为使用计算卡的总数。

3. 多机多卡评测样例

   节点0（主节点）命令：

   ```shell
      source toolkit/benchmarks/run_harness.sh \
       --register_path mindformers/research/llama3_1 \
       --model mf \
       --model_args pretrained=model_dir,use_parallel=True,tp=8,dp=1 \
       --tasks lambada \
       --batch_size 2 8 4 192.168.0.0 8118 0 output/msrun_log False 300
   ```

   节点1（副节点）命令：

   ```shell
      source toolkit/benchmarks/run_harness.sh \
       --register_path mindformers/research/llama3_1 \
       --model mf \
       --model_args pretrained=model_dir,use_parallel=True,tp=8,dp=1 \
       --tasks lambada \
       --batch_size 2 8 4 192.168.0.0 8118 1 output/msrun_log False 300
   ```

   节点n（副节点）命令：

   ```shell
      source toolkit/benchmarks/run_harness.sh \
       --register_path mindformers/research/llama3_1 \
       --model mf \
       --model_args pretrained=model_dir,use_parallel=True,tp=8,dp=1 \
       --tasks lambada \
       --batch_size BATCH_SIZE WORKER_NUM LOCAL_WORKER MASTER_ADDR MASTER_PORT NODE_RANK output/msrun_log False CLUSTER_TIME_OUT
   ```

   - `BATCH_SIZE`为模型批处理样本数；
   - `WORKER_NUM`为所有节点中使用计算卡的总数；
   - `LOCAL_WORKER`为当前节点中使用计算卡的数量；
   - `MASTER_ADDR`为分布式启动主节点的ip；
   - `MASTER_PORT`为分布式启动绑定的端口号；
   - `NODE_RANK`为当前节点的rank id；
   - `CLUSTER_TIME_OUT`为分布式启动的等待时间，单位为秒。

   多机多卡评测需要分别在不同节点运行脚本，并将参数MASTER_ADDR设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数NODE_RANK不同。

### 查看评测结果

执行评测命令后，评测结果将会在终端打印出来。以gsm8k为例，评测结果如下，其中Filter对应匹配模型输出结果的方式，n-shot对应数据集内容格式，Metric对应评测指标，Value对应评测分数，Stderr对应分数误差。

| Tasks | Version | Filter           | n-shot | Metric      |   | Value  |   | Stderr |
|-------|--------:|------------------|-------:|-------------|---|--------|---|--------|
| gsm8k |       3 | flexible-extract |      5 | exact_match | ↑ | 0.5034 | ± | 0.0138 |
|       |         | strict-match     |      5 | exact_match | ↑ | 0.5011 | ± | 0.0138 |

## VLMEvalKit评测

### 基本介绍

[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
是一款专为大型视觉语言模型评测而设计的开源工具包，支持在各种基准测试上对大型视觉语言模型进行一键评估，无需进行繁重的数据准备工作，让评估过程更加简便。它支持多种图文多模态评测集和视频多模态评测集，支持多种API模型以及基于PyTorch和HF的开源模型，支持自定义prompt和评测指标。基于VLMEvalKit评测框架对MindSpore Transformers进行适配后，支持加载MindSpore Transformers中多模态大模型进行评测。

目前已适配的模型和支持的评测数据集如下表所示（其余模型和评测数据集正在积极适配中，请关注版本更新）：

| 适配的模型 | 支持的评测任务                                           |
|--|---------------------------------------------------|
| cogvlm2-image-llama3-chat | MME、MMBench、COCO Caption、MMMU_DEV_VAL、TextVQA_VAL |
| cogvlm2-video-llama3-chat | MMBench-Video、MVBench                             |

### 支持特性说明

1. 支持自动下载评测数据集；
2. 一键生成评测结果。

### 安装

#### 下载代码并编译，安装依赖包

1. 下载并修改代码：由于开源框架在跑MVBench数据集时存在已知问题，所以需要使用导入patch补丁的方式修改源码。获取[eval.patch](https://github.com/user-attachments/files/17956417/eval.patch)，下载放入本地目录中。导入patch时要使用patch文件的绝对路径。

    执行以下命令：

    ```bash
    git clone https://github.com/open-compass/VLMEvalKit.git
    cd VLMEvalKit
    git checkout 78a8cef3f02f85734d88d534390ef93ecc4b8bed
    git apply /path/to/eval.patch
    ```

2. 安装依赖包

    在下载好的代码中，找到requirements.txt（VLMEvalKit/requirements.txt）文件，修改成如下内容：

    ```txt
    gradio==4.40.0
    huggingface_hub==0.24.2
    imageio==2.35.1
    matplotlib==3.9.1
    moviepy==1.0.3
    numpy==1.26.4
    omegaconf==2.3.0
    openai==1.3.5
    opencv-python==4.10.0.84
    openpyxl==3.1.5
    pandas==2.2.2
    peft==0.12.0
    pillow==10.4.0
    portalocker==2.10.1
    protobuf==5.27.2
    python-dotenv==1.0.1
    requests==2.32.3
    rich==13.7.1
    sentencepiece==0.2.0
    setuptools==69.5.1
    sty==1.0.6
    tabulate==0.9.0
    tiktoken==0.7.0
    timeout-decorator==0.5.0
    torch==2.5.1
    tqdm==4.66.4
    transformers==4.43.3
    typing_extensions==4.12.2
    validators==0.33.0
    xlsxwriter==3.2.0
    torchvision==0.20.1
    ```

    执行命令：

    ```bash
    pip install -r requirements.txt
    ```

#### 安装FFmpeg

Ubuntu系统按照如下步骤安装：

1. 更新系统包列表，安装编译FFmpeg所需的系统依赖库。

      ```bash
      apt-get update
      apt-get -y install autoconf automake build-essential libass-dev libfreetype6-dev libsdl2-dev libtheora-dev libtool libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev pkg-config texinfo zlib1g-dev yasm libx264-dev libfdk-aac-dev libmp3lame-dev libopus-dev libvpx-dev
      ```

2. 从FFmpeg官网下载FFmpeg4.1.11的源码压缩包，解压源码包并进入解压后的目录；配置FFmpeg的编译选项：指定FFmpeg的安装路径（绝对路径），生成共享库，启用对特定编解码器的支持，启用非自由和GPL许可的功能；编译并安装FFmpeg。

      ```bash
      wget --no-check-certificate https://www.ffmpeg.org/releases/ffmpeg-4.1.11.tar.gz
      tar -zxvf ffmpeg-4.1.11.tar.gz
      cd ffmpeg-4.1.11
      ./configure --prefix=/{path}/ffmpeg-xxx --enable-shared --enable-libx264 --enable-libfdk-aac --enable-libmp3lame --enable-libopus --enable-libvpx --enable-nonfree --enable-gpl
      make && make install
      ```

OpenEuler系统按照如下步骤安装：

1. 从FFmpeg官网下载FFmpeg4.1.11的源码压缩包，解压源码包并进入解压后的目录；配置FFmpeg的编译选项：指定FFmpeg的安装路径（绝对路径）；编译并安装FFmpeg。

      ```bash
      wget --no-check-certificate https://www.ffmpeg.org/releases/ffmpeg-4.1.11.tar.gz
      tar -zxvf ffmpeg-4.1.11.tar.gz
      cd ffmpeg-4.1.11
      ./configure --enable-shared --disable-x86asm --prefix=/path/to/ffmpeg
      make && make install
      ```

2. 配置环境变量，`FFMPEG_PATH`需要指定安装FFmpeg的绝对路径，以便系统能够正确找到和使用FFmpeg及其相关库。

      ```bash
      vi ~/.bashrc
      export FFMPEG_PATH=/path/to/ffmpeg/
      export LD_LIBRARY_PATH=$FFMPEG_PATH/lib:$LD_LIBRARY_PATH
      source ~/.bashrc
      ```

#### 安装Decord

Ubuntu系统按照如下步骤安装：

1. 拉取Decord代码，进入`decord`目录，执行以下命令：

      ```bash
      git clone --recursive -b v0.6.0 https://github.com/dmlc/decord.git
      cd decord
      ```

2. 创建并进入`build`目录，配置Decord的编译选项，禁用CUDA支持，启用Release模式（优化性能），指定FFmpeg的安装路径，编译Decord库。将编译生成的libdecord.so库文件复制到系统库目录，复制到`decord`的`python`目录。

      ```bash
      mkdir build
      cd build
      cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -DFFMPEG_DIR=/{path}/ffmpeg-4.1.11 && make
      cp libdecord.so /usr/local/lib/
      cp libdecord.so ../python/decord/libdecord.so
      ```

3. 进入`decord`目录中的`python`文件夹，安装`numpy`依赖项，安装Decord的python包。将FFmpeg的库路径（绝对路径）添加到`LD_LIBRARY_PATH`环境变量中，确保运行时能够找到FFmpeg的共享库。

      ```bash
      cd /path/to/decord/python
      pip install numpy
      python setup.py install
      export LD_LIBRARY_PATH=/path/to/ffmpeg-4.1.11/lib/:$LD_LIBRARY_PATH
      ```

4. 执行Python命令，测试Decord是否安装成功，没有报错即为安装成功。

      ```bash
      python -c "import decord; from decord import VideoReader"
      ```

OpenEuler系统按照如下步骤安装：

1. 拉取Decord代码，进入`decord`目录。

      ```bash
      git clone --recursive -b v0.6.0 https://github.com/dmlc/decord
      cd decord
      ```

2. 创建并进入`build`目录，配置Decord的编译选项，指定FFmpeg的安装路径(绝对路径)，编译Decord库；进入`decord`目录中的python文件夹，配置环境变量，指定`PYTHONPATH`；安装Decord的python包。

      ```bash
      mkdir build && cd build
      cmake -DFFMPEG_DIR=/path/ffmpeg-4.1.11 ..
      make
      cd ../python
      pwd=$PWD
      echo "PYTHONPATH=$PYTHONPATH:$pwd" >> ~/.bashrc
      source ~/.bashrc
      python3 setup.py install
         ```

3. 执行python命令，测试Decord是否安装成功，没有报错即为安装成功。

      ```bash
      python -c "import decord; from decord import VideoReader"
      ```

### 评测

#### 评测前准备

1. 创建一个新目录，例如名称为`model_dir`，用于存储模型yaml文件；
2. 在上个步骤创建的目录中放置模型推理yaml配置文件（predict_xxx_.yaml），不同模型的推理yaml配置文件的目录位置参考[模型库](../start/models.md)各模型说明文档中的模型文件树；
3. 配置yaml配置文件。

    以[predict_cogvlm2_image_llama3_chat_19b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/cogvlm2/predict_cogvlm2_image_llama3_chat_19b.yaml)配置为例：

    ```yaml
    load_checkpoint: "/{path}/model.ckpt"  # 指定权重文件路径
    model:
      model_config:
        use_past: True                         # 开启增量推理
        is_dynamic: False                       # 关闭动态shape

      tokenizer:
        vocab_file: "/{path}/tokenizer.model"  # 指定tokenizer文件路径
    ```

   配置yaml文件，参考[配置文件说明](../appendix/conf_files.md)。
4. MMbench-Video数据集评测需要使用GPT-4 Turbo模型进行评测打分，请提前准备好相应的API Key，并放在VLMEvalKit/.env文件中，内容如下所示：

   ```text
   OPENAI_API_KEY=your_apikey
   ```

5. MVBench数据集评测开始时，如果提示需要输入HuggingFace密钥，请按提示输入，保证后续评测的正常执行。

#### 拉起评测任务

在MindSpore Transformers本地代码仓根目录下执行脚本：[run_vlmevalkit.sh](https://gitee.com/mindspore/mindformers/blob/r1.5.0/toolkit/benchmarks/run_vlmevalkit.sh)。

执行如下命令拉起评测任务：

```shell
#!/bin/bash

source toolkit/benchmarks/run_vlmevalkit.sh \
 --data MMMU_DEV_VAL \
 --model cogvlm2-image-llama3-chat \
 --verbose \
 --work_dir /path/to/cogvlm2-image-eval-result \
 --model_path model_dir
```

### 评测参数

| 参数                | 类型  | 参数介绍                                                                                           | 是否必须      |
|-------------------|-----|------------------------------------------------------------------------------------------------|-----------|
| `--data`          | str | 数据集名称，可传入多个数据集，空格分割。                                                                           | 是         |
| `--model`         | str | 模型名称。                                                                                          | 是         |
| `--verbose`       | /   | 输出评测运行过程中的日志。                                                                                  | 否         |
| `--work_dir`      | str | 存放评测结果的目录，默认存储在当前执行目录的`outputs`文件夹下。                                                           | 否         |
| `--model_path`    | str | 包含配置文件的文件夹路径。                                                                                  | 是         |
| `--register_path` | str | 外挂代码所在目录的绝对路径。比如[research](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research)目录下的模型目录。 | 否（外挂代码必填） |

如果因网络限制，服务器不支持在线下载图文数据集时，可以将本地下载好的以.tsv结尾的数据集文件上传至服务器~/LMUData目录下，进行离线评测。（例如：~/LMUData/MME.tsv 或 ~/LMUData/MMBench_DEV_EN.tsv 或 ~/LMUData/COCO_VAL.tsv）

### 查看评测结果

按照上述方式评估后，在存储评测结果的目录中，找到以.json或以.csv结尾的文件查看评估的结果。

评测样例结果如下，其中`Bleu`和`ROUGE_L`表示评估翻译质量的指标，`CIDEr`表示评估图像描述任务的指标。

```json
{
   "Bleu": [
      15.523950970070652,
      8.971141548228058,
      4.702477458554666,
      2.486860744700995
   ],
   "ROUGE_L": 15.575063213115946,
   "CIDEr": 0.01734615519604295
}
```

## 使用VideoBench数据集进行模型评测

### 基本介绍

[Video-Bench](https://github.com/PKU-YuanGroup/Video-Bench/tree/main) 是首个针对 Video-LLM 的综合评估基准，具有三级能力评估，可以系统地评估模型在视频专属理解、先验知识融入和基于视频的决策能力方面的表现。

### 评测前准备

1. 数据集下载

    下载[Video-Bench中的视频数据](https://huggingface.co/datasets/LanguageBind/Video-Bench)，解压后按照如下目录格式进行放置：

    ```text
    egs/VideoBench/
      └── Eval_video
            ├── ActivityNet
            │     ├── v__2txWbQfJrY.mp4
            │     ...
            ├── Driving-decision-making
            │     ├── 1.mp4
            │     ...
            ...
    ```

2. 文本下载

    下载[Video-Bench中的文本数据](https://github.com/PKU-YuanGroup/Video-Bench/tree/main?tab=readme-ov-file)，解压后按照如下目录格式进行放置：

    ```text
    egs/Video-Bench/
      └── Eval_QA
            ├── Youcook2_QA_new.json等json文件
            ...
    ```

3. 所有问题的正确答案下载

    下载[Video-Bench中的答案数据](https://huggingface.co/spaces/LanguageBind/Video-Bench/resolve/main/file/ANSWER.json)。

> 注：Video-Bench中的文本数据按照“egs/VideoBench/Eval_QA”（目录至少两层，且最后一层是`Eval_QA`）的路径格式进行存储；Video-Bench中的视频数据按照“egs/VideoBench/Eval_video”（目录至少两层，且最后一层是`Eval_video`）的路径格式进行存储。

### 评测

执行脚本路径可参考链接：[eval_with_videobench.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/toolkit/benchmarks/eval_with_videobench.py)。

#### 执行推理脚本，获取推理结果

```shell
python toolkit/benchmarks/eval_with_videobench.py \
--model_path model_path \
--dataset_name dataset_name \
--Eval_QA_root Eval_QA_root \
--Eval_Video_root Eval_Video_root \
--chat_conversation_output_folder output
```

> 参数`Eval_QA_root`填写Eval_QA的上一层目录；参数`Eval_Video_root`填写Eval_video的上一层目录。

**参数说明**

| **参数**                             | **是否必选** | **说明**                                     |
|------------------------------------|---------|--------------------------------------------|
| `--model_path`                     | 是       | 存储模型相关文件的文件夹路径，包含模型配置文件及模型词表文件。            |
| `--dataset_name`                   | 否       | 评测数据子集名称，默认为None，评测VideoBench的所有子集。        |
| `--Eval_QA_root`                   | 是       | 存放VideoBench数据集的json文件目录。 |
| `--Eval_Video_root`                | 是       | 存放VideoBench数据集的视频文件目录。                    |
| `--chat_conversation_output_folder` | 否       | 生成结果文件的目录。默认存放在当前目录的Chat_results文件夹下。      |

运行结束后，在chat_conversation_output_folder目录下会生成对话结果文件。

#### 根据生成结果进行评测打分

Video-Bench可以根据模型生成的答案利用ChatGPT或T5进行评估，最终得到13个数据子集的最终分数。

例如：使用ChatGPT进行评估打分：

```shell
python Step2_chatgpt_judge.py \
--model_chat_files_folder ./Chat_results \
--apikey sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
--chatgpt_judge_output_folder ./ChatGPT_Judge

python Step3_merge_into_one_json.py \
--chatgpt_judge_files_folder ./ChatGPT_Judge \
--merge_file ./Video_Bench_Input.json
```

上述评测打分命令中的脚本路径为：[Step2_chatgpt_judge.py](https://github.com/PKU-YuanGroup/Video-Bench/blob/main/Step2_chatgpt_judge.py)、[Step3_merge_into_one_json.py](https://github.com/PKU-YuanGroup/Video-Bench/blob/main/Step3_merge_into_one_json.py)

ChatGPT可能会将部分问题的回答视为格式错误，因此需要多次运行Step2_chatgpt_judge.py以确保每个问题都由ChatGPT进行验证。

## FAQ

1. 使用Harness或VLMEvalKit进行评测，在加载HuggingFace数据集时，报错`SSLError`：

   参考[SSL Error报错解决方案](https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models)。

   注意：关闭SSL校验存在风险，可能暴露在中间人攻击（MITM）下。仅建议在测试环境或你完全信任的连接里使用。

2. 使用VLMEvalKit中的MVBench数据集进行评测，出现`AssertionError`：

   由于开源框架`VLMEvalKit`在跑`MVBench`数据集时存在已知问题，请参考开源框架的[issue](https://github.com/open-compass/VLMEvalKit/issues/888)进行修改，或删除评测过程中产生的文件（由参数`--work_dir`指定，默认在当前执行目录的`outputs`文件夹）重新执行。