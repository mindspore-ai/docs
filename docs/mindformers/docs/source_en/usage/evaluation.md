# Evaluation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/usage/evaluation.md)

## Harness Evaluation

### Introduction

[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) is an open-source language model evaluation framework that provides evaluation of more than 60 standard academic datasets, supports multiple evaluation modes such as HuggingFace model evaluation, PEFT adapter evaluation, and vLLM inference evaluation, and supports customized prompts and evaluation metrics, including the evaluation tasks of the loglikelihood, generate_until, and loglikelihood_rolling types. After MindSpore Transformers is adapted based on the Harness evaluation framework, the MindSpore Transformers model can be loaded for evaluation.

The currently verified models and supported evaluation tasks are shown in the table below (the remaining models and evaluation tasks are actively being verified and adapted, please pay attention to version updates):

| Verified models | Supported evaluation tasks                     |
|-----------------|------------------------------------------------|
| Llama3   | gsm8k, ceval-valid, mmlu, cmmlu, race, lambada |
| Llama3.1 | gsm8k, ceval-valid, mmlu, cmmlu, race, lambada |
| Qwen2    | gsm8k, ceval-valid, mmlu, cmmlu, race, lambada |

### Installation

Harness supports two installation methods: pip installation and source code compilation installation. Pip installation is simpler and faster, source code compilation and installation are easier to debug and analyze, and users can choose the appropriate installation method according to their needs.

#### pip Installation

Users can execute the following command to install Harness (Recommend using version 0.4.4):

```shell
pip install lm_eval==0.4.4
```

#### Source Code Compilation Installation

Users can execute the following command to compile and install Harness:

```bash
git clone --depth 1 -b v0.4.4 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

### Usage

#### Preparations Before Evaluation

  1. Create a new directory with e.g. the name `model_dir` for storing the model yaml files.
  2. Place the model inference yaml configuration file (predict_xxx_.yaml) in the directory created in the previous step. The directory location of the reasoning yaml configuration file for different models refers to [model library](../start/models.md).
  3. Configure the yaml file. If the model class, model Config class, and model Tokenzier class in yaml use cheat code, that is, the code files are in [research](https://gitee.com/mindspore/mindformers/tree/dev/research) directory or other external directories, it is necessary to modify the yaml file: under the corresponding class `type` field, add the `auto_register` field in the format of `module.class`. (`module` is the file name of the script where the class is located, and `class` is the class name. If it already exists, there is no need to modify it.).

      Using [predict_1lama3_1_8b. yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/llama3_1/llama3_1_8b/predict_llama3_1_8b.yaml) configuration as an example, modify some of the configuration items as follows:

      ```yaml
      run_mode: 'predict'    # Set inference mode
      load_checkpoint: 'model.ckpt'    # path of ckpt
      processor:
        tokenizer:
          vocab_file: "tokenizer.model"    # path of tokenizer
          type: Llama3Tokenizer
          auto_register: llama3_tokenizer.Llama3Tokenizer
      ```

      For detailed instructions on each configuration item, please refer to the [configuration description](../appendix/conf_files.md).
  4. If you use the `ceval-valid`, `mmlu`, `cmmlu`, `race`, and `lambada` datasets for evaluation, you need to set `use_flash_attention` to `False`. Using `predict_lama3_1_8b.yaml` as an example, modify the yaml as follow:

      ```yaml
      model:
        model_config:
          # ...
          use_flash_attention: False  # Set to False
          # ...
       ```

#### Evaluation Example

Execute the script of [run_harness.sh](https://gitee.com/mindspore/mindformers/blob/dev/toolkit/benchmarks/run_harness.sh) to evaluate.

The following table lists the parameters of the script of `run_harness.sh`:

| Parameter           | Type | Description                                                                                                                                                                                   | Required |
|---------------|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `--register_path`| str | The absolute path of the directory where the cheat code is located. For example, the model directory under the [research](https://gitee.com/mindspore/mindformers/tree/dev/research) directory. | No(The cheat code is required)     |
| `--model`       | str  | The value must be `mf`, indicating the MindSpore Transformers evaluation policy.                                                                                                                          | Yes      |
| `--model_args`  | str  | Model and evaluation parameters. For details, see MindSpore Transformers model parameters.                                                                                                            | Yes      |
| `--tasks`       | str  | Dataset name. Multiple datasets can be specified and separated by commas (,).                                                                                                                 | Yes      |
| `--batch_size`  | int  | Number of batch processing samples.                                                                                                                                                           | No       |

The following table lists the parameters of `model_args`:

| Parameter          | Type | Description                                                              | Required |
|--------------|------|--------------------------------------------------------------------------|----------|
| `pretrained`   | str  | Model directory.                                                         | Yes      |
| `max_length`   | int  | Maximum length of model generation.                                      | No       |
| `use_parallel` | bool | Enable parallel strategy (It must be enabled for multi card evaluation). | No       |
| `tp`           | int  | The number of parallel tensors.                                          | No       |
| `dp`           | int  | The number of parallel data.                                             | No       |

Harness evaluation supports single-device single-card, single-device multiple-card, and multiple-device multiple-card scenarios, with sample evaluations for each scenario listed below:

1. Single Card Evaluation Example

   ```shell
      source toolkit/benchmarks/run_harness.sh \
       --register_path mindformers/research/llama3_1 \
       --model mf \
       --model_args pretrained=model_dir \
       --tasks gsm8k
   ```

2. Multi Card Evaluation Example

   ```shell
      source toolkit/benchmarks/run_harness.sh \
       --register_path mindformers/research/llama3_1 \
       --model mf \
       --model_args pretrained=model_dir,use_parallel=True,tp=4,dp=1 \
       --tasks ceval-valid \
       --batch_size BATCH_SIZE WORKER_NUM
   ```

    - `BATCH_SIZE` is the sample size for batch processing of models;
    - `WORKER_NUM` is the number of compute devices.

3. Multi-Device and Multi-Card Example

   Node 0 (Master) Command:

      ```shell
         source toolkit/benchmarks/run_harness.sh \
          --register_path mindformers/research/llama3_1 \
          --model mf \
          --model_args pretrained=model_dir,use_parallel=True,tp=8,dp=1 \
          --tasks lambada \
          --batch_size 2 8 4 192.168.0.0 8118 0 output/msrun_log False 300
      ```

   Node 1 (Secondary Node) Command:

      ```shell
         source toolkit/benchmarks/run_harness.sh \
          --register_path mindformers/research/llama3_1 \
          --model mf \
          --model_args pretrained=model_dir,use_parallel=True,tp=8,dp=1 \
          --tasks lambada \
          --batch_size 2 8 4 192.168.0.0 8118 1 output/msrun_log False 300
      ```

   Node n (Nth Node) Command:

      ```shell
         source toolkit/benchmarks/run_harness.sh \
          --register_path mindformers/research/llama3_1 \
          --model mf \
          --model_args pretrained=model_dir,use_parallel=True,tp=8,dp=1 \
          --tasks lambada \
          --batch_size BATCH_SIZE WORKER_NUM LOCAL_WORKER MASTER_ADDR MASTER_PORT NODE_RANK output/msrun_log False CLUSTER_TIME_OUT
      ```

   - `BATCH_SIZE` is the sample size for batch processing of models;
   - `WORKER_NUM` is the total number of compute devices used on all nodes;
   - `LOCAL_WORKER` is the number of compute devices used on the current node;
   - `MASTER_ADDR` is the ip address of the primary node to be started in distributed mode;
   - `MASTER_PORT` is the Port number bound for distributed startup;
   - `NODE_RANK` is the Rank ID of the current node;
   - `CLUSTER_TIME_OUT`is the waiting time for distributed startup, in seconds.

   To execute the multi-node multi-device script for evaluating, you need to run the script on different nodes and set MASTER_ADDR to the IP address of the primary node. The IP address should be the same across all nodes, and only the NODE_RANK parameter varies across nodes.

### Viewing the Evaluation Results

After executing the evaluation command, the evaluation results will be printed out on the terminal. Taking gsm8k as an example, the evaluation results are as follows, where Filter corresponds to the way the matching model outputs results, n-shot corresponds to content format of dataset, Metric corresponds to the evaluation metric, Value corresponds to the evaluation score, and Stderr corresponds to the score error.

| Tasks | Version | Filter           | n-shot | Metric      |   | Value  |   | Stderr |
|-------|--------:|------------------|-------:|-------------|---|--------|---|--------|
| gsm8k |       3 | flexible-extract |      5 | exact_match | ↑ | 0.5034 | ± | 0.0138 |
|       |         | strict-match     |      5 | exact_match | ↑ | 0.5011 | ± | 0.0138 |

## VLMEvalKit Evaluation

### Overview

[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
is an open source toolkit designed for large visual language model evaluation, supporting one-click evaluation of large visual language models on various benchmarks, without the need for complicated data preparation, making the evaluation process easier. It supports a variety of graphic multimodal evaluation sets and video multimodal evaluation sets, a variety of API models and open source models based on PyTorch and HF, and customized prompts and evaluation metrics. After adapting MindSpore Transformers based on VLMEvalKit evaluation framework, it supports loading multimodal large models in MindSpore Transformers for evaluation.

The currently adapted models and supported evaluation datasets are shown in the table below (the remaining models and evaluation datasets are actively being adapted, please pay attention to version updates):

| Adapted models | Supported evaluation datasets                     |
|--|---------------------------------------------------|
| cogvlm2-image-llama3-chat | MME, MMBench, COCO Caption, MMMU_DEV_VAL, TextVQA_VAL |
| cogvlm2-video-llama3-chat | MMBench-Video, MVBench                             |

### Supported Feature Descriptions

1. Supports automatic download of evaluation datasets;
2. Generate results with one click.

### Installation

#### Downloading the Code and Compiling, Installing Dependency Packages

1. Download and modify the code: Due to known issues with open source frameworks running MVBench datasets, it is necessary to modify the code by importing patch. Get [eval.patch](https://github.com/user-attachments/files/17956417/eval.patch) and download and place it in the local directory. When importing the patch, use the absolute path of the patch.

    Execute the following command:

    ```bash
    git clone https://github.com/open-compass/VLMEvalKit.git
    cd VLMEvalKit
    git checkout 78a8cef3f02f85734d88d534390ef93ecc4b8bed
    git apply /path/to/eval.patch
    ```

2. Install dependency packages

    Find the requirements.txt (VLMEvalKit/requirements.txt) file in the downloaded code and modify it to the following content:

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

    Execute Command:

    ```bash
    pip install -r requirements.txt
    ```

#### Installing FFmpeg

For Ubuntu systems follow the steps below to install:

1. Update the system package list and install the system dependency libraries required for compiling FFmpeg and decode.

      ```bash
      apt-get update
      apt-get -y install autoconf automake build-essential libass-dev libfreetype6-dev libsdl2-dev libtheora-dev libtool libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev pkg-config texinfo zlib1g-dev yasm libx264-dev libfdk-aac-dev libmp3lame-dev libopus-dev libvpx-dev
      ```

2. Download the compressed source code package of FFmpeg4.1.11 from the FFmpeg official website, unzip the source code package and enter the decompressed directory; Configure compilation options for FFmpeg: specify the installation path (absolute path) of FFmpeg, generate shared libraries, enable support for specific codecs, and enable no free and GPL licensed features; Compile and install FFmpeg.

      ```bash
      wget --no-check-certificate https://www.ffmpeg.org/releases/ffmpeg-4.1.11.tar.gz
      tar -zxvf ffmpeg-4.1.11.tar.gz
      cd ffmpeg-4.1.11
      ./configure --prefix=/{path}/ffmpeg-xxx --enable-shared --enable-libx264 --enable-libfdk-aac --enable-libmp3lame --enable-libopus --enable-libvpx --enable-nonfree --enable-gpl
      make && make install
      ```

Install OpenEuler system according to the following steps:

1. Download the compressed source code package of FFmpeg4.1.11 from the FFmpeg official website, unzip the source code package and enter the decompressed directory; Configure compilation options for FFmpeg: specify the installation path (absolute path) for FFmpeg; Compile and install FFmpeg.

      ```bash
      wget --no-check-certificate https://www.ffmpeg.org/releases/ffmpeg-4.1.11.tar.gz
      tar -zxvf ffmpeg-4.1.11.tar.gz
      cd ffmpeg-4.1.11
      ./configure --enable-shared --disable-x86asm --prefix=/path/to/ffmpeg
      make && make install
      ```

2. Configure environment variables, `FFMPEG-PATH` requires specifying the absolute path for installing FFmpeg so that the system can correctly locate and use FFmpeg and its related libraries.

      ```bash
      vi ~/.bashrc
      export FFMPEG_PATH=/path/to/ffmpeg/
      export LD_LIBRARY_PATH=$FFMPEG_PATH/lib:$LD_LIBRARY_PATH
      source ~/.bashrc
      ```

#### Installing Decord

Install Ubuntu system according to the following steps:

1. Pull the Decord code, enter the Decord directory, initialize and update Decord dependencies, and execute the following command:

      ```bash
      git clone https://github.com/dmlc/decord.git
      cd decord
      ```

2. Create and enter the `build` directory, configure the compilation options for Decord, disable CUDA support, enable Release mode (optimize performance), specify the installation path for FFmpeg, and compile the Decord library. Copy the compiled libdecord.so library file to the system library directory and to the `python` directory of `decord`.

      ```bash
      mkdir build
      cd build
      cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -DFFMPEG_DIR=/{path}/ffmpeg-4.1.11 && make
      cp libdecord.so /usr/local/lib/
      cp libdecord.so ../python/decord/libdecord.so
      ```

3. Go to the python folder in the `decord` directory, install the numpy dependency, and install the python package for Decord. Add the library path (absolute path) of FFmpeg to the environment variable `LD_LIBRARY_PATH` to ensure that the runtime can find the shared library of FFmpeg.

      ```bash
      cd /path/to/decord/python
      pip install numpy
      python setup.py install
      export LD_LIBRARY_PATH=/path/to/ffmpeg-4.1.11/lib/:$LD_LIBRARY_PATH
      ```

4. Execute Python commands to test if the Decord installation is successful. If there are no errors, it means the installation is successful.

      ```bash
      python -c "import decord; from decord import VideoReader"
      ```

For OpenEuler systems follow the steps below to install:

1. Pull the Decord code and enter the `decord` directory.

      ```bash
      git clone --recursive https://github.com/dmlc/decord
      cd decord
      ```

2. Create and enter the build directory, configure the compilation options for Decord, specify the installation path (absolute path) for ffmpeg, and compile the `decord` library; Enter the `python` folder in the `decord` directory, configure environment variables, and specify `PYTHONPATH`; Install the python package for Decord.

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

3. Execute python commands to test if the Decord installation is successful. If there are no errors, it means the installation is successful.

      ```bash
      python -c "import decord; from decord import VideoReader"
      ```

### Evaluation

#### Preparations Before Evaluation

1. Create a new directory, for example named `model_dir`, to store the model yaml file;
2. Place the model inference yaml configuration file (predict_xxx_. yaml) in the directory created in the previous step. For details, Please refer to the inference content of description documents for each model in the [model library](../start/models.md);
3. Configure the yaml file.

    Using [predict_cogvlm2_image_llama3_chat_19b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/cogvlm2/predict_cogvlm2_image_llama3_chat_19b.yaml) configuration as an example:

    ```yaml
    load_checkpoint: "/{path}/model.ckpt"  # Specify the path to the weights file
    model:
      model_config:
        use_past: True                         # Turn on incremental inference
        is_dynamic: False                       # Turn off dynamic shape

      tokenizer:
        vocab_file: "/{path}/tokenizer.model"  # Specify the tokenizer file path
    ```

    Configure the yaml file. Refer to [configuration description](../appendix/conf_files.md).
4. The MMBench-Video dataset evaluation requires the use of the GPT-4 Turbo model for evaluation and scoring. Please prepare the corresponding API Key in advance and put it in the VLMEvalKit/.env file as follows:

   ```text
   OPENAI_API_KEY=your_apikey
   ```

5. At the beginning of MVBench dataset evaluation, if you are prompted to enter the HuggingFace key, please follow the prompts to ensure the normal execution of subsequent evaluation.

#### Pulling Up the Evaluation Task

Execute the script in the root directory of the MindSpore Transformers local code repository: [run_vlmevalkit.sh](https://gitee.com/mindspore/mindformers/blob/dev/toolkit/benchmarks/run_vlmevalkit.sh).

Execute the following command to initiate the evaluation task:

```shell
#!/bin/bash

source toolkit/benchmarks/run_vlmevalkit.sh \
 --data MMMU_DEV_VAL \
 --model cogvlm2-image-llama3-chat \
 --verbose \
 --work_dir /path/to/cogvlm2-image-eval-result \
 --model_path model_dir
```

### Evaluation Parameters

| Parameters      | Type  | Descriptions                                                                                                                               | Compulsory(Y/N)|
|-----------------|-----|--------------------------------------------------------------------------------------------------------------------------------------------|------|
| `--data`        | str | Name of the dataset, multiple datasets can be passed in, split by spaces.                                                                  | Y    |
| `--model`       | str | Name of the model.                                                                                                                         | Y    |
| `--verbose`     | /   | Outputs logs from the evaluation run.                                                                                                      | N    |
| `--work_dir`    | str | Directory for storing evaluation results. By default, evaluation results are stored in the `outputs` folder of the current execution directory by default. | N    |
| `--model_path`  | str | The folder path containing the model configuration file.                                                                                   | Y    |
| `--register_path`| str | The absolute path of the directory where the cheat code is located. For example, the model directory under the [research](https://gitee.com/mindspore/mindformers/tree/dev/research) directory. | No(The cheat code is required)     |

If the server does not support online downloading of image datasets due to network limitations, you can upload the downloaded .tsv dataset file to the ~/LMUData directory on the server for offline evaluation. (For example: ~/LMUData/MME.tsv or ~/LMUData/MMBench_DEV_EN.tsv or ~/LMUData/COCO_VAL.tsv)

### Viewing Review Results

After evaluating in the above way, find the file ending in .json or .csv in the directory where the evaluation results are stored to view the evaluation results.

The results of the evaluation examples are as follows, where `Bleu` and `ROUGE_L` denote the metrics for evaluating the quality of the translation, and `CIDEr` denotes the metrics for evaluating the image description task.

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

## Using the VideoBench Dataset for Model Evaluation

### Overview

[Video-Bench](https://github.com/PKU-YuanGroup/Video-Bench/tree/main) is the first comprehensive evaluation benchmark for Video-LLMs, featuring a three-level ability assessment that systematically evaluates models in video-exclusive understanding, prior knowledge incorporation, and video-based decision-making abilities.

### Preparations Before Evaluation

1. Download Dataset

    Download [Videos of Video-Bench](https://huggingface.co/datasets/LanguageBind/Video-Bench), place it in the following directory format after decompression:

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

2. Download Json

    Download [Jsons of Video-Bench](https://github.com/PKU-YuanGroup/Video-Bench/tree/main?tab=readme-ov-file), place it in the following directory format after decompression:

    ```text
    egs/Video-Bench/
      └── Eval_QA
            ├── Youcook2_QA_new.json and other json files
            ...
    ```

3. Download the correct answers to all questions

    Download [Answers of Video-Bench](https://huggingface.co/spaces/LanguageBind/Video-Bench/resolve/main/file/ANSWER.json).

> Notes: The text data in Video-Bench is stored in the path format of 'egs/VideoBench/Eval-QA'(The directory should have at least two layers, and the last layer should be `EvalQA`); The video data in Video-Bench is stored in the path format of "egs/VideoBench/Eval_video"(The directory should have at least two layers, and the last layer should be `Eval_video`).

### Evaluation

The execution script path can refer to the link: [eval_with_videobench.py](https://gitee.com/mindspore/mindformers/blob/dev/toolkit/benchmarks/eval_with_videobench.py).

#### Executing Inference Script to Obtain Inference Results

```shell
python toolkit/benchmarks/eval_with_videobench.py \
--model_path model_path \
--dataset_name dataset_name \
--Eval_QA_root Eval_QA_root \
--Eval_Video_root Eval_Video_root \
 --chat_conversation_output_folder output
```

> The parameter `Eval_QA_root` path is filled in the previous directory of Eval-QA; The parameter `Eval_Video_root` path is filled in the previous directory of Eval_video.

**Parameters Description**

| **Parameters**                      | **Compulsory(Y/N)** | **Description**                                                                                                 |
|-------------------------------------|---------------------|-----------------------------------------------------------------------------------------------------------------|
| `--model_path`                      | Y                   | The folder path for storing model related files, including model configuration files and model vocabulary files. |
| `--dataset_name`                    | N                   | Evaluation datasets name, default to None, evaluates all subsets of VideoBench.                                 |
| `--Eval_QA_root`                    | Y                   | Directory for storing JSON files of VideoBench dataset.                         |
| `--Eval_Video_root`                 | Y                   | The video file directory for storing the VideoBench dataset.                                                    |
| `--chat_conversation_output_folder` | N                   | Directory for generating result files. By default, it is stored in the Chat_desults folder of the current directory.                                                                         |

After running, a dialogue result file will be generated in the chat_conversation_output_folder directory.

#### Evaluating and Scoring Based on the Generated Results

Video-Bench can evaluate the answers generated by the model using ChatGPT or T5, and ultimately obtain the final scores for 13 subsets of data.

For example, using ChatGPT for evaluation and scoring:

```shell
python Step2_chatgpt_judge.py \
--model_chat_files_folder ./Chat_results \
--apikey sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
--chatgpt_judge_output_folder ./ChatGPT_Judge

python Step3_merge_into_one_json.py \
--chatgpt_judge_files_folder ./ChatGPT_Judge \
--merge_file ./Video_Bench_Input.json
```

The script path in the above evaluation scoring command is: [Step2_chatgpt_judge.py](https://github.com/PKU-YuanGroup/Video-Bench/blob/main/Step2_chatgpt_judge.py), or [Step3_merge_into_one_json.py](https://github.com/PKU-YuanGroup/Video-Bench/blob/main/Step3_merge_into_one_json.py).

Since ChatGPT may answer some formatting errors, you need to run below Step2_chatgpt_judge.py multiple times to ensure that each question is validated by chatgpt.

## FAQ

1. Use Harness or VLMEvalKit for evaluation, when loading the HuggingFace datasets, report `SSLError`:

   Refer to [SSL Error reporting solution](https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models).

   Note: Turning off SSL verification is risky and may be exposed to MITM. It is only recommended to use it in the test environment or in the connection you fully trust.

2. An `AssertionError` occurs when MVBench dataset is used in VLMEvalKit for evaluation:

   Because the open source framework `VLMEvalKit` has known problems when running `MVBench` dataset. Modify the file by referring to the [issue](https://github.com/open-compass/VLMEvalKit/issues/888) of the open-source framework, or delete the files generated during the evaluation and run the command again (specified by the `--work_dir` parameter, in the `outputs` folder of the current execution directory by default).