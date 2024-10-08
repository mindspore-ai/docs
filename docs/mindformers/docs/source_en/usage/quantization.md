# Quantization

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/usage/quantization.md)

## Overview

Quantization is an important technology for compressing foundation models. It converts floating-point parameters in a model into low-precision integer parameters to compress the parameters. As the parameters and specifications of a model increase, quantization can effectively reduce the model storage space and loading time during model deployment, improving the model inference performance.

MindFormers integrates the MindSpore Golden Stick tool component to provide a unified quantization inference process, facilitating out-of-the-box use.

## Auxiliary Installation

Before using the quantization inference function, install MindSpore Golden Stick. For details, see [Installation](https://gitee.com/mindspore/golden-stick/blob/master/README.md/#documents).

Download the source code and go to the `golden_stick` directory.

```bash
bash build.sh
pip install output/mindspore_gs-0.6.0-py3-none-any.whl
```

Run the following commands to verify the installation:

```bash
pip show mindspore_gs

# Name: mindspore_gs
# Version: 0.6.0
# Summary: A MindSpore model optimization algorithm set..
# Home-page: https://www.mindspore.cn
# Author: The MindSpore Authors
# Author-email: contact@mindspore.cn
# License: Apache 2.0
```

## Procedure

Based on actual operations, quantization may be decomposed into the following steps:

1. **Selecting a model:**
   Select a language model. Currently, the Llama2_13B and Llama2_70B models support quantization.

2. **Downloading the model weights:**
   Download the weight of the corresponding model from the HuggingFace model library and convert the model to the CKPT format by referring to [Weight Conversion](https://www.mindspore.cn/mindformers/docs/en/dev/function/weight_conversion.html).

3. **Converting the quantization model weight:**
   Run the conversion script `quant_ckpt.py` in the mindspore_gs library to convert the original weight in step 2 to the quantization weight.

4. **Preparing the quantization configuration file:**
   Use the built-in quantization inference configuration file of MindFormers that matches the model. The quantization-related configuration item is `model.model_config.quantization_config`.

   The following uses the `llama2_13b_rtn` quantization model as an example. The default quantization configuration is as follows:

   ```yaml
     quantization_config:
       quant_method: 'rtn'
       weight_dtype: 'int8'
       activation_dtype: None
       kvcache_dtype: None
       outliers_suppression: None
       modules_to_not_convert: ['lm_head']
       algorithm_args: {}
   ```

   | Parameter                  | Attribute| Description                                         | Type | Value Range            |
   | ---------------------- | ---- |:----------------------------------------------| --------- |------------------|
   | quant_method           | Required| Supported quantization algorithm. Currently, only the RTN, Smooth_Quant, and PTQ algorithms are supported.              | str       | rtn/smooth_quant/ptq |
   | weight_dtype           | Required| Quantized weight type. Currently, only int8 is supported.                        | str       | int8/None           |
   | activation_dtype       | Required| Activation type of the parameter. **None** indicates that the original computing type (**compute_dtype**) of the network remains unchanged.     | str       | int8/None        |
   | kvcache_dtype          | Optional| KVCache quantization type. If the value is **None** or not specified, the original KVCache data type remains unchanged.       | str       | int8/None        |
   | outliers_suppression   | Optional| Algorithm type used for abnormal value suppression. Currently, only smooth suppression is supported.       | str       | smooth/None        |
   | modules_to_not_convert | Required| Layer that is not quantized.                                    | List[str] | /                |
   | algorithm_args         | Required| Configurations of different algorithm types for connecting to the MindSpore Golden Stick. For example, **alpha** is set to **0.5** for the Smooth_Quant algorithm.| Dict      | /                |

5. **Executing inference tasks:**
   Implement the inference script based on the `generate` API and run the script to obtain the inference result.

## Using the RTN Quantization Algorithm to Perform A16W8 Quantization Inference Based on the Llama2_13B Model

### Selecting a Model

In this practice, the Llama2-13B model is used for single-device quantization inference.

In this practice, `AutoModel.from_pretrained()` is used to instantiate a model by specifying the models or weight path. You need to create a storage directory in advance.

```shell
mkdir /data/tutorial/llama2_13b_rtn_a16w8_dir
```

> Note: Currently, the AutoModel.from_pretrained() API does not support instantiation by specifying parameters based on the quantized model name.

Directory structure of a single device

```shell
llama2_13b_rtn_a16w8_dir
  ├── predict_llama2_13b_rtn.yaml
  └── llama2_13b_rtn_a16w8.ckpt
```

### Downloading the Model Weights

MindFormers provides pretrained weights and vocabulary files that have been converted for pretraining, fine-tuning, and inference. You can also download the official HuggingFace weights and perform the operations in [Converting Model Weights](#converting-model-weights) before using these weights.

You can download the vocabulary at [tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model).

| Model           |                                                MindSpore Weight                                                |                      HuggingFace Weight                      |
|:----------------|:----------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------:|
| llama2-13b      | [llama2-13b-fp16.ckpt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2-13b-fp16.ckpt) | [Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf) |

> Note: All weights of Llama2 need to be obtained by [submitting an application](https://ai.meta.com/resources/models-and-libraries/llama-downloads) to Meta. If necessary, apply for the weights by yourself.

### Converting Model Weights

Go to the root directory `golden-stick` of the mindspore_gs library and run the quantization weight conversion script.

```bash
python example/ptq/quant_ckpt.py -c /path/to/predict_llama2_13b.yaml -s /path/to/boolq/dev.jsonl -t boolq -a rtn-a16w8 > log_rtn_a16w8_quant 2>&
```

Set `load_checkpoint` in `predict_llama2_13b.yaml` to the path for storing the original weight downloaded in the previous step.

During the conversion, `boolq` is used to verify the datasets. You can download it at [the boolq dataset link](https://github.com/svinkapeppa/boolq). After the download is complete, specify the path for storing `dev.jsonl` in the preceding script.

Run the script to copy the generated quantization weight file to the `llama2_13b_rtn_a16w8_dir` directory.

```shell
cp output/rtn-a16w8_ckpt/rank_0/rtn-a16w8.ckpt /data/tutorial/llama2_13b_rtn_a16w8_dir/llama2_13b_rtn_a16w8.ckpt
```

### Preparing the Quantization Configuration File

The configuration file [predict_llama2_13b_rtn.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/predict_llama2_13b_rtn.yaml) is provided in MindFormers. You need to copy it to the `llama2_13b_rtn_a16w8_dir` directory.

```shell
cp configs/llama2/predict_llama2_13b_rtn.yaml /data/tutorial/llama2_13b_rtn_a16w8_dir
```

### Executing Inference Tasks

1. **Script instances**

   Replace the [run_llama2_generate.py](https://gitee.com/mindspore/mindformers/blob/dev/scripts/examples/llama2/run_llama2_generate.py) script in MindFormers with the following code.

   In this practice, the quantization model is instantiated based on the `AutoModel.from_pretrained()` API. You need to modify the parameters in the API to the created directory.

   You can call the `generate` API to obtain the inference result. For details about the parameters, see the [AutoModel](https://www.mindspore.cn/mindformers/docs/en/dev/mindformers/mindformers.AutoModel.html) and [generate](https://www.mindspore.cn/mindformers/docs/en/dev/generation/mindformers.generation.GenerationMixin.html) API documents.

   ```python
   """llama2 predict example."""
   import argparse
   import os

   import mindspore as ms
   from mindspore import Tensor, Model
   from mindspore.common import initializer as init

   from mindformers import AutoModel
   from mindformers import MindFormerConfig, logger
   from mindformers.core.context import build_context
   from mindformers.core.parallel_config import build_parallel_config
   from mindformers.models.llama import LlamaTokenizer
   from mindformers.trainer.utils import transform_and_load_checkpoint


   def main(config_path, use_parallel, load_checkpoint):
       # Construct the input content.
       inputs = ["I love Beijing, because",
                 "LLaMA is a",
                 "Huawei is a company that"]
       batch_size = len(inputs)

       # Generate model configurations based on the YAML file.
       config = MindFormerConfig(config_path)
       config.use_parallel = use_parallel
       device_num = os.getenv('MS_WORKER_NUM')
       logger.info(f"Use device number: {device_num}, it will override config.model_parallel.")
       config.parallel_config.model_parallel = int(device_num) if device_num else 1
       config.parallel_config.data_parallel = 1
       config.parallel_config.pipeline_stage = 1
       config.load_checkpoint = load_checkpoint

       # Initialize the environment.
       build_context(config)
       build_parallel_config(config)
       model_name = config.trainer.model_name

       # Instantiate a tokenizer.
       tokenizer = LlamaTokenizer.from_pretrained(model_name)
       # Instantiate the model.
       network = AutoModel.from_pretrained("/data/tutorial/llama2_13b_rtn_a16w8_dir",
                                           download_checkpoint=False)
       model = Model(network)

       # Load weights.
       if config.load_checkpoint:
           logger.info("----------------Transform and load checkpoint----------------")
           seq_length = config.model.model_config.seq_length
           input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
           infer_data = network.prepare_inputs_for_predict_layout(input_ids)
           transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

       inputs_ids = tokenizer(inputs, max_length=config.model.model_config.seq_length, padding="max_length")["input_ids"]

       outputs = network.generate(inputs_ids,
                                  max_length=config.model.model_config.max_decode_length,
                                  do_sample=config.model.model_config.do_sample,
                                  top_k=config.model.model_config.top_k,
                                  top_p=config.model.model_config.top_p)
       for output in outputs:
           print(tokenizer.decode(output))


   if __name__ == "__main__":
       parser = argparse.ArgumentParser()
       parser.add_argument('--config_path', default='predict_llama2_7b.yaml', type=str,
                           help='model config file path.')
       parser.add_argument('--use_parallel', action='store_true',
                           help='if run model prediction in parallel mode.')
       parser.add_argument('--load_checkpoint', type=str,
                           help='load model checkpoint path or directory.')

       args = parser.parse_args()
       main(
           args.config_path,
           args.use_parallel,
           args.load_checkpoint
       )
   ```

2. **Startup script**

   MindFormers provides a quick inference script for the `Llama2` model, supporting single-device, multi-device, and multi-batch inferences.

   ```shell
   # Script usage
   bash scripts/examples/llama2/run_llama2_predict.sh PARALLEL CONFIG_PATH CKPT_PATH DEVICE_NUM
   # Parameters
   PARALLEL:    specifies whether to use multi-device inference. 'single' indicates single-device inference, and 'parallel' indicates multi-device inference.
   CONFIG_PATH: model configuration file path.
   CKPT_PATH:   path of the model weight file.
   DEVICE_NUM:  number of used devices. This parameter takes effect only when multi-device inference is enabled.
   ```

   Single-Device Inference

   ```bash
   bash scripts/examples/llama2/run_llama2_predict.sh single /data/tutorial/llama2_13b_w8a16_dir/predict_llama2_13b_w8a16.yaml /data/tutorial/llama2_13b_w8a16_dir/llama2_13b_w8a16.ckpt
   ```

   The inference result is as follows:

   ```text
   'text_generation_text': [I love Beijing, because it is a city that is constantly constantly changing. I have been living here for ......]
   'text_generation_text': [LLaMA is a large-scale, open-source, multimodal, multilingual, multitask, and multimodal pretrained language model. It is ......]
   'text_generation_text': [Huawei is a company that has been around for a long time. ......]
   ```
