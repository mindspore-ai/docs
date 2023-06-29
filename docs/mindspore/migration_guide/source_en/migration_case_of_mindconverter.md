# Using Mindcoverter to Perform Migration

Translator: [ChanJiatao](https://gitee.com/ChanJiatao)

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/migration_guide/source_en/migration_case_of_mindconverter.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

## Overview

To convert a PyTorch model to MindSpore model, you first need to export the PyTorch model to an ONNX model, and then use the MindConverter CLI tool to perform script and weight migration.

HuggingFace Transformers is a mainstream natural language processing third-party library under the PyTorch framework. We take BertForMaskedLM in Transformer as an example to demonstrate the migration process.

## Environment Preparation

In this case, the following third-party Python libraries need to be installed:

```bash
pip install torch==1.5.1
pip install transformers==4.2.2
pip install mindspore==1.2.0
pip install mindinsight==1.2.0
pip install onnx
```

> When installing 'ONNX' third-party libraries, you need to install 'protobuf-compiler' and 'libprotoc-dev' in advance. If there are no above two libraries, you can use the command 'apt-get install protobuf-compiler libprotoc-dev' to install them.

## ONNX Model Export

First instantiate the BertForMaskedLM in HuggingFace and the corresponding tokenizer (you need to download the model weight, vocabulary, model configuration and other data when you use it for the first time).  

Regarding the use of HuggingFace, this article just gives a brief presentation. For detail, please refer to the [HuggingFace user documentation](https://huggingface.co/transformers/model_doc/bert.html).  

The model can predict the words that are masked in the sentence.

```python
from transformers.models.bert import BertForMaskedLM, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
```

We use the model for reasoning and generate several sets of test cases to verify the correctness of the model migration.

Here we take a sentence as an example: `china is a poworful country, its capital is beijing.`

We mask `beijing`, and then input `china is a poworful country, its capital is [MASK].` to the model, the expected output of the model should be `beijing`.

```python
import numpy as np
import torch

text = "china is a poworful country, its capital is [MASK]."
tokenized_sentence = tokenizer(text)

mask_idx = tokenized_sentence["input_ids"].index(tokenizer.convert_tokens_to_ids("[MASK]"))
input_ids = np.array([tokenized_sentence["input_ids"]])
attention_mask = np.array([tokenized_sentence["attention_mask"]])
token_type_ids = np.array([tokenized_sentence["token_type_ids"]])

# Get [MASK] token id.
print(f"MASK TOKEN id: {mask_idx}")
print(f"Tokens: {input_ids}")
print(f"Attention mask: {attention_mask}")
print(f"Token type ids: {token_type_ids}")

model.eval()
with torch.no_grad():
    predictions = model(input_ids=torch.tensor(input_ids),
                        attention_mask=torch.tensor(attention_mask),
                        token_type_ids=torch.tensor(token_type_ids))
    predicted_index = torch.argmax(predictions[0][0][mask_idx])
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(f"Pred id: {predicted_index}")
    print(f"Pred token: {predicted_token}")
    assert predicted_token == "beijing"
```

```text
MASK TOKEN id: 12
Tokens: [[  101  2859  2003  1037 23776 16347  5313  2406  1010  2049  3007  2003
    103  1012   102]]
Attention mask: [[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]]
Token type ids: [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
Pred id: 7211
Pred token: beijing

```

HuggingFace provides tools to export ONNX models. The following methods can be used to export HuggingFace's pre-trained models as ONNX models:

```python
from pathlib import Path
from transformers.convert_graph_to_onnx import convert

# Exported onnx model path.
saved_onnx_path = "./exported_bert_base_uncased/bert_base_uncased.onnx"
convert("pt", model, Path(saved_onnx_path), 11, tokenizer)
```

```text
Creating folder exported_bert_base_uncased
Using framework PyTorch: 1.5.1+cu101
Found input input_ids with shape: {0: 'batch', 1: 'sequence'}
Found input token_type_ids with shape: {0: 'batch', 1: 'sequence'}
Found input attention_mask with shape: {0: 'batch', 1: 'sequence'}
Found output output_0 with shape: {0: 'batch', 1: 'sequence'}
Ensuring inputs are in correct order
position_ids is not present in the generated input list.
Generated inputs order: ['input_ids', 'attention_mask', 'token_type_ids']
```

According to the print information, we can see that the exported ONNX model has 3 input nodes: `input_ids`, `token_type_ids`, `attention_mask`, and the corresponding input axis. The output node has an `output_0`.

So far, the ONNX model is successfully exported, and then the accuracy of the exported ONNX model is verified (the ONNX model export process is executed on the ARM environment, and the user may need to compile and install the PyTorch and Transformers third-party library).

## ONNX Model Validation

We still use the sentence from the PyTorch model for reasoning `china is a poworful country, its capital is [MASK].` as input to observe whether the ONNX model performs as expected.

```python
import onnx
import onnxruntime as ort

model = onnx.load(saved_onnx_path)
sess = ort.InferenceSession(bytes(model.SerializeToString()))
result = sess.run(
    output_names=None,
    input_feed={"input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids}
)[0]
predicted_index = np.argmax(result[0][mask_idx])
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

print(f"ONNX Pred id: {predicted_index}")
print(f"ONNX Pred token: {predicted_token}")
assert predicted_token == "beijing"
```

```text
ONNX Pred id: 7211
ONNX Pred token: beijing

```

As you can see, the exported ONNX model functions are exactly the same as the original PyTorch model. Next, you can use MindConverter to perform script and weight migration!

## Using MindConverter to Perform Model Script and Weight Migration

When MindConverter performs model conversion, it needs to specify the model path (`--model_file`), input node (`--input_nodes`), input node size (`--shape`), and output node (`--output_nodes`).

The generated script output path (`--output`) and conversion report path (`--report`) are optional parameters. The default is the output directory under the current path. If the output directory does not exist, it will be created automatically.

```bash
mindconverter --model_file ./exported_bert_base_uncased/bert_base_uncased.onnx --shape 1,128 1,128 1,128  \
               --input_nodes input_ids token_type_ids attention_mask  \
               --output_nodes output_0  \
               --output ./converted_bert_base_uncased  \
               --report ./converted_bert_base_uncased
```

```text
MindConverter: conversion is completed.

```

**Seeing "MindConverter: conversion is completed." means that the model has been successfully converted!**

After the conversion is completed, the following files are generated in this directory: -Model definition script(suffix is .py) -Weight ckpt file(suffix is .ckpt) -Weight mapping before and after migration(suffix is .json) -Conversion report(suffix is .txt).

Check the conversion result with the ls command.

```bash
ls ./converted_bert_base_uncased
```

```text
bert_base_uncased.ckpt  report_of_bert_base_uncased.txt
bert_base_uncased.py    weight_map_of_bert_base_uncased.json

```

You can see that all files have been generated.

After the migration is complete, we will verify the accuracy of the model after the migration.

## MindSpore Model Validation

We still use `china is a poworful country, its capital is [MASK].` as input to observe whether the model performance after migration meets expectations.

Since the tool needs to freeze the size of the model during conversion, when using MindSpore for reasoning and verification, the sentence needs to be filled (Pad) to a fixed length. The sentence can be filled with the following function.

In reasoning, the sentence length must be less than the maximum sentence length during conversion (here our longest sentence length is 128, which is specified by `--shape 1,128` in the conversion phase).

```python
def padding(input_ids, attn_mask, token_type_ids, target_len=128):
    length = len(input_ids)
    for i in range(target_len - length):
        input_ids.append(0)
        attn_mask.append(0)
        token_type_ids.append(0)
    return np.array([input_ids]), np.array([attn_mask]), np.array([token_type_ids])
```

```python
from converted_bert_base_uncased.bert_base_uncased import Model as MsBert
from mindspore import load_checkpoint, load_param_into_net, context, Tensor


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
padded_input_ids, padded_attention_mask, padded_token_type = padding(tokenized_sentence["input_ids"],
                                                                     tokenized_sentence["attention_mask"],
                                                                     tokenized_sentence["token_type_ids"],
                                                                     target_len=128)
padded_input_ids = Tensor(padded_input_ids)
padded_attention_mask = Tensor(padded_attention_mask)
padded_token_type = Tensor(padded_token_type)

model = MsBert()
param_dict = load_checkpoint("./converted_bert_base_uncased/bert_base_uncased.ckpt")
not_load_params = load_param_into_net(model, param_dict)
output = model(padded_attention_mask, padded_input_ids, padded_token_type)

assert not not_load_params

predicted_index = np.argmax(output.asnumpy()[0][mask_idx])
print(f"ONNX Pred id: {predicted_index}")
assert predicted_index == 7211
```

```text
ONNX Pred id: 7211

```

So far, the script and weight migration using MindConverter is complete.

Users can perform training, inference, and deployment scripts according to application scenarios to implement personal business logic.

## Frequently Asked Questions

**Q: How to modify the shape specifications such as Batch size and Sequence length of the migrated script so that the model can support data inference and training of any size?**

A: After the migration, the script has shape restrictions, which are usually caused by the Reshape operator or other operators that involve changes in the tensor arrangement. Taking the above Bert migration as an example, first create two global variables to represent the expected batch size and sentence length, then modify the target size of the Reshape operation and replace it with the corresponding batch size and sentence length global variables.

**Q: The definition of the class name in the generated script does not conform to the developer's habits, such as "class Module0(nn.Cell)". Will manual modification affect the weight loading after conversion?**

A: The weight loading is only related to the variable name and class structure, so the class name can be modified without affecting the weight loading. If the structure of the class needs to be adjusted, the corresponding weight names needs to be modified synchronously to adapt to the structure of the migrated model.
