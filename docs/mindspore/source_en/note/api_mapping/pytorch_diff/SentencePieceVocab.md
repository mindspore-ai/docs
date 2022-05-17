# Function differences with torchtext.data.functional.load_sp_model

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/pytorch_diff/SentencePieceVocab.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## torchtext.data.functional.load_sp_model

```python
torchtext.data.functional.load_sp_model(
    spm
)
```

For more information, see [torchtext.data.functional.load_sp_model](https://pytorch.org/text/0.10.0/data_functional.html#load-sp-model).

## mindspore.dataset.text.utils.SentencePieceVocab

```python
classmindspore.dataset.text.utils.SentencePieceVocab
```

For more information, see [mindspore.dataset.text.utils.SentencePieceVocab](https://www.mindspore.cn/docs/en/r1.7/api_python/dataset_text/mindspore.dataset.text.SentencePieceVocab.html#mindspore.dataset.text.SentencePieceVocab).

## Differences

PyTorch: Load a sentencepiece model for file.

MindSpore: SentencePiece object that is used to perform words segmentation.

## Code Example

```python
import mindspore.dataset as ds
from mindspore.dataset import text
from mindspore.dataset.text import SentencePieceModel, SPieceTokenizerOutType, to_str
from torchtext.data.functional import load_sp_model

# In MindSpore, return tokenizer from vocab object.
sentence_piece_vocab_file = "/path/to/test_sentencepiece/botchan.txt"

vocab = text.SentencePieceVocab.from_file([sentence_piece_vocab_file], 5000, 0.9995,
                                          SentencePieceModel.WORD, {})
tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)
text_file_dataset_dir = "/path/to/testTokenizerData/sentencepiece_tokenizer.txt"
text_file_dataset = ds.TextFileDataset(dataset_files=text_file_dataset_dir)
text_file_dataset = text_file_dataset.map(operations=tokenizer)

for i in text_file_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    ret = to_str(i["text"])
    for key, value in enumerate(ret):
        print(value)
# Out:
# ▁I
# ▁saw
# ▁a
# ▁girl
# ▁with
# ▁a
# ▁telescope.

# In torch, return the sentencepiece model according to the input model path.
sp_model = load_sp_model("m_user.model")
sp_model = load_sp_model(open("m_user.model", 'rb'))
```
