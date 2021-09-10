# Function differences with torchtext.data.functional.sentencepiece_numericalizer

## torchtext.data.functional.sentencepiece_numericalizer

```python
torchtext.data.functional.sentencepiece_numericalizer(
    sp_model
    )
```

## mindspore.dataset.text.transforms.SentencePieceTokenizer

```python
class mindspore.dataset.text.transforms.SentencePieceTokenizer(
    mode,
    out_type
    )
```

## Differences

PyTorch: A sentencepiece model to numericalize a text sentence into a generator according to the ids.

MindSpore: According to the incoming sentencepiece model, the input text is segmented and marked; the output type is string or int type.

## Code Example

```python
import mindspore.dataset as ds
from mindspore.dataset import text
from mindspore.dataset.text import SentencePieceModel, SPieceTokenizerOutType
from torchtext.data.functional import sentencepiece_numericalizer

# In MindSpore, return tokenizer from vocab object.
sentence_piece_vocab_file = "/path/to/datasets/1.txt"

vocab = text.SentencePieceVocab.from_file(
    [sentence_piece_vocab_file],
    5000,
    0.9995,
    SentencePieceModel.UNIGRAM,
    {})
tokenizer = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.INT)
text_file_dataset_dir = "/path/to/datasets/2.txt"
text_file_dataset1 = ds.TextFileDataset(dataset_files=text_file_dataset_dir)
text_file_dataset = text_file_dataset1.map(operations=tokenizer)

for item in text_file_dataset:
    print(item[0])
    break
# Out:
# [ 165   28    8   11 4746 1430    4]

# In torch, return the sentencepiece model according to the input model path.
sp_id_generator = sentencepiece_numericalizer(sp_model)
list_a = ["sentencepiece encode as pieces", "examples to   try!"]
list(sp_id_generator(list_a))
```
