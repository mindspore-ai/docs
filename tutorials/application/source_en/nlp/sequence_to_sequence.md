# Text Translation Implemented by Seq2Seq Model

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/application/source_en/nlp/sequence_to_sequence.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

Sequence to sequence model, also called Seq2Seq model. It is a variant of the recurrent neural network (RNN) that breaks through a limitation of RNN models on the input and output sequence length and maps an input sequence to another output sequence with different length. Therefore, it is commonly used in machine translation.

The Seq2Seq model consists of encoder and decoder. The encoder encodes an input sequence into a vector with fixed length, and the decoder converts the vector into a vector with variable length.

![avatar1](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/application/source_zh_cn/nlp/images/seq2seq_1.png)

> Image source:
>
> <https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb>

Later, an attention mechanism is introduced after encoder and decoder, so that the model performs better in each task.

## Data Preparation

The dataset we use is the **Multi30K dataset**, which is a large-scale dataset containing more than 30,000 images and each image has text descriptions in two languages.

- English description and corresponding German translation
- Five independent and non-translated English and German descriptions that contain different details

Because descriptions of images in different languages collected by the model are independent, the trained model can be better applicable to multi-modal content with noise.

![avatar2](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/application/source_zh_cn/nlp/images/seq2seq_2.png)

> Image source:
>
> Elliott, D., Frank, S., Sima'an, K., & Specia, L. (2016).Multi30K: Multilingual English-German Image Descriptions. CoRR, 1605.00459.

First, we need to install the following dependency:

- BLEU Score calculation: `pip install nltk`

### Data Downloading Module

Use `download` to download data and decompress the `tar.gz` file to a specified folder.

The directory structure of the downloaded dataset is as follows:

```text
home_path/.mindspore_examples
├─test
│      test2016.de
│      test2016.en
│      test2016.fr
│
├─train
│      train.de
│      train.en
│
└─valid
        val.de
        val.en
```

```python
from download import download
from pathlib import Path
from tqdm import tqdm
import os

# Addresses for downloading training, validation, and test datasets
urls = {
    'train': 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz',
    'valid': 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz',
    'test': 'http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_test2016.tar.gz'
}

# Set the storage path to `home_path/.mindspore_examples`.
cache_dir = Path.home() / '.mindspore_examples'

train_path = download(urls['train'], os.path.join(cache_dir, 'train'), kind='tar.gz')
valid_path = download(urls['valid'], os.path.join(cache_dir, 'valid'), kind='tar.gz')
test_path = download(urls['test'], os.path.join(cache_dir, 'test'), kind='tar.gz')
```

### Data Preprocessing

When using data to perform operations such as model training, we need to preprocess the data as follows:

1. Load the dataset. Currently, the data is text in the form of sentences and requires word tokenization, that is, split sentences into independent tokens (characters or words).
2. Map each token to a numeric index starting from 0 (to save storage space and filter out tokens with low frequency). A set composed of tokens and numeric indexes is called vocabulary.
3. Add special placeholders to indicate the start and end of a sequence and unify the sequence length, and create a data iterator.

#### Data Loader

```python
import re


class Multi30K():
    """Multi30K dataset loader

    Load the Multi30K dataset and process it as a Python iteration object.

    """

    def __init__(self, path):
        self.data = self._load(path)

    def _load(self, path):

        def tokenize(text):
            # Tokenize sentences and unify the case.
            text = text.rstrip()
            return [tok.lower() for tok in re.findall(r'\w+|[^\w\s]', text)]

        # Read Multi30K data and perform word tokenization.
        members = {i.split('.')[-1]: i for i in os.listdir(path)}
        de_path = os.path.join(path, members['de'])
        en_path = os.path.join(path, members['en'])
        with open(de_path, 'r') as de_file:
            de = de_file.readlines()[:-1]
            de = [tokenize(i) for i in de]
        with open(en_path, 'r') as en_file:
            en = en_file.readlines()[:-1]
            en = [tokenize(i) for i in en]

        return list(zip(de, en))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
```

```python
train_dataset, valid_dataset, test_dataset = Multi30K(train_path), Multi30K(valid_path), Multi30K(test_path)
```

Test the decompression and word tokenization results and print the English and German text of test dataset, where we can see that each word and punctuation have been separated.

```python
for de, en in test_dataset:
    print(f'de = {de}')
    print(f'en = {en}')
    break
```

Output:

```text
de = ['ein', 'mann', 'mit', 'einem', 'orangefarbenen', 'hut', ',', 'der', 'etwas', 'anstarrt', '.']
en = ['a', 'man', 'in', 'an', 'orange', 'hat', 'starring', 'at', 'something', '.']
```

#### Vocabulary

```python
class Vocab:
    """Build a vocabulary based on the word frequency dictionary.""

    special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']

    def __init__(self, word_count_dict, min_freq=1):
        self.word2idx = {}
        for idx, tok in enumerate(self.special_tokens):
            self.word2idx[tok] = idx

        # Filter out tokens with low frequency.
        filted_dict = {
            w: c
            for w, c in word_count_dict.items() if c >= min_freq
        }
        for w, _ in filted_dict.items():
            self.word2idx[w] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        self.bos_idx = self.word2idx['<bos>'] # Special placeholder: start of a sequence
        self.eos_idx = self.word2idx['<eos>'] # Special placeholder: end of a sequence
        self.pad_idx = self.word2idx['<pad>'] # Special placeholder: supplementary character
        self.unk_idx = self.word2idx['<unk>'] # Special placeholders: low-frequency tokens or unknown tokens

    def _word2idx(self, word):
        """Map words to numeric indexes."""
        if word not in self.word2idx:
            return self.unk_idx
        return self.word2idx[word]

    def _idx2word(self, idx):
        """Map numeric indexes to words."""
        if idx not in self.idx2word:
            raise ValueError('input index is not in vocabulary.')
        return self.idx2word[idx]

    def encode(self, word_or_list):
        """Map a word or word array to a numeric index or numeric index array."""
        if isinstance(word_or_list, list):
            return [self._word2idx(i) for i in word_or_list]
        return self._word2idx(word_or_list)

    def decode(self, idx_or_list):
        """Map a numeric index or numeric index array to a single word or word array."""
        if isinstance(idx_or_list, list):
            return [self._idx2word(i) for i in idx_or_list]
        return self._idx2word(idx_or_list)

    def __len__(self):
        return len(self.word2idx)
```

After using the user-defined word frequency dictionary to test, we can see that the vocabulary has removed the token c whose frequency is less than 2. Since four default special placeholders are added, the overall length of vocabulary is 4 - 1 + 4 = 7.

```python
word_count = {'a':20, 'b':10, 'c':1, 'd':2}

vocab = Vocab(word_count, min_freq=2)
len(vocab)
```

Output:

```text
7
```

Use `Counter` and `OrderedDict` in `collections` to calculate the frequency of each word in the entire English and German text. Build a word frequency dictionary, and then convert the dictionary into a vocabulary.

There is a tip when allocating numeric indexes: Map common tokens to indexes with smaller values to save spaces.

```python
from collections import Counter, OrderedDict

def build_vocab(dataset):
    de_words, en_words = [], []
    for de, en in dataset:
        de_words.extend(de)
        en_words.extend(en)

    de_count_dict = OrderedDict(sorted(Counter(de_words).items(), key=lambda t: t[1], reverse=True))
    en_count_dict = OrderedDict(sorted(Counter(en_words).items(), key=lambda t: t[1], reverse=True))

    return Vocab(de_count_dict, min_freq=2), Vocab(en_count_dict, min_freq=2)
```

```python
de_vocab, en_vocab = build_vocab(train_dataset)
print('Unique tokens in de vocabulary:', len(de_vocab))
```

Output:

```text
Unique tokens in de vocabulary: 7882
```

#### Data Iterator

The last step of data preprocessing is to create a data iterator. After the previous processing (including batch processing, adding start and end placeholders, and unifying the sequence length), we return the data as tensors.

The following parameters are required for creating a data iterator:

- `dataset`: dataset after tokenization
- `de_vocab`: German vocabulary
- `en_vocab`: English vocabulary
- `batch_size`: batch size, that is, the number of sequences in a batch
- `max_len`: maximum length of the sequence. The value is equal to the maximum valid text length plus 2 (placeholders for the start and end of a sequence). If the length is less than the maximum, supplement the length. If the length exceeds the maximum, discard the excess.
- `drop_remainder`: indicates whether to discard the remaining batch.

```python
import mindspore

class Iterator():
    """Create a data iterator."""
    def __init__(self, dataset, de_vocab, en_vocab, batch_size, max_len=32, drop_reminder=False):
        self.dataset = dataset
        self.de_vocab = de_vocab
        self.en_vocab = en_vocab

        self.batch_size = batch_size
        self.max_len = max_len
        self.drop_reminder = drop_reminder

        length = len(self.dataset) // batch_size
        self.len = length if drop_reminder else length + 1  # The number of batches

    def __call__(self):
        def pad(idx_list, vocab, max_len):
            """Unify the sequence length and record the valid length."""
            idx_pad_list, idx_len = [], []
            # If the current sequence length exceeds the maximum, the excess part is discarded. If the current sequence length is less than the maximum length, the length is padded with placeholders.
            for i in idx_list:
                if len(i) > max_len - 2:
                    idx_pad_list.append(
                        [vocab.bos_idx] + i[:max_len-2] + [vocab.eos_idx]
                    )
                    idx_len.append(max_len)
                else:
                    idx_pad_list.append(
                        [vocab.bos_idx] + i + [vocab.eos_idx] + [vocab.pad_idx] * (max_len - len(i) - 2)
                    )
                    idx_len.append(len(i) + 2)
            return idx_pad_list, idx_len

        def sort_by_length(src, trg):
            """Sort German/English field lengths."""
            data = zip(src, trg)
            data = sorted(data, key=lambda t: len(t[0]), reverse=True)
            return zip(*list(data))

        def encode_and_pad(batch_data, max_len):
            """Convert text data in batches into numeric indexes and unify the length of each sequence."""
            # Convert tokens in current batches into indexes.
            src_data, trg_data = zip(*batch_data)
            src_idx = [self.de_vocab.encode(i) for i in src_data]
            trg_idx = [self.en_vocab.encode(i) for i in trg_data]

            # Unify the sequence length.
            src_idx, trg_idx = sort_by_length(src_idx, trg_idx)
            src_idx_pad, src_len = pad(src_idx, de_vocab, max_len)
            trg_idx_pad, _ = pad(trg_idx, en_vocab, max_len)

            return src_idx_pad, src_len, trg_idx_pad

        for i in range(self.len):
            # Obtain data in current batches.
            if i == self.len - 1 and not self.drop_reminder:
                batch_data = self.dataset[i * self.batch_size:]
            else:
                batch_data = self.dataset[i * self.batch_size: (i+1) * self.batch_size]

            src_idx, src_len, trg_idx = encode_and_pad(batch_data, self.max_len)
            # Convert sequence data into tensors.
            yield mindspore.Tensor(src_idx, mindspore.int32), \
                mindspore.Tensor(src_len, mindspore.int32), \
                mindspore.Tensor(trg_idx, mindspore.int32)

    def __len__(self):
        return self.len
```

```python
train_iterator = Iterator(train_dataset, de_vocab, en_vocab, batch_size=128, max_len=32, drop_reminder=True)
valid_iterator = Iterator(valid_dataset, de_vocab, en_vocab, batch_size=128, max_len=32, drop_reminder=False)
test_iterator = Iterator(test_dataset, de_vocab, en_vocab, batch_size=128, max_len=32, drop_reminder=False)
```

## Model Building

### Encoder

In the encoder, we input a sequence $X=\{x_1, x_2, ..., x_T\}$, convert it into a vector at the embedding layer, cyclically calculate the hidden state $H=\{h_1, h_2, ..., h_T\}$, and return the context vector $z=h_T$ in the last hidden state.

There are many ways to implement the encoder. Here, we use the gated recurrent units (GRU). Based on the RNN, it introduces the gate mechanism to control the input hidden state and the information output from the hidden state. The update gate (also called memory gate and represented by $z_t$) is used to control a degree to which state information $h_{t-1}$ at the previous time is brought into the current state $h_t$. The reset gate (represented by $r_t$) controls how much information in the previous state $h_t$ is written to the current candidate set $n_t$.

$$h_t = \text{RNN}(e(x_t), h_{t-1})$$

Generally, bidirectional GRUs are used for text translation. That is, the text before and after the translation is considered during training. Each layer of the bidirectional GRUs consists of two RNNs. The hidden state of the forward RNN is calculated cyclically from left to right, and the hidden state of the reverse RNN is calculated from right to left. The formula is as follows:

$$\begin{align*}
h_t^\rightarrow &= \text{EncoderGRU}^\rightarrow(e(x_t^\rightarrow),h_{t-1}^\rightarrow)\\
h_t^\leftarrow &= \text{EncoderGRU}^\leftarrow(e(x_t^\leftarrow),h_{t-1}^\leftarrow)
\end{align*}$$

After observing the last word in a sentence, each RNN network outputs a context vector. The output of the forward RNN is $z^\rightarrow=h_T^\rightarrow$, and the output of the reverse RNN is $z^\leftarrow=h_T^\leftarrow$.

![avatar3](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/application/source_zh_cn/nlp/images/seq2seq_3.png)

> Image source:
>
> <https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb>

The encoder returns `outputs` and `hidden`.

- `outputs` indicates the hidden state of the top layer of the bidirectional GRUs. The shape is \[max_len, batch_size, hid_dim * num_directions\]. Take the time $t=1$ as an example, its output is a concatenation of the top hidden state at the time $t=1$ in the forward RNN and the reverse RNN at the time $t=T$, which is $h_1 = [h_1^\rightarrow; h_{T}^\leftarrow]$.

- `hidden` indicates the final hidden state of each layer, that is, the context vector mentioned above. To use the initial hidden state $s_0$ as the decoder, a single context vector $z$ is required because the decoder is not bidirectional. Therefore, we concatenate the two context vectors together, passing them through a fully connected layer $g$, and apply the activation function $tanh$.

$$z=\tanh(g(h_T^\rightarrow, h_T^\leftarrow)) = \tanh(g(z^\rightarrow, z^\leftarrow)) = s_0$$

MindSpore provides GRU interfaces which can be directly invoked during encoder setup. You can set `bidirectional=True` to enable bidirectional GRU.

```python
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

class Encoder(nn.Cell):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, is_ascend):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim) # Embedding layer

        if is_ascend:
            self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True).to_float(compute_dtype) # Bidirectional GRU layer
            self.fc = nn.Dense(enc_hid_dim * 2, dec_hid_dim).to_float(compute_dtype) # Fully-connected layer

        else:
            self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)  # Bidirectional GRU layer
            self.fc = nn.Dense(enc_hid_dim * 2, dec_hid_dim)  # Fully-connected layer

        self.dropout = nn.Dropout(p=1-dropout) # Dropout, preventing overfitting

    def construct(self, src, src_len):
        """Encoder Building

        Args:
            src: indicates the source sequence, which has been converted into a numeric index and has a unified length. shape = [src len, batch_size]
            src_len: indicates the valid length. shape = [batch_size, ]
        """

        # Convert the source sequence into a vector and perform dropout.
        # shape = [src len, batch size, emb dim]
        embedded = self.dropout(self.embedding(src))
        # Calculate the output.
        # shape = [src len, batch size, enc hid dim*2]
        outputs, hidden = self.rnn(embedded, seq_length=src_len)
        # Combine two context functions to adapt to the decoder.
        # shape = [batch size, dec hid dim]
        hidden = ops.tanh(self.fc(ops.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)))

        return outputs, hidden
```

### Attention Layer

In machine translation, each generated word may correspond to a different word in the source sentence, while the traditional Seq2Seq model with no attention mechanism prefers to focus on the last word in the sentence. To further optimize the model, we introduce the attention mechanism.

The attention mechanism is to give higher weight to corresponding words in the source and target sentence. It integrates all information encoded and decoded so far, and outputs a vector $a_t$ indicating the attention weight, which is used to determine which words should be given higher attention in the next prediction $\hat{y}_{t+}$.

![avatar4](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/application/source_zh_cn/nlp/images/seq2seq_4.png)

> Image source:
>
> <https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb>

First, we need to calculate the matching degree $E_t$ between each hidden state in the encoder and the hidden state at the previous time in the decoder.

Up to the current time $t$, all information in the encoder is a combination of hidden states of all forward and reverse RNNs $H$, and is a sequence with $T$ tensors; the information in the decoder is a hidden state $s_{t-1}$ at the previous time, and is an independent tensor. To unify their dimensions, we need to repeat the decoder hidden state $s_{t-1}$ at the previous time for $T$ times, stack up the processed decoder and encoder information, input the information to the linear layer `att` and the activation function $\text{tanh}$, and calculate the energy $E_t$ between the hidden state of encoder and decoder.

$$E_t = \tanh(\text{attn}(s_{t-1}, H))$$

For current $E_t$, the shape of the tensor in each batch is \[dec hid dim, src len\]. Note that the final attention weight needs to be applied to the source sequence. Therefore, the dimension of the attention weight should correspond to the dimension \[src len\] of the source sentence. To this end, we introduce a learnable tensor $v$.

$$\hat{a}_t = v E_t$$

We can think of $v$ as the weight of the weighted sum of all encoder hidden states, that is, the attention degree to each word in the source sequence. The parameter of $v$ is randomly initialized and learned with the rest of the model in backpropagation. Besides, $v$ does not depend on time, so $v$ used for each time step in decoding is consistent.

Finally, we use the $\text{softmax}$ function to ensure that the size of each element in the attention vector $a_t$ ranges from 0 to 1 and the sum of all elements is 1.

$$a_t = \text{softmax}(\hat{a_t})$$

```python
class Attention(nn.Cell):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        if is_ascend:
            # Attention linear layer
            self.attn = nn.Dense((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim).to_float(compute_dtype)
            # v, represented by a linear layer without bias
            # shape = [1, dec hid dim]
            self.v = nn.Dense(dec_hid_dim, 1, has_bias=False).to_float(compute_dtype)
        else:
            self.attn = nn.Dense((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
            self.v = nn.Dense(dec_hid_dim, 1, has_bias=False)

    def construct(self, hidden, encoder_outputs, mask):
        """Attention layer

        Args:
            hidden: indicates the hidden state of the decoder at the previous time. shape = [batch size, dec hid dim]
            encoder_outputs: indicates the encoder output, forward and reverse RNN hidden state. shape = [src len, batch size, enc hid dim * 2]
            mask: replaces the attention weight of the <pad> placeholder with 0 or a small value. shape = [batch size, src len]
        """

        src_len = encoder_outputs.shape[0]

        # Repeat the decoder hidden state *src len* times to unify dimensions.
        # shape = [batch size, src len, dec hid dim]
        hidden = ops.tile(hidden.expand_dims(1), (1, src_len, 1))

        # Exchange the first and second dimensions in the encoder output to unify dimensions.
        # shape = [batch size, src len, enc hid dim*2]
        encoder_outputs = encoder_outputs.transpose(1, 0, 2)

        # Calculate E_t.
        # shape = [batch size, src len, dec hid dim]
        energy = ops.tanh(self.attn(ops.concat((hidden, encoder_outputs), axis=2)))

        # Calculate v * E_t.
        # shape = [batch size, src len]
        attention = self.v(energy).squeeze(2)

        # The attention weight of the <pad> placeholder in the sequence does not need to be considered.
        attention = attention.masked_fill(mask == 0, -1e10)

        return ops.softmax(attention, axis=1)
```

### Decoder

The decoder contains the attention layer. We apply the obtained attention weight vector $a_t$ on the encoder hidden state $H$ to obtain a vector $w_t$ representing a weighted sum of the encoder hidden states.

$$w_t = a_t H$$

We pass the vector $w_t$, together with the embedded input word $d(y_t)$ and the previous decoder hidden state $s_{t-1}$, into the decoder RNN network, and send the output to the linear layer $f$ to obtain a word prediction at the next time in the target sentence.

$$s_t = \text{DecoderGRU}(d(y_t), w_t, s_{t-1})$$

$$\hat{y}_{t+1} = f(d(y_t), w_t, s_t)$$

![avatar5](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/application/source_zh_cn/nlp/images/seq2seq_5.png)

> Image source:
>
> <https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb>

```python
class Decoder(nn.Cell):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, is_ascend):
        super().__init__()
        self.is_ascend = is_ascend
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        if is_ascend:
            self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim).to_float(compute_dtype)
            self.fc_out = nn.Dense((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim).to_float(compute_dtype)
        else:
            self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
            self.fc_out = nn.Dense((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(p=1-dropout)

    def construct(self, inputs, hidden, encoder_outputs, mask):
        """Decoder Building

        Args:
            input: indicates the input words. shape = [batch size]
            hidden: indicates the hidden state of the decoder at the previous time. shape = [batch size, dec hid dim]
            encoder_outputs: indicates the encoder output, forward and reverse RNN hidden state. shape = [src len, batch size, enc hid dim * 2]
            mask: replaces the attention weight of the <pad> placeholder with 0 or a small value. shape = [batch size, src len]
        """

        # Add additional dimensions for the input.
        # shape = [1, batch size]
        inputs = inputs.expand_dims(0)

        # Embedding output of the input word, d (y_t)
        # shape = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(inputs))
        if self.is_ascend:
            embedded = embedded.astype(compute_dtype)

        # Attention weight vector, a_t
        # shape = [batch size, src len]
        a = self.attention(hidden, encoder_outputs, mask)

        # Add additional dimensions for the attention weight.
        # shape = [batch size, 1, src len]
        a = a.expand_dims(1)

        # Exchange the first and second dimensions in the encoder hidden state.
        # shape = [batch size, src len, enc hid dim * 2]
        encoder_outputs = encoder_outputs.transpose(1, 0, 2)

        # Calculate w_t.
        # shape = [batch size, 1, enc hid dim * 2]
        weighted = ops.bmm(a, encoder_outputs)

        # Exchange the first and second dimensions of w_t.
        # shape = [1, batch size, enc hid dim * 2]
        weighted = weighted.transpose(1, 0, 2)

        # Stack the embedded and weighted, and then pass them into the RNN layer.
        # rnn_input shape = [1, batch size, (enc hid dim * 2) + emb dim]
        # output shape = [seq len = 1, batch size, dec hid dim * n directions]
        # hidden shape = [n layers (1) * n directions (1) = 1, batch size, dec hid dim]
        rnn_input = ops.concat((embedded, weighted), axis=2)
        output, hidden = self.rnn(rnn_input, hidden.expand_dims(0))

        # Remove the redundancy of dimension 1.
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        # Stack the embedded, weighted, and hidden, and pass them into the linear layer to predict the next word.
        # shape = [batch size, output dim]
        prediction = self.fc_out(ops.concat((output, weighted, embedded), axis=1))

        return prediction, hidden.squeeze(0), a.squeeze(1)
```

### Seq2Seq

The Seq2Seq wrapper merges the encoder and decoder we created earlier.

The overall process is as follows:

1. An empty number column `outputs` is initialized to store every prediction result.
2. The source sequence $X$ is used as the encoder input to output $z$ and $H$.
3. The initial decoder hidden state is a context vector output in the encoder, that is, the last encoder hidden state $s_0 = z = h_T$.
4. The initial decoder input $y_1$ is a placeholder \<bos\> indicating the start of a sequence.
5. Repeat the following steps:
    - Set the input $y_t$ of $t$ at this time, the previous hidden state $s_{t-1}$, and all encoder hidden states $H$ as inputs.
    - Output a prediction $\hat{y}_{t+1}$ for the next time and a new hidden state $s_t$.
    - Save the prediction result to `outputs`.
    - Determine whether to use teacher forcing. If yes, use $y_{t+1} = \hat{y}_{t+1}$. If no, the word in the target sequence is used as the input at the next time.

```python
from mindspore import Tensor

class Seq2Seq(nn.Cell):
    def __init__(self, encoder, decoder, src_pad_idx, teacher_forcing_ratio):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.teacher_forcing_ratio = teacher_forcing_ratio  # The possibility of using teacher forcing

    def create_mask(self, src):
        """Mark the position of the <pad> placeholder in each sequence."""
        mask = (src != self.src_pad_idx).astype(mindspore.int32).swapaxes(1, 0)
        return mask

    def construct(self, src, src_len, trg, trg_len=None):
        """Seq2Seq model building

        Args:
            src: indicates the source sequence. shape = [src len, batch size]
            src_len: indicates the length of a source sequence. shape = [batch size]
            trg: indicates the target sequence. shape = [trg len, batch size]
            trg_len: indicates the length of a target sequence. shape = [trg len, batch size]
        """
        if trg_len is None:
            trg_len = trg.shape[0]

        #Storage decoder output
        outputs = []

        # Encoder:
        # Input: the source sequence and source sequence length
        # Output 1: encoder_outputs, indicating the hidden state of all forward and reverse RNNs in the encoder
        # Output 2: hidden, indicating the output after the last encoder hidden state in the forward and reverse RNNs is passed into the linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)

        #The first decoder input is the placeholder <bos>, indicating the start of a sequence.
        inputs = trg[0]

        # Mark the position of the <pad> placeholder in the source sequence.
        # shape = [batch size, src len]
        mask = self.create_mask(src)

        for t in range(1, trg_len):

            # Decoder:
            # Input: source sentence sequence (inputs), previous hidden state (hidden), and encoder hidden state of all forward and reverse RNNs
            # Mark <pad> in each sentence so that this part can be ignored when the attention weight is calculated.
            # Output: prediction result (output), new hidden status (hidden), and attention weight (ignored)
            output, hidden, _ = self.decoder(inputs, hidden, encoder_outputs, mask)

            # Save the prediction result into the previous storage.
            outputs.append(output)

            #Find the token with the maximum prediction probability.
            top1 = output.argmax(1).astype(mindspore.int32)

            if self.training:
                #If the model is in the training state, teacher forcing is used based on the preset probability.
                minval = Tensor(0, mindspore.float32)
                maxval = Tensor(1, mindspore.float32)
                teacher_force = ops.uniform((1,), minval, maxval) < self.teacher_forcing_ratio
                # If teacher forcing is used, the corresponding token in the target sequence is used as the next input.
                # If teacher forcing is not used, the prediction result is used as the next input.
                inputs = trg[t] if teacher_force else top1
            else:
                inputs = top1

        # Integrate all output as a tensor.
        outputs = ops.stack(outputs, axis=0)

        return outputs.astype(dtype)
```

## Model Training

Model parameters, encoder, attention layer, decoder, and Seq2Seq network initialization.

Here we manually implement mixed precision, i.e. we compute with `compute_dtype` (mindspore.float16) in the process and convert the result back to `dtype` (mindspore.float32) in the final output.

```python
input_dim = len(de_vocab) # Input dimension
output_dim = len(en_vocab) # Output dimension
enc_emb_dim = 256 # Encoder embedding layer dimension
dec_emb_dim = 256 # Decoder embedding layer dimension
enc_hid_dim = 512 # Encoder hidden layer dimension
dec_hid_dim = 512 # Decoder hidden layer dimension
enc_dropout = 0.5  # Encoder Dropout
dec_dropout = 0.5  # Decoder Dropout
src_pad_idx = de_vocab.pad_idx  # Numeric index of the pad placeholder in the German vocabulary
trg_pad_idx = en_vocab.pad_idx  # Numeric index of the pad placeholder in the English vocabulary

is_ascend = mindspore.get_context('device_target') == 'Ascend'
compute_dtype = mindspore.float32  # Data type in calculation
dtype = mindspore.float32 # Type of the returned data

attn = Attention(enc_hid_dim, dec_hid_dim, is_ascend)
encoder = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout, is_ascend)
decoder = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attn, is_ascend)

model = Seq2Seq(encoder, decoder, src_pad_idx, 0.5)
```

Initialize the loss function and optimizer.

```python
opt = nn.Adam(model.trainable_params(), learning_rate=0.001) # Loss function
loss_fn = nn.CrossEntropyLoss(ignore_index=trg_pad_idx) # Optimizer
```

Note that the updated weight may be too large during model training. This will cause value overflow or underflow, resulting in gradient explosion. To solve this problem, you need to use gradient clipping after calculating the gradient after backpropagation, and then transfer the clipped gradient to the optimizer for network update.

```python
import mindspore.ops as ops

def clip_by_norm(clip_norm, t, axis=None):
    """Regularize the tensor t based on the given tensor t and clipping parameter clip_norm.

    So that L2-norm of t in the axes dimension is less than or equal to the value of clip_norm.

    Args:
        t: tensor of type float
        clip_norm: scalar of type float. It is the gradient clipping threshold. The value must be greater than 0.
        axis: Union[None, int, tuple(int)]. The data type is int32. Calculate the L2-norm dimension, if the result is Norm, all dimensions are referenced.
    """

    # Calculate L2-norm.
    t2 = t * t
    l2sum = t2.sum(axis=axis, keepdims=True)
    pred = l2sum > 0
    # Replace the element whose value is 0 in the sum with 1 to avoid NaN.
    l2sum_safe = ops.select(pred, l2sum, ops.ones_like(l2sum))
    l2norm = ops.select(pred, ops.sqrt(l2sum_safe), l2sum)
    # Compare the value of L2-norm with clip_norm. If the value of L2-norm exceeds the threshold, perform the clipping.
    # Clipping method: output(x) = (x * clip_norm)/max(|x|, clip_norm)
    intermediate = t * clip_norm
    cond = l2norm > clip_norm
    t_clip = intermediate / ops.select(cond, l2norm, clip_norm)

    return t_clip

```

During model training, use the validation dataset for validation and evaluation, and save the model with the best effect.

```python
def forward_fn(src, src_len, trg):
    """Forward network""
    src = src.swapaxes(0, 1)
    trg = trg.swapaxes(0, 1)

    output = model(src, src_len, trg)
    output_dim = output.shape[-1]
    output = output.view(-1, output_dim)
    trg = trg[1:].view(-1)
    loss = loss_fn(output, trg)

    return loss


# Backpropagation calculation gradient
grad_fn = mindspore.value_and_grad(forward_fn, None, opt.parameters)

def train_step(src, src_len, trg, clip):
    """Single-step training"""
    loss, grads = grad_fn(src, src_len, trg)
    grads = ops.HyperMap()(ops.partial(clip_by_norm, clip), grads) # Clipping gradient.
    opt(grads) #Update network parameters.

    return loss


def train(iterator, clip, epoch=0):
    """Model training""
    model.set_train(True)
    num_batches = len(iterator)
    total_loss = 0 # The training loss of all batches
    total_steps = 0 # Number of training steps

    with tqdm(total=num_batches) as t:
        t.set_description(f'Epoch: {epoch}')
        for src, src_len, trg in iterator():
            loss = train_step(src, src_len, trg, clip)  # Loss of the current batch
            total_loss += loss.asnumpy()
            total_steps += 1
            curr_loss = total_loss / total_steps  # Average loss of current batch
            t.set_postfix({'loss': f'{curr_loss:.2f}'})
            t.update(1)

    return total_loss / total_steps


def evaluate(iterator):
    """Model validation""
    model.set_train(False)
    num_batches = len(iterator)
    total_loss = 0 # The training loss of all batches
    total_steps = 0 # Number of training steps

    with tqdm(total=num_batches) as t:
        for src, src_len, trg in iterator():
            loss = forward_fn(src, src_len, trg)  # Loss of the current batch
            total_loss += loss.asnumpy()
            total_steps += 1
            curr_loss = total_loss / total_steps  # Average loss of current batch
            t.set_postfix({'loss': f'{curr_loss:.2f}'})
            t.update(1)

    return total_loss / total_steps
```

```python
from mindspore import save_checkpoint

num_epochs = 10 # Number of training epochs
clip = 1.0 # Gradient clipping threshold
best_valid_loss = float('inf') # Current best validation loss
ckpt_file_name = os.path.join(cache_dir, 'seq2seq.ckpt') # Model save path

for i in range(num_epochs):
    # Train the model and update the network weight.
    train_loss = train(train_iterator, clip, i)
    # Validate the model after the network weight is updated.
    valid_loss = evaluate(valid_iterator)

    # Save the model with the best effect.
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        save_checkpoint(model, ckpt_file_name)
```

Output:

```text
Epoch: 0: 100%|██████████| 226/226 [04:17<00:00,  1.14s/it, loss=4.90]
100%|██████████| 8/8 [00:06<00:00,  1.24it/s, loss=4.74]
Epoch: 1: 100%|██████████| 226/226 [02:45<00:00,  1.37it/s, loss=3.88]
100%|██████████| 8/8 [00:01<00:00,  4.60it/s, loss=3.98]
Epoch: 2: 100%|██████████| 226/226 [02:46<00:00,  1.36it/s, loss=3.19]
100%|██████████| 8/8 [00:01<00:00,  4.54it/s, loss=3.63]
Epoch: 3: 100%|██████████| 226/226 [02:47<00:00,  1.35it/s, loss=2.73]
100%|██████████| 8/8 [00:01<00:00,  4.49it/s, loss=3.46]
Epoch: 4: 100%|██████████| 226/226 [02:48<00:00,  1.34it/s, loss=2.40]
100%|██████████| 8/8 [00:01<00:00,  4.56it/s, loss=3.38]
Epoch: 5: 100%|██████████| 226/226 [02:47<00:00,  1.35it/s, loss=2.12]
100%|██████████| 8/8 [00:01<00:00,  4.50it/s, loss=3.37]
Epoch: 6: 100%|██████████| 226/226 [02:45<00:00,  1.37it/s, loss=1.91]
100%|██████████| 8/8 [00:01<00:00,  4.55it/s, loss=3.40]
Epoch: 7: 100%|██████████| 226/226 [02:45<00:00,  1.36it/s, loss=1.74]
100%|██████████| 8/8 [00:01<00:00,  4.60it/s, loss=3.44]
Epoch: 8: 100%|██████████| 226/226 [02:45<00:00,  1.37it/s, loss=1.59]
100%|██████████| 8/8 [00:01<00:00,  4.54it/s, loss=3.44]
Epoch: 9: 100%|██████████| 226/226 [02:44<00:00,  1.37it/s, loss=1.47]
100%|██████████| 8/8 [00:01<00:00,  4.57it/s, loss=3.50]
```

## Model Inference

```python
def translate_sentence(sentence, de_vocab, en_vocab, model, max_len=32):
    """Give a German sentence and return English translation."""
    model.set_train(False)
    # Segment the input sentences.
    if isinstance(sentence, str):
        tokens = [tok.lower() for tok in re.findall(r'\w+|[^\w\s]', sentence.rstrip())]
    else:
        tokens = [token.lower() for token in sentence]

    # Add the start and end placeholders to unify the sequence length.
    if len(tokens) > max_len - 2:
        src_len = max_len
        tokens = ['<bos>'] + tokens[:max_len - 2] + ['<eos>']
    else:
        src_len = len(tokens) + 2
        tokens = ['<bos>'] + tokens + ['<eos>'] + ['<pad>'] * (max_len - src_len)

    # Convert German words into numeric indexes.
    src = de_vocab.encode(tokens)
    src = mindspore.Tensor(src, mindspore.int32).expand_dims(1)
    src_len = mindspore.Tensor([src_len], mindspore.int32)
    trg = mindspore.Tensor([en_vocab.bos_idx], mindspore.int32).expand_dims(1)

    # Obtain the prediction result and convert it into English words.
    outputs = model(src, src_len, trg, max_len)
    trg_indexes = [int(i.argmax(1).asnumpy()) for i in outputs]
    eos_idx = trg_indexes.index(en_vocab.eos_idx) if en_vocab.eos_idx in trg_indexes else -1
    trg_tokens = en_vocab.decode(trg_indexes[:eos_idx])

    return trg_tokens
```

Use any set of text in the test dataset for prediction.

```python
from mindspore import load_checkpoint, load_param_into_net

# Load the trained model.
param_dict = load_checkpoint(ckpt_file_name)
load_param_into_net(model, param_dict)

# Take the first group of sentences in the test dataset as an example.
example_idx = 0

src = test_dataset[example_idx][0]
trg = test_dataset[example_idx][1]

print(f'src = {src}')
print(f'trg = {trg}')
```

Output:

```text
src = ['ein', 'mann', 'mit', 'einem', 'orangefarbenen', 'hut', ',', 'der', 'etwas', 'anstarrt', '.']
trg = ['a', 'man', 'in', 'an', 'orange', 'hat', 'starring', 'at', 'something', '.']
```

View the prediction results.

```python
translation = translate_sentence(src, de_vocab, en_vocab, model)

print(f'predicted trg = {translation}')
```

Output:

```text
predicted trg = ['a', 'man', 'in', 'an', 'orange', 'hat', ',', 'something', '.']
```

## BLEU Score

The bilingual evaluation understudy (BLEU) is an algorithm for measuring the quality of sentences generated by the text translation model. It focuses on evaluating the similarity between the translation $\text{pred}$ by machine and the reference translation $\text{label}$ by humans. The score of each segment is calculated by comparing the segments of the machine translation with the reference translation, and the score is summed up with the weight. The basic rule is as follows:

1. Punish predictions that are too short. That is, if a machine translation is excessively short compared to a reference translation, it will be imposed penalty with high hit rate.
2. Configure high weights for long paragraphs. That is, if a complete hit of a long paragraph occurs, it indicates that the machine translation is close to the reference translation.

The BLEU formula is as follows:

$$exp(min(0, 1-\frac{len(\text{label})}{len(\text{pred})})\Pi^k_{n=1}p_n^{1/2^n})$$

- `len(label)`: length of the translation by humans
- `len(pred)`: length of the translation by machine
- `p_n`: n-gram precision

```python
from nltk.translate.bleu_score import corpus_bleu

def calculate_bleu(dataset, de_vocab, en_vocab, model, max_len=50):
    trgs = []
    pred_trgs = []

    for data in dataset:

        src = data[0] #Source sentences: German
        trg = data[1] #Target sentences: English

        # Obtain the model prediction result.
        pred_trg = translate_sentence(src, de_vocab, en_vocab, model, max_len)
        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return corpus_bleu(trgs, pred_trgs)

# Calculate the BLEU score.
bleu_score = calculate_bleu(test_dataset, de_vocab, en_vocab, model)

print(f'BLEU score = {bleu_score*100:.2f}')
```

Output:

```text
BLEU score = 31.54
```

## Reference

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2014. Neural machine translation by jointly learning
to align and translate. arXiv preprint arXiv:1409.0473.
