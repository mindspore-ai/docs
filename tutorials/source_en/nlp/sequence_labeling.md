[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/nlp/sequence_labeling.md)

# LSTM+CRF Sequence Labeling

> This case does not support running on the Windows operating system.

## Overview

Sequence labeling refers to the process of labeling each token for a given input sequence. Sequence labeling is usually used to extract information from text, including word segmentation, part-of-speech tagging, and named entity recognition (NER). The following uses NER as an example:

| Input Sequence| the | wall | street | journal | reported | today | that | apple | corporation | made | money |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|Output Labeling| B | I | I | I | O | O | O | B | I | O | O |

As shown in the preceding table, `the wall street journal` and `apple corporation` are place names and need to be identified. We predict the label of each input word and identify the entity based on the label.
> A common labeling method for NER is used, that is, BIOE labeling. The beginning of an entity is labeled as B, other parts are labeled as I, and non-entity is labeled as O.

## Conditional Random Field (CRF)

It can be learned from the preceding example that labeling a sequence is actually performing label prediction on each token in the sequence, and may be directly considered as a simple multi-classification problem. However, sequence labeling not only needs to classify and predict a single token, but also directly associates adjacent tokens. The `the wall street journal` is used as an example.

| Input Sequence| the | wall | street | journal | |
| --- | --- | --- | --- | --- | --- |
| Output Labeling| B | I | I | I | √ |
| Output Labeling| O | I | I | I | × |

As shown in the preceding table, the four tokens contained in the correct entity depend on each other. A word before I must be B or I. However, in the error output, the token `the` is marked as O, which violates the dependency. If NER is regarded as a multi-classification problem, the prediction probability of each word is independent and similar problems may occur. Therefore, an algorithm that can learn the association relationship is introduced to ensure the correctness of the prediction result. CRF is a [probabilistic graphical model](https://en.wikipedia.org/wiki/Graphical_model) suitable for this scenario. The definition and parametric form of conditional random field are briefly analyzed in the following.

> Considering the linear sequence feature of the sequence labeling problem, the CRF described in this section refers to the linear chain CRF.

Assume that $x=\{x_0, ..., x_n\}$ indicates the input sequence, $y=\{y_0, ..., y_n\}, y \in Y$ indicates the output labeling sequence, where $n$ indicates the maximum length of the sequence, and $Y$ indicates the set of all possible output sequences corresponding to $x$. The probability of the output sequence $y$ is as follows:

$$\begin{align}P(y|x) = \frac{\exp{(\text{Score}(x, y)})}{\sum_{y' \in Y} \exp{(\text{Score}(x, y')})} \qquad (1)\end{align}$$

If $x_i$ and $y_i$ are the $i$th token and the corresponding label in the sequence, $\text{Score}$ must calculate the mapping between $x_i$ and $y_i$ and capture the relationship between adjacent labels $y_{i-1}$ and $y_{i}$. Therefore, two probability functions are defined:

1. The emission probability function $\psi_\text{EMIT}$ indicates the probability of $x_i \rightarrow y_i$.
2. The transition probability function $\psi_\text{TRANS}$ indicates the probability of $y_{i-1} \rightarrow y_i$.

The formula for calculating $\text{Score}$ is as follows:

$$\begin{align}\text{Score}(x,y) = \sum_i \log \psi_\text{EMIT}(x_i \rightarrow y_i) + \log \psi_\text{TRANS}(y_{i-1} \rightarrow y_i) \qquad (2)\end{align} $$

Assume that the label set is $T$. Build a matrix $\textbf{P}$ with a size of $|T|x|T|$ to store the transition probability between labels. A hidden state $h$ output by the encoding layer (which may be Dense, LSTM, or the like) may be directly considered as an emission probability. In this case, the formula for calculating $\text{Score}$ can be converted into the following:

$$\begin{align}\text{Score}(x,y) = \sum_i h_i[y_i] + \textbf{P}_{y_{i-1}, y_{i}} \qquad (3)\end{align}$$

> For details about the complete CRF-based deduction, see [Log-Linear Models, MEMMs, and CRFs](http://www.cs.columbia.edu/~mcollins/crf.pdf).

Next, we use MindSpore to implement the CRF parameterization based on the preceding formula. First, a forward training part of a CRF layer is implemented, the CRF and a loss function are combined, and a negative log likelihood (NLL) function commonly used for a classification problem is selected.

$$\begin{align}\text{Loss} = -log(P(y|x)) \qquad (4)\end{align} $$

According to the formula $(1)$,

$$\begin{align}\text{Loss} = -log(\frac{\exp{(\text{Score}(x, y)})}{\sum_{y' \in Y} \exp{(\text{Score}(x, y')})}) \qquad (5)\end{align} $$

$$\begin{align}= log(\sum_{y' \in Y} \exp{(\text{Score}(x, y')}) - \text{Score}(x, y) \end{align}$$

According to the formula $(5)$, the minuend is called Normalizer, and the subtrahend is called Score. The final loss is obtained after the subtraction.

### Score Calculation

First, the score corresponding to the correct label sequence is calculated according to the formula $(3)$. It should be noted that, in addition to the transition probability matrix $\textbf{P}$, two vectors whose sizes are $|T|$ need to be maintained, and are respectively used as transition probabilities at the beginning and the end of the sequence. In addition, a mask matrix $mask$ is introduced. When multiple sequences are packed into a batch, the filled values are ignored. In this way, the $\text{Score}$ calculation contains only valid tokens.

```python
def compute_score(emissions, tags, seq_ends, mask, trans, start_trans, end_trans):
    # emissions: (seq_length, batch_size, num_tags)
    # tags: (seq_length, batch_size)
    # mask: (seq_length, batch_size)

    seq_length, batch_size = tags.shape
    mask = mask.astype(emissions.dtype)

    # Set score to the initial transition probability.
    # shape: (batch_size,)
    score = start_trans[tags[0]]
    # score += Probability of the first emission
    # shape: (batch_size,)
    score += emissions[0, mnp.arange(batch_size), tags[0]]

    for i in range(1, seq_length):
        # Probability that the label is transited from i-1 to i (valid when mask == 1).
        # shape: (batch_size,)
        score += trans[tags[i - 1], tags[i]] * mask[i]

        # Emission probability of tags[i] prediction(valid when mask == 1).
        # shape: (batch_size,)
        score += emissions[i, mnp.arange(batch_size), tags[i]] * mask[i]

    # End the transition.
    # shape: (batch_size,)
    last_tags = tags[seq_ends, mnp.arange(batch_size)]
    # score += End transition probability
    # shape: (batch_size,)
    score += end_trans[last_tags]

    return score
```

### Normalizer Calculation

According to the formula $(5)$, Normalizer is the Log-Sum-Exp of scores of all possible output sequences corresponding to $x$. In this case, if the enumeration method is used for calculation, each possible output sequence score needs to be calculated, and there are $|T|^{n}$ results in total. Here, we use the dynamic programming algorithm to improve the efficiency by reusing the calculation result.

Assume that you need to calculate the scores $\text{Score}_{i}$ of all possible output sequences from token $0$ to token $i$. In this case, scores $\text{Score}_{i-1}$ of all possible output sequences from the $0$th token to the $i-1$th token may be calculated first. Therefore, the Normalizer can be rewritten as follows:

$$log(\sum_{y'_{0,i} \in Y} \exp{(\text{Score}_i})) = log(\sum_{y'_{0,i-1} \in Y} \exp{(\text{Score}_{i-1} + h_{i} + \textbf{P}})) \qquad (6)$$

$h_i$ is the emission probability of the $i$th token, and $\textbf{P}$ is the transition matrix. Because the emission probability matrix $h$ and the transition probability matrix $\textbf{P}$ are independent of the sequence path calculation of $y$, we can obtain that:

$$log(\sum_{y'_{0,i} \in Y} \exp{(\text{Score}_i})) = log(\sum_{y'_{0,i-1} \in Y} \exp{(\text{Score}_{i-1}})) + h_{i} + \textbf{P} \qquad (7)$$

According to formula (7), the Normalizer is implemented as follows:

```python
def compute_normalizer(emissions, mask, trans, start_trans, end_trans):
    # emissions: (seq_length, batch_size, num_tags)
    # mask: (seq_length, batch_size)

    seq_length = emissions.shape[0]

    # Set score to the initial transition probability and add the first emission probability.
    # shape: (batch_size, num_tags)
    score = start_trans + emissions[0]

    for i in range(1, seq_length):
        # The score dimension is extended to calculate the total score.
        # shape: (batch_size, num_tags, 1)
        broadcast_score = score.expand_dims(2)

        # The emission dimension is extended to calculate the total score.
        # shape: (batch_size, 1, num_tags)
        broadcast_emissions = emissions[i].expand_dims(1)

        # Calculate score_i according to formula (7).
        # In this case, broadcast_score indicates all possible paths from token 0 to the current token.
        # log_sum_exp corresponding to score
        # shape: (batch_size, num_tags, num_tags)
        next_score = broadcast_score + trans + broadcast_emissions

        # Perform the log_sum_exp operation on score_i to calculate the score of the next token.
        # shape: (batch_size, num_tags)
        next_score = ops.logsumexp(next_score, dim=1)

        # The score changes only when mask == 1.
        # shape: (batch_size, num_tags)
        score = mnp.where(mask[i].expand_dims(1), next_score, score)

    # Add the end transition probability.
    # shape: (batch_size, num_tags)
    score += end_trans
    # Calculate log_sum_exp based on the scores of all possible paths.
    # shape: (batch_size,)
    return ops.logsumexp(score, dim=1)
```

### Viterbi Algorithm

After the forward training part is completed, the decoding part needs to be implemented. Here we select the [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm) that is suitable for finding the optimal path of the sequence. Similar to calculating Normalizer, dynamic programming is used to solve all possible prediction sequence scores. The difference is that the label with the maximum score corresponding to token $i$ needs to be saved during decoding. The label is used by the Viterbi algorithm to calculate the optimal prediction sequence.

After obtaining the maximum probability score $\text{Score}$ and the label history $\text{History}$ corresponding to each token, use the Viterbi algorithm to calculate the following formula:

$$P_{0,i} = max(P_{0, i-1}) + P_{i-1, i}$$

The 0th token to the $i$th token correspond to sequences with a maximum probability. Only sequences with a maximum probability corresponding to the 0th token to the $i-1$th token and labels with a maximum probability corresponding to the $i$th token to the $i-1$th token need to be considered. Therefore, we solve each label with the highest probability in reverse order to form the optimal prediction sequence.

> Due to the syntax restrictions of static graphs, the Viterbi algorithm is used to solve the optimal prediction sequence as a post-processing function and is not included in the implementation of the CRF layer.

```python
def viterbi_decode(emissions, mask, trans, start_trans, end_trans):
    # emissions: (seq_length, batch_size, num_tags)
    # mask: (seq_length, batch_size)

    seq_length = mask.shape[0]

    score = start_trans + emissions[0]
    history = ()

    for i in range(1, seq_length):
        broadcast_score = score.expand_dims(2)
        broadcast_emission = emissions[i].expand_dims(1)
        next_score = broadcast_score + trans + broadcast_emission

        # Obtain the label with the maximum score corresponding to the current token and save the label.
        indices = next_score.argmax(axis=1)
        history += (indices,)

        next_score = next_score.max(axis=1)
        score = mnp.where(mask[i].expand_dims(1), next_score, score)

    score += end_trans

    return score, history

def post_decode(score, history, seq_length):
    # Use Score and History to calculate the optimal prediction sequence.
    batch_size = seq_length.shape[0]
    seq_ends = seq_length - 1
    # shape: (batch_size,)
    best_tags_list = []

    # Decode each sample in a batch in sequence.
    for idx in range(batch_size):
        # Search for the label that maximizes the prediction probability corresponding to the last token.
        # Add it to the list of best prediction sequence stores.
        best_last_tag = score[idx].argmax(axis=0)
        best_tags = [int(best_last_tag.asnumpy())]

        # Repeatedly search for the label with the maximum prediction probability corresponding to each token and add the label to the list.
        for hist in reversed(history[:seq_ends[idx]]):
            best_last_tag = hist[idx][best_tags[-1]]
            best_tags.append(int(best_last_tag.asnumpy()))

        # Reset the solved label sequence in reverse order to the positive sequence.
        best_tags.reverse()
        best_tags_list.append(best_tags)

    return best_tags_list
```

### CRF Layer

After the code of the forward training part and the code of the decoding part are completed, a complete CRF layer is assembled. Considering that the input sequence may be padded, the actual length of the input sequence needs to be considered during CRF input. Therefore, in addition to the emissions matrix and label, the `seq_length` parameter is added to transfer the length of the sequence before padding and implement the `sequence_mask` method for generating the mask matrix.

Based on the preceding code, `nn.Cell` is used for encapsulation. The complete CRF layer is implemented as follows:

```python
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore.common.initializer import initializer, Uniform

def sequence_mask(seq_length, max_length, batch_first=False):
    """Generate the mask matrix based on the actual length and maximum length of the sequence."""
    range_vector = mnp.arange(0, max_length, 1, seq_length.dtype)
    result = range_vector < seq_length.view(seq_length.shape + (1,))
    if batch_first:
        return result.astype(ms.int64)
    return result.astype(ms.int64).swapaxes(0, 1)

class CRF(nn.Cell):
    def __init__(self, num_tags: int, batch_first: bool = False, reduction: str = 'sum') -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.reduction = reduction
        self.start_transitions = ms.Parameter(initializer(Uniform(0.1), (num_tags,)), name='start_transitions')
        self.end_transitions = ms.Parameter(initializer(Uniform(0.1), (num_tags,)), name='end_transitions')
        self.transitions = ms.Parameter(initializer(Uniform(0.1), (num_tags, num_tags)), name='transitions')

    def construct(self, emissions, tags=None, seq_length=None):
        if tags is None:
            return self._decode(emissions, seq_length)
        return self._forward(emissions, tags, seq_length)

    def _forward(self, emissions, tags=None, seq_length=None):
        if self.batch_first:
            batch_size, max_length = tags.shape
            emissions = emissions.swapaxes(0, 1)
            tags = tags.swapaxes(0, 1)
        else:
            max_length, batch_size = tags.shape

        if seq_length is None:
            seq_length = mnp.full((batch_size,), max_length, ms.int64)

        mask = sequence_mask(seq_length, max_length)

        # shape: (batch_size,)
        numerator = compute_score(emissions, tags, seq_length-1, mask, self.transitions, self.start_transitions, self.end_transitions)
        # shape: (batch_size,)
        denominator = compute_normalizer(emissions, mask, self.transitions, self.start_transitions, self.end_transitions)
        # shape: (batch_size,)
        llh = denominator - numerator

        if self.reduction == 'none':
            return llh
        if self.reduction == 'sum':
            return llh.sum()
        if self.reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.astype(emissions.dtype).sum()

    def _decode(self, emissions, seq_length=None):
        if self.batch_first:
            batch_size, max_length = emissions.shape[:2]
            emissions = emissions.swapaxes(0, 1)
        else:
            batch_size, max_length = emissions.shape[:2]

        if seq_length is None:
            seq_length = mnp.full((batch_size,), max_length, ms.int64)

        mask = sequence_mask(seq_length, max_length)

        return viterbi_decode(emissions, mask, self.transitions, self.start_transitions, self.end_transitions)
```

## BiLSTM+CRF Model

After CRF is implemented, a bidirectional LSTM+CRF model is designed to train NER tasks. The model structure is as follows:

```text
nn.Embedding -> nn.LSTM -> nn.Dense -> CRF
```

The LSTM extracts a sequence feature, obtains an emission probability matrix by means of Dense layer transformation, and finally sends the emission probability matrix to the CRF layer. The sample code is as follows:

```python
class BiLSTM_CRF(nn.Cell):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Dense(hidden_dim, num_tags, 'he_uniform')
        self.crf = CRF(num_tags, batch_first=True)

    def construct(self, inputs, seq_length, tags=None):
        embeds = self.embedding(inputs)
        outputs, _ = self.lstm(embeds, seq_length=seq_length)
        feats = self.hidden2tag(outputs)

        crf_outs = self.crf(feats, tags, seq_length)
        return crf_outs
```

After the model design is complete, two examples and corresponding labels are generated, and a vocabulary and a label table are built.

```python
embedding_dim = 16
hidden_dim = 32

training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_idx = {}
word_to_idx['<pad>'] = 0
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

tag_to_idx = {"B": 0, "I": 1, "O": 2}
```

```python
len(word_to_idx)
```

```text
21
```

Instantiate the model, select an optimizer, and send the model and optimizer to the Wrapper.

> The NLLLoss has been calculated at the CRF layer. Therefore, you do not need to set Loss.

```python
model = BiLSTM_CRF(len(word_to_idx), embedding_dim, hidden_dim, len(tag_to_idx))
optimizer = nn.SGD(model.trainable_params(), learning_rate=0.01, weight_decay=1e-4)
```

```python
grad_fn = ms.value_and_grad(model, None, optimizer.parameters)

def train_step(data, seq_length, label):
    loss, grads = grad_fn(data, seq_length, label)
    optimizer(grads)
    return loss
```

Pack the generated data into a batch, pad the sequence with insufficient length based on the maximum sequence length, and return tensors consisting of the input sequence, output label, and sequence length.

```python
def prepare_sequence(seqs, word_to_idx, tag_to_idx):
    seq_outputs, label_outputs, seq_length = [], [], []
    max_len = max([len(i[0]) for i in seqs])

    for seq, tag in seqs:
        seq_length.append(len(seq))
        idxs = [word_to_idx[w] for w in seq]
        labels = [tag_to_idx[t] for t in tag]
        idxs.extend([word_to_idx['<pad>'] for i in range(max_len - len(seq))])
        labels.extend([tag_to_idx['O'] for i in range(max_len - len(seq))])
        seq_outputs.append(idxs)
        label_outputs.append(labels)

    return ms.Tensor(seq_outputs, ms.int64), \
            ms.Tensor(label_outputs, ms.int64), \
            ms.Tensor(seq_length, ms.int64)
```

```python
data, label, seq_length = prepare_sequence(training_data, word_to_idx, tag_to_idx)
data.shape, label.shape, seq_length.shape
```

```text
((2, 11), (2, 11), (2,))
```

After the model is precompiled, 500 steps are trained.

> Training process visualization depends on the `tqdm` library, which can be installed by running the `pip install tqdm` command.

```python
from tqdm import tqdm

steps = 500
with tqdm(total=steps) as t:
    for i in range(steps):
        loss = train_step(data, seq_length, label)
        t.set_postfix(loss=loss)
        t.update(1)
```

```text
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:23<00:00, 21.13it/s, loss=0.3487625]
```

Finally, let's observe the model effect after 500 steps of training. First, use the model to predict possible path scores and candidate sequences.

```python
score, history = model(data, seq_length)
score
```

```text
Tensor(shape=[2, 3], dtype=Float32, value=
[[ 3.15928860e+01,  3.63119812e+01,  3.17248516e+01],
 [ 2.81416149e+01,  2.61749763e+01,  3.24760780e+01]])
```

Perform post-processing on the predicted score.

```python
predict = post_decode(score, history, seq_length)
predict
```

```text
[[0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2], [0, 1, 2, 2, 2, 2, 0, 2, 2]]
```

Finally, convert the predicted index sequence into a label sequence, print the output result, and view the effect.

```python
idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}

def sequence_to_tag(sequences, idx_to_tag):
    outputs = []
    for seq in sequences:
        outputs.append([idx_to_tag[i] for i in seq])
    return outputs
```

```python
sequence_to_tag(predict, idx_to_tag)
```

```text
[['B', 'I', 'I', 'I', 'O', 'O', 'O', 'B', 'I', 'O', 'O'],
 ['B', 'I', 'O', 'O', 'O', 'O', 'B', 'O', 'O']]
```
