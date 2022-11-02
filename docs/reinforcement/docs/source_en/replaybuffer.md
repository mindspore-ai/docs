# ReplayBuffer Usage Introduction

<a href="https://gitee.com/mindspore/docs/blob/master/docs/reinforcement/docs/source_en/replaybuffer.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Brief Introduction of ReplayBuffer

In reinforcement learning, ReplayBuffer is a common basic data storage method, whose functions is to store the data obtained from the interaction of an intelligent body with its environment.

Solve the following problems by using ReplayBuffer:

1. Stored historical data can be extracted by sampling to break the correlation of training data, so that the sampled data have independent and identically distributed characteristics.
2. Provide temporary storage of data and improve the utilization of data.

## ReplayBuffer Implementation of MindSpore Reinforcement Learning

Typically, algorithms people use native Python data structures or Numpy data structures to construct ReplayBuffer, or general reinforcement learning frameworks also provide standard API encapsulation. The difference is that MindSpore implements the ReplayBuffer structure on the device side. On the one hand, the structure can reduce the frequent copying of data between Host and Device when using GPU hardware, and on the other hand, expressing the ReplayBuffer in the form of MindSpore operator can build a complete IR graph and enable various graph optimizations of MindSpore GRAPH_MODE to improve the overall performance.

In MindSpore, two kinds of ReplayBuffer are provided, UniformReplayBuffer and PriorityReplayBuffer, which are used for common FIFO storage and storage with priority, respectively. The following is an example of UniformReplayBuffer implementation and usage.

ReplayBuffer is represented as a List of Tensors, and each Tensor represents a set of data stored by column (e.g., a set of [state, action, reward]). The data that is newly put into the UniformReplayBuffer is updated in a FIFO mechanism with insert, search, and sample functions.

### Parameter Explanation

Create a UniformReplayBuffer with the initialization parameters batch_size, capacity, shapes, and types.

* batch_size indicates the size of the data at a time for sample, an integer value.
* capacity indicates the total capacity of the created UniformReplayBuffer, an integer value.
* shapes indicates the shape size of each set of data in Buffer, expressed as a list.
* types indicates the data type corresponding to each set of data in the Buffer, represented as a list.

### Functions Introduction

#### 1 Insert

The insert method takes a set of data as input, and needs to satisfy that the shape and type of the data are the same as the created UniformReplayBuffer parameters. No output.
To simulate the FIFO characteristics of a circular queue, we use two cursors to determine the head and effective length count of the queue. The following figure shows the process of several insertion operations.

1. The total size of the buffer is 6. In the initial state, the cursor head and count are both 0.
2. After inserting a batch_size of 2, the current head is unchanged and count is added by 2.
3. After continuing to insert a batch_size of 4, the queue is full and the count is 6.
4. After continuing to insert a batch_size of 2, overwrite updates the old data and adds 2 to the head.

#### 2 Search

The search method accepts an index as an input, indicating the specific location of the data to be found. The output is a set of Tensor, as shown in the following figure:

1. If the UniformReplayBuffer is just full or not full, the corresponding data is found directly according to the index.
2. For data that has been overwritten, remap it by cursors.

![get_item schematic diagram](https://gitee.com/mindspore/docs/blob/master/docs/reinforcement/docs/source_zh_cn/images/get.png)

#### 3 Sample

The sampling method has no input and the output is a set of Tensor with the size of the batch_size when the UniformReplayBuffer is created. This is shown in the following figure:
Assuming that batch_size is 3, a random set of indexes will be generated in the operator, and this random set of indexes has two cases:

1. Packet ordering: each index means the real data position, which needs to be remapped by cursor operation.
2. No packet ordering: each index does not represent the real position and is obtained directly.

Both approaches have a slight impact on randomness, and the default is to use no-packet ordering to get the best performance.

## UniformReplayBuffer Introduction of MindSpore Reinforcement Learning

### Creation of UniformReplayBuffer

MindSpore Reinforcement Learning provides a standard ReplayBuffer API. The user can use the ReplayBuffer created by the framework by means of a configuration file, shaped like the configuration file of [dqn](https://gitee.com/mindspore/reinforcement/blob/master/mindspore_rl/algorithm/dqn/config.py).

```python
'replay_buffer':
    {'number': 1,
     'type': UniformReplayBuffer,
     'capacity': 100000,
     'data_shape': [(4,), (1,), (1,), (4,)],
     'data_type': [ms.float32, ms.int32, ms.foat32, ms.float32],
     'sample_size': 64}
```

Alternatively, users can use the interfaces directly to create the required data structures:

```python
from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
import mindspore as ms
sample_size = 2
capacity = 100000
shapes = [(4,), (1,), (1,), (4,)]
types = [ms.float32, ms.int32, ms.float32, ms.float32]
replaybuffer = UniformReplayBuffer(sample_size, capacity, shapes, types)
```

### Using the Created UniformReplayBuffer

Take [UniformReplayBuffer](https://gitee.com/mindspore/reinforcement/blob/master/mindspore_rl/core/uniform_replay_buffer.py) created in the form of an API to perform data manipulation as an example:

* Insert operation

```python
state = ms.Tensor([0.1, 0.2, 0.3, 0.4], ms.float32)
action = ms.Tensor([1], ms.int32)
reward = ms.Tensor([1], ms.float32)
new_state = ms.Tensor([0.4, 0.3, 0.2, 0.1], ms.float32)
replaybuffer.insert([state, action, reward, new_state])
replaybuffer.insert([state, action, reward, new_state])
```

* Search operation

```python
exp = replaybuffer.get_item(0)
```

* Sample operation

```python
samples = replaybuffer.sample()
```

* Reset operation

```python
replaybuffer.reset()
```

* The size of the current buffer used

```python
size = replaybuffer.size()
```

* Determine if the current buffer is full

```python
if replaybuffer.full():
    print("Full use of this buffer.")
```
