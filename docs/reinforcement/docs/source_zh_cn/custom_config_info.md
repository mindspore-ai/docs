# 强化学习配置说明

<a href="https://gitee.com/mindspore/docs/blob/master/docs/reinforcement/docs/source_zh_cn/custom_config_info.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>
&nbsp;&nbsp;

## 概述

深度强化学习作为当前发展最快的方向之一，新算法层出不穷。MindSpore Reinforcement将强化学习算法建模为Actor、Learner、Policy、Environment、ReplayBuffer等对象，从而提供易扩展、高重用的强化学习框架。与此同时，深度强化学习算法相对复杂，网络训练效果受到众多参数影响，MindSpore Reinforcement提供了集中的参数配置接口，将算法实现和部署细节进行分离，同时便于用户快速调整模型算法。

本文以DQN算法为例介绍如何使用MindSpore Reinforcement算法和训练参数配置接口，帮助用户快速定制和调整强化学习算法。

您可以从[https://gitee.com/mindspore/reinforcement/tree/master/example/dqn](https://gitee.com/mindspore/reinforcement/tree/master/example/dqn)获取DQN算法代码。

## 算法相关参数配置

MindSpore-RL使用`algorithm_config`定义逻辑组件和相应的超参配置。`algorithm_config`是一个Python字典，分别描述actor、learner、policy_and_network、collect_environment、eval_environment和replaybuffer。框架可以基于配置执行算法，用户仅需聚焦算法设计。

下述代码定义了一组算法配置，并使用algorithm_config创建`Session`，`Session`负责分配资源并执行计算图编译和执行。

```python
from mindspore_rl.mindspore_rl import Session
algorithm_config = {
    'actor': {...},
    'learner': {...},
    'policy_and_network': {...},
    'collect_environment': {...},
    'eval_environment': {...},
    'replay_buffer': {...}
}

session = Session(algorithm_config)
session.run(...)
```

下文将详细介绍algorithm_config中各个参数含义及使用方式。

### Policy配置参数

Policy通常用于智能体决策下一步需要执行的行为，算法中需要policy类型名`type`和参数`params`：

- `type`：指定Policy的类型，Actor通过Policy决策应该采取的动作。在深度强化学习中，Policy通常采用深度神经网络提取环境特征，并输出下一步采取的动作。
- `params`：指定实例化相应Policy的参数。这里需要注意的是，`params`和`type`需要匹配。

以下样例中定义策略和参数配置，Policy是由用户定义的`DQNPolicy`，并指定了epsilon greedy衰减参数，学习率，网络模型隐层等参数，框架会采用`DQNPolicy(policy_params)`方式创建Policy对象。

```python
from dqn.src.dqn import DQNPolicy

policy_params = {
    'epsi_high': 0.1,        # epsi_high/epsi_low/decay共同控制探索-利用比例
    'epsi_low': 0.1,         # epsi_high：最大探索比例，epsi_low：最低探索比例，decay：衰减步长
    'decay': 200,
    'state_space_dim': 0,    # 状态空间维度大小，0表示从外部环境中读取状态空间信息
    'action_space_dim': 0,   # 动作空间维度大小，0表示从外部环境中获取动作空间信息
    'hidden_size': 100,      # 隐层维度
}

algorithm_config = {
    ...
    'policy_and_network': {
        'type': DQNPolicy,
        'params': policy_params,
    },
    ...
}
```

|  键值  |        类型        |             范围              |                             说明                             |
| :----: | :----------------: | :---------------------------: | :----------------------------------------------------------: |
|   type   |       Class        | 用户定义的继承learner并实现虚函数的类 |        和用户定义的继承learner并实现虚函数的类名相同         |
|  params(可选)  | Dictionary |     任意key value形式的值或者None     | 自定义参数，用户可以通过key value的形式传入任何值 |

### Environment配置参数

`collect_environment`和`eval_environment`分别表示运行过程中收集数据的环境和用来评估模型的环境，算法中需要指定类型名`number`，`type`和参数`params`：

- `number`：在算法中所需要的环境数量。

- `type`：指定环境的类型名，这里可以是Reinforcement内置的环境，例如`Environment`，也可以是用户自定义的环境类型。

- `params`：指定实例化相应外部环境的参数。需要注意的是，`params`和`type`需要匹配。

以下样例中定义了外部环境配置，框架会采用`Environment(name='CartPole-v0')`方式创建`CartPole-v0`外部环境。`collect_environment`和`eval_environment`的配置参数是一样的。

```python
from mindspore_rl.environment import GymEnvironment
collect_env_params = {'name': 'CartPole-v0'}
eval_env_params = {'name': 'CartPole-v0'}
algorithm_config = {
    ...
    'collect_environment': {
        'number': 1,
        'type': GymEnvironment,            # 外部环境类名
        'params': collect_env_params       # 环境参数
    },
    'eval_environment': {
        'number': 1,
        'type': GymEnvironment,            # 外部环境类名
        'params': eval_env_params          # 环境参数
    },
    ...
}
```

|        键值        |    类型    |             范围              |                             说明                             |
| :----------------: | :--------: | :---------------------------: | :----------------------------------------------------------: |
|    number(可选)    |  Integer   |            [1, +∞)            | 当用户选择填写number这项时，填入的环境数量至少为1个。当用户不选择填入number这项时，框架会直接创建环境实例而不会调用`MultiEnvironmentWrapper`类来包装环境 |
| num_parallel(可选) |  Integer   |          [1, number]          | 不填时默认开启环境并行。用户可通过填写num_parallel: 1来关闭环境并行，或者配置自己需要的并行参数。 |
|        type        |   Class    |      Environment类的子类      |                         外部环境类名                         |
|    params(可选)    | Dictionary | 任意key value形式的值或者None |      自定义参数，用户可以通过key value的形式传入任何值       |

### Actor配置参数

`Actor`负责与外部环境交互。通常`Actor`需要基于`Policy` 与`Env`交互，部分算法中还会将交互得到的经验存入`ReplayBuffer`中，因此`Actor`会持有`Policy`和`Environment`，并且按需创建`ReplayBuffer`。`Actor配置参数`中，`policies/networks`指定`Policy`中的成员对象名称。

以下代码中定义`DQNActor`配置，框架会采用`DQNActor(algorithm_config['actor'])`方式创建Actor。

```python
algorithm_config = {
    ...
    'actor': {
        'number': 1,                                                        # Actor个数
        'type': DQNActor,                                                   # Actor类名
        'policies': ['init_policy', 'collect_policy', 'eval_policy'],       # 从Policy中提取名为init_policy/collect_policy/eval_policy成员对象，用于构建Actor
        'pass_environment': True                                            # 是否把环境作为成员变量传入到actor中
        'share_env': True                                                   # 每个actor是否共享环境
    }
    ...
}
```

|       键值       |      类型      |                范围                 |                             说明                             |
| :--------------: | :------------: | :---------------------------------: | :----------------------------------------------------------: |
|      number      |    Integer     |               [1, +∞)               |              目前actor数量暂时不支持1以外的数值              |
|       type       |     Class      | 用户定义的继承actor并实现虚函数的类 |         和用户定义的继承actor并实现虚函数的类名相同          |
|   params(可选)   |   Dictionary   |    任意key value形式的值或者None    |      自定义参数，用户可以通过key value的形式传入任何值       |
|     policies     | List of String |     和用户定义的策略变量名相同      | 列表中的所有String都应该和用户定义的策略类中初始化的策略变量名一一对应 |
|  networks(可选)  | List of String |       和用户定义的网络变量名相同     | 列表中的所有String都应该和用户定义的策略类中初始化的网络变量名一一对应 |
| pass_environment |    Boolean     |            True 或 False            | 如果值为False，将不能从actor中获得collect_environment和eval_evironment的实例 |
|  share_env(可选) |    Boolean     |            True 或 False            | 默认值为True, 即各个actor共享一个环境。如果为False, 则单独为每个actor创建一个collect环境实例 |

### ReplayBuffer配置参数

在部分算法中，`ReplayBuffer`用于储存Actor和环境交互的经验。之后会从`ReplayBuffer`中取出数据，用于网络训练。

```python
from mindspore_rl.core.replay_buffer import ReplayBuffer
algorithm_config = {
    ...
    'replay_buffer': {'number': 1,
                      'type': ReplayBuffer,
                      'capacity': 100000,                                           # ReplayBuffer容量
                      'sample_size': 64,                                            # 采样Batch Size
                      'data_shape': [(4,), (1,), (1,), (4,)],                       # ReplayBuffer的维度信息
                      'data_type': [ms.float32, ms.int32, ms.float32, ms.float32]}, # ReplayBuffer数据类型
}
```

|       键值        |            类型             |                 范围                 |                         说明                          |
| :---------------: | :-------------------------: | :----------------------------------: | :---------------------------------------------------: |
|      number       |           Integer           |               [1, +∞)                |                   需要的Buffer数量                    |
|       type        |            Class            | 用户定义或者框架提供的ReplayBuffer类 |     用户定义或者框架提供的ReplayBuffer的类名相同      |
|     capacity      |           Integer           |               [0, +∞)                |                   ReplayBuffer容量                    |
|    data_shape     |    List of Integer Tuple    |               [0, +∞)                | Tuple中的第一个值需要和环境数量相等，如是单环境则不填 |
|     data_type     | List of mindspore data type |      需要是MindSpore的数据类型       |         data_type的长度和data_shape的长度相同         |
| sample_size(可选) |           Integer           |            [0, capacity]             |          值必须小于capacity。不填时，默认为1          |

### Learner配置参数

`Learner`负责基于历史经验对网络权重进行更新。`Learner`中持有`Policy`中定义的DNN网络（由`networks`指定`Policy`的成员对象名称），用于损失函数计算和网络权重更新。

以下代码中定义`DQNLearner`配置，框架会采用`DQNLearner(algorithm_config['learner'])`方式创建Learner。

```python
from dqn.src.dqn import DQNLearner
learner_params = {'gamma': 0.99,
                  'lr': 0.001,             # 学习率
                 }  
algorithm_config = {
    ...
    'learner': {
        'number': 1,                                      # Learner个数
        'type': DQNLearner,                               # Learner类名
        'params': learner_params,                         # Learner 需要的参数
        'networks': ['policy_network', 'target_network']  # Learner从Policy中提取名为target_net/policy_network成员对象，用于更新
    },
    ...
}
```

|     键值     |      类型      |                 范围                  |                             说明                             |
| :----------: | :------------: | :-----------------------------------: | :----------------------------------------------------------: |
|    number    |    Integer     |                [1, +∞)                |             目前learner数量暂时不支持1以外的数值             |
|     type     |     Class      | 用户定义的继承learner并实现虚函数的类 |        和用户定义的继承learner并实现虚函数的类名相同         |
| params(可选) |   Dictionary   |     任意key value形式的值或者None     |      自定义参数，用户可以通过key value的形式传入任何值       |
|   networks   | List of String |        和定义的网络名变量相同         | 列表中的所有String都应该和用户定义的策略类中初始化的网络变量名一一对应 |
