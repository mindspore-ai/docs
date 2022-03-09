# MindSpore RL Configuration Instruction

<a href="https://gitee.com/mindspore/docs/blob/master/docs/reinforcement/docs/source_en/custom_config_info.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>
&nbsp;&nbsp;

## Overview

Recent years, deep reinforcement learning is developing by leaps and bounds, new algorithms come out every year. To offer high scalability and reuable reinforcement framework, MindSpore RL separates an algorithm into several parts, such as Actor, Learner, Policy, Environment, ReplayBuffer, etc. Moreover, due to the complexity of deep reinforcement learning algorithm, its performance is largely influenced by different hyper-parameters. MindSpore RL provides centeral configuration API, which decouples the algorithm from deployment and execution considerations to help users adjust model and algorithm conveniently.

This instruction uses DQN algorithm as an example to introduce how to use this configuration API, and help users customize their algorithms.

You can obtain the code of DQN algorithm from [https://gitee.com/mindspore/reinforcement/tree/master/example/dqn](https://gitee.com/mindspore/reinforcement/tree/master/example/dqn).

## Configuration Details

MindSpore RL uses `algorithm_config` to define each algorithm component and corresponding hyper-parameters. `algorithm_config` is a Python dictionary, which describes actor, learner, policy, collect_environment, eval_environment and replay buffer respectively. Framework can arrange the execution and deployment, which means that user only needs to focus on the algorithm design.

The following code defines a set of algorithm configurations and uses `algorithm_config` to create a `Session`. `Session` is responsible for allocating resources and executing computational graph compilation and execution.

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

Each parameter and their instruction in algorithm_config will be described below.

### Policy Configuration

Policy is usually used to determine the behaviour (or action) that agent will execute in the next step, it takes `type` and `params` as the subitems.

- `type` : specify the name of Policy, Actor determines the action through Policy. In deep reinforcement learning, Policy usually uses deep neural network to extract the feature of environment, and outputs the action in the next step.
- `params` : specify the parameter that used during creating the instance of Policy. One thing should be noticed is that `type` and `params` need to be matched.

```python
from dqn.src.dqn import DQNPolicy

policy_params = {
    'epsi_high': 0.1,        # epsi_high/epsi_low/decay control the proportion of exploitation and exploration
    'epsi_low': 0.1,         # epsi_high：the highest probability of exploration，epsi_low：the lowest probability of exploration，
    'decay': 200,            # decay：the step decay
    'state_space_dim': 0,    # the dimension of state space，0 means that it will read from the environment automatically
    'action_space_dim': 0,   # the dimension of action space，0 means that it will read from the environment automatically
    'hidden_size': 100,      # the dimension of hidden layer
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

|       key        |    Type    |                  Range                  |                         Description                          |
| :--------------: | :--------: | :-------------------------------------: | :----------------------------------------------------------: |
|       type       |   Class    |         The user-defined class          |       This type is the same name as user-defined class       |
| params(optional) | Dictionary | Any value with key value format or None | Customized parameter, user can input any value with key value format |

### Environment Configuration

`collect_environment` and `eval_environment` are used to collect experience during interaction with environment and evaluate model after training respectively. `number`,  `type` and `params` need to be provided to create their instances.

- `number`: number of environment used in the algorithm.
- `type` : specify the name of environment, which could be either environment from MindSpore RL, such as `GymEnvironment` or user defined environment.
- `params` : specify the parameter that used during creating the instance of environment. One thing should be noticed is that `type` and `params` need to be matched.

The following example defines the configuration of environment. Framework will create a`CartPole-v0` environment like `Environment(name='CartPole-v0')` . The configuration of `collect_environment` and `eval_environment` are the same.

```python
from mindspore_rl.environment import GymEnvironment
collect_env_params = {'name': 'CartPole-v0'}
eval_env_params = {'name': 'CartPole-v0'}
algorithm_config = {
    ...
    'collect_environment': {
        'type': GymEnvironment,            # the class name of environment
        'params': collect_env_params       # parameter of environment
    },
    'eval_environment': {
        'number': 1,
        'type': GymEnvironment,            # the class name of environment
        'params': eval_env_params          # parameter of environment
    },
    ...
}
```

|          key           |    Type    |                            Range                             |                         Description                          |
| :--------------------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   number (optional)    |  Integer   |                           [1, +∞)                            | When user fills the number of environment, number must be larger than 0. When user does not fill it, framework will not wrap environment by  `MultiEnvironmentWrapper` |
| num_parallel(optional) |  Integer   |                         [1, number]                          | If user does not fill it, the environment will run in parallel by default. User can fill num_parallel: 1 to turn off the parallel environment, or enter their own parallel configuration |
|          type          |   Class    | The subclass of environment that is user-defined and implemented |                The class name of environment                 |
|         params         | Dictionary |           Any value with key value format or None            | Customized parameter, user can input any value with key value format |

### Actor Configuration

`Actor` is charge of interacting with environment. Generally, `Actor` interacts with `Env` through `Policy`. Some algorithms will store the experience which obtained during the interaction into `ReplayBuffer`. Therefore, `Actor` will take `Policy` , `Environment` and `ReplayBuffer`. In Actor configuration, `policies`  and `networks` need to specify the name of member variable in `Policy`.

The following code defines the configuration of  `DQNActor` . Framework will create the instance of Actor like `DQNActor(algorithm_config['actor'])`.

```python
algorithm_config = {
    ...
    'actor': {
        'number': 1,                                                        # the number of Actor
        'type': DQNActor,                                                   # the class name of Actor
        'policies': ['init_policy', 'collect_policy', 'eval_policy'],       # Take the policies that called init_policy, collect_policy and eval_policy in Policy class as input to create the instance of actor
        'pass_environment': True                                            # Whether pass the environment instance to actor
    }
    ...
}
```

|        key         |      Type      |                           Range                            |                         Description                          |
| :----------------: | :------------: | :--------------------------------------------------------: | :----------------------------------------------------------: |
|       number       |    Integer     |                          [1, +∞)                           |          Number of Actor, currently only support 1           |
|        type        |     Class      | The subclass of actor that is user-defined and implemented | This type is the same name as the subclass of actor that is user-defined and implemented |
|  params(optional)  |   Dictionary   |          Any value with key value format or None           | Customized parameter, user can input any value with key value format |
|      policies      | List of String |      Same variable name as the user-defined policies       | Every string in list must match with policies' name which is user initialized in  defined policy class |
| networks(optional) | List of String |      Same variable name as the user-defined networks       | Every string in list must match with networks' name which is user initialized in  defined policy class |
|  pass_environment  |    Boolean     |                       True or False                        | If this value is False, user can not obtain `collect_environment` and `eval_environment` instance in actor |

### ReplayBuffer Configuration

For part of algorithms, `ReplayBuffer` is used to store experience which is obtained by interaction between actor and environment. Then experience will be used to train the network.

```python
algorithm_config = {
    ...
    'replay_buffer': {'number': 1
                      'type': ReplayBuffer,
                      'capacity': 100000,                                            # the capacity of ReplayBuffer
                      'sample_size': 64,                                             # sample Batch Size
                      'data_shape': [(4,), (1,), (1,), (4,)],                        # the dimension info of ReplayBuffer
                      'data_type': [ms.float32, ms.int32, ms.float32, ms.float32]},  # the data type of ReplayBuffer
}
```

|          key          |            Type             |                    Range                    |                         Description                          |
| :-------------------: | :-------------------------: | :-----------------------------------------: | :----------------------------------------------------------: |
|        number         |           Integer           |                   [1, +∞)                   |                Number of replaybuffer created                |
|         type          |            Class            | User-defined or provided ReplayBuffer class | This type is the same name as the user-defined or provided ReplayBuffer class |
|       capacity        |           Integer           |                   [0, +∞)                   |                 The capacity of ReplayBuffer                 |
|      data_shape       |    List of Integer Tuple    |                   [0, +∞)                   | The first number of tuple must equal to number of environment |
|       data_type       | List of mindspore data type |       Belongs to MindSpore data type        | The length of this list must equal to the length of data_shape |
| sample_size(optional) |           Integer           |                [0, capacity]                | The maximum value is the capacity of replay buffer. Default 1 |

### Learner Configuration

`Learner` is used to update the weights of neural network according to experience. `Learner` holds the DNN which is defined in `Policy` (the name of member variable in `Policy` match with the contains in `networks`), which is used to calculate the loss and update the weights of neural network.

The following code defines the configuration of `DQNLearner` . Framework will create the instance of Learner like `DQNLearner(algorithm_config['learner'])`.

```python
from dqn.src.dqn import DQNLearner
learner_params = {'gamma': 0.99,
                  'lr': 0.001,                          # learning rate
                 }  
algorithm_config = {
    ...
    'learner': {
        'number': 1,                                      # the number of Learner
        'type': DQNLearner,                               # the class name of Learner
        'params': learner_params,                         # the decay rate
        'networks': ['policy_network', 'target_network']  # Learner takes the policy_network and target_network from DQNPolicy as input argument to update the network
    },
    ...
}
```

|   key    |      Type      |                       Range                        |                         Description                          |
| :------: | :------------: | :------------------------------------------------: | :----------------------------------------------------------: |
|  number  |    Integer     |                      [1, +∞)                       |          Number of Actor, currently only support 1           |
|   type   |     Class      | The user-defined and implement subclass of learner | This type is the same name as the subclass of learner that is user-defined and implemented |
|  params  |   Dictionary   |      Any value with key value format or None       | Customized parameter, user can input any value with key value format. |
| networks | List of String |   Same variable name as the user-defined network   | Every string in list must match with networks' name which is user initialized in  defined policy class |
