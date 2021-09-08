# Deep Q Learning (DQN) with MindSpore Reinforcement

`Linux` `Ascend` `GPU` `CPU` `reinforcement learning`

<!-- TOC -->

- [Deep Q Learning (DQN) with MindSpore Reinforcement](#deep-q-learning-dqn-with-mindspore-reinforcement)
    - [summary](#summary)
    - [Specifying the Actor-Learner-Environment Abstraction for DQN](#specifying-the-actor-learner-environment-abstraction-for-dqn)
        - [Defining the DQNTrainer class](#defining-the-dqntrainer-class)
        - [Defining the DQNPolicy class](#defining-the-dqnpolicy-class)
        - [Defining the DQNActor class](#defining-the-dqnactor-class)
        - [Defining the DQNLearner class](#defining-the-dqnlearner-class)
    - [Execute and view results](#execute-and-view-results)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/reinforcement/docs/source_en/dqn.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## summary

To implement an reinforcement learning algorithm with MindSpore Reinforcement, a user needs to:

- provide an algorithm configuration, which separates the implementation of the algorithm from its deployment details;
- implement the algorithm based on an actor-learner-environment abstraction;
- create a session object that executes the implemented algorithm.

This tutorial shows the use of the MindSpore Reinforcement API to implement the Deep Q Learning (DQN) algorithm. Note that, for [clarity](https://dictionary.cambridge.org/dictionary/english/clarity) and readability, only API-related code sections are presented, and irrelevant code is omitted. The source code of the full DQN implementation for MindSpore Reinforcement can be found [here]

You can find DQN example at [https://gitee.com/mindspore/reinforcement/tree/master/example/dqn](https://gitee.com/mindspore/reinforcement/tree/master/example/dqn)

## Specifying the Actor-Learner-Environment Abstraction for DQN

The DQN algorithm requires two deep neural networks, a *policy network* for approximating the action-value function (Q function) and a *target network* for stabilising the training. The policy network is the strategy on how to act on the environment, and the goal of the DQN algorithm is to train the policy network for maximum reward. In addition, the DQN algorithm uses an *experience replay* technique to maintain previous observations for off-policy learning, where an actor uses different behavioural policies to act on the environment.

MindSpore Reinforcement uses an *algorithm configuration* to specify the logical components (agents, actors, learners, environments) required by the DQN algorithm and the associated hyperparameters. It can execute the algorithm with different strategies based on the provided configuration, which allows the user to focus on the algorithm design.

The algorithm configuration is a Python dictionary that specifies how to construct different components of the DQN algorithm. The hyper-parameters of each component are configured in separate Python dictionaries. The DQN algorithm configuration can be defined as follows:

```python
dqn_algorithm_configuration = {
   'actor': {
       'number': 1,
       'class': DQNActor,
       'parameters': None,
       'policies': ['init_policy', 'collect_policy', 'evaluation_policy'],
       'networks': ['policy_network', 'target_network'],
       'environment': True,
       'replay_buffer': {'capacity': 100000, 'shape': [(4,), (1,), (1,), (4,)],
                         'sample_size': 64,
                         'type': [mindspore.float32, mindspore.int32,
                                  mindspore.float32, mindspore.float32]},
   },

   'learner': {
       'number': 1,
       'class': DQNLearner,
       'parameters': learner_parameters,
       'networks': ['target_network', 'policy_network_train']
    },

   'policy': {
       'class': DQNPolicy,
       'parameters': policy_parameters
    },

   'environment': {
       'class': Environment,
       'parameters': environment_parameters
    }
}

```

The configuration defines four top-level entries, each corresponding to an algorithmic component: *actor, learner, policy* and the *environment*. Each entry corresponds to a class, which must be defined by the user to implement the DQN algorithm’s logic.

A top-level entry has sub-entries that describe the component. The *number* entry defines the number of instances of the component used by the algorithm. The *class* entry refers to the name of the Python class that must be defined to implement the component. The *parameters* entry provides the necessary hyper-parameters for the component. The *policies* entry defines the policies used by the component. The *networks* entry lists all neural networks used by this component. The *environment* states if the component interacts with the environment. In the DQN example, only actors interact with the environment. The *reply_buffer* defines the *capacity, shape, sample size and data type* of the replay buffer.

For the DQN algorithm, we configure one actor `'number': 1`, three behaviour policies `'policies': ['init_policy', 'collect_policy', 'evaluation_policy']`, two neural networks `'networks': ['policy_network', 'target_network']`,  the environment `'environment': True`, and a replay buffer `'replay_buffer':{'capacity':100000,'shape':[...],'sample_size':64,'type':[..]}`.

The replay buffer's capacity is set to 100,000, and its sample size is 64. It stores data of a tensor type with shape `[(4,), (1,), (1,), (4,)]`. The second dimension has the type int32 and other dimensions are of the float32 type and. Both types are provided by MindSpore: `'type': [mindspore.float32, mindspore.int32, mindspore.float32, mindspore.float32]}`.

Other components are defined in a similar way -- please refer to the [complete code example] and the [MindSpore Reinforcement API documentation] for more details.

Note that MindSpore Reinforcement uses a single *policy* class to define all policies and neural networks used by the algorithm. In this way, it hides the complexity of data sharing and communication between policies and neural networks.

MindSpore Reinforcement executes the algorithm in the context of a *session*. A session allocates resources (on one or more cluster machines) and executes the compiled computational graph. A user passes the algorithm configuration to instantiate a Session class:

```python
dqn_session = Session(dqn_algorithm_config)
```

To execute the DQN algorithm, a user invokes the run method on the Session object:

```python
dqn_session.run(class_type=DQNTrainer, episode=650, parameters=trainer_parameters)
```

The `run`method takes a  DQNTrainer class as input, which will be described below. It describes the training loop used for the DQN algorithm.

To leverage MindSpore's computational graph feature, users set the execution mode to `GRAPH_MODE`.

```python
from mindspore import context
context.set_context(mode=context.GRAPH_MODE)
```

The `GRAPH_MODE` enables functions and methods that are annotated with `@ms_function` to be compiled into the [MindSpore computational graph](https://www.mindspore.cn/docs/programming_guide/en/master/api_structure.html) for auto-parallelisation and acceleration. In this tutorial, we use this feature to implement an efficient `DQNTrainer` class.

### Defining the DQNTrainer class

The `DQNTrainer` class expresses the training loop that iteratively collects experience from the replay buffer and trains the targeted models. It must inherit from the `Trainer` class, which is part of the MindSpore Reinforcement API.

The `Trainer` base class contains an `MSRL` (MindSpore Reinforcement Learning) object, which allows the algorithm implementation to interact with MindSpore Reinforcement to implement the training logic. The `MSRL` class instantiates the RL algorithm components based on the previously defined algorithm configuration. It provides the function handlers that transparently bind to methods of actors, learners, or the replay buffer object, as defined by users. As a result, the `MSRL` class enables users to focus on the algorithm logic, while it transparently handles object creation, data sharing and communication between different algorithmic components on one or more workers. Users instantiate the `MSRL` object by creating the previously mentioned `Session` object with the algorithm configuration.

The `DQNTrainer` must overload the train method. In this tutorial, it is defined as follows:

```python
class DQNTrainer(Trainer):
    ...
    def train(self, episode):
        self.init_training()
        for i in range(episode):
           reward, episode_steps=self.train_one_epoch(self.update_period)
        reward = self.evaluation()
```

The `train` method first calls `init_training` to initialize the training. It then trains the models for the specified number of episodes (iterations), with each episode calling the user-defined `train_one_epoch` method. Finally, the train method evaluates the policy to obtain a reward value by calling the `evaluation` method.

In each iteration of the training loop, the `train_one_epoch` method is invoked to train an episode:

```python
@ms_function
def train_one_epoch(self, update_period=5):
    state, done = self.msrl.agent_reset()
    total_reward = self.zero_value
    steps = self.zero_value
    while not done:
        done, r, state = self.msrl.agent_act(state)
        self.msrl.agent_learn(self.msrl.sample_replay_buffer())
        total_reward += r
        steps += 1
    return total_reward, steps
```

The `@ms_function` annotation states that this method will be compiled into a MindSpore computational graph for acceleration. To support this, all scalar values must be defined as tensor types, e.g. `self.zero_value = Tensor(0, mindspore.float32)`.

The `train_one_epoch` method first calls the `msrl.agent_reset` function (provided by the MindSpore Reinforcement API) to reset the environment. It then collects the experience from the environment with the `msrl.agent_act` function handler, and uses the `msrl.agent_learn` function to train the target model. The input of `msrl.agent_learn` is the sampled results returned by  `msrl.sample_replay_buffer`.

The `init_training` and the `evaluation` methods are implemented analogously -- please refer to the [complete DQN code example] for details.

### Defining the DQNPolicy class

To implement the neural networks and define the policies, a user defines the `DQNPolicy` class:

```python
class DQNPolicy():
     def __init__(self, params):
         self.policy_network = FullyConnectedNetwork(
                       params['state_space_dim'],
                       params['hidden_size'],params['action_space_dim'])
         self.target_network = FullyConnectedNetwork(
                       params['state_space_dim'],
                       params['hidden_size'],params['action_space_dim'])
```

The constructor takes as input the previously-defined hyper-parameters of the Python dictionary type, `policy_parameters`.

Before defining the policy network and the target network, users must define the structure of the neural networks using MindSpore operators. For example, they may be objects of the `FullyConnectedNetwork` class, which is defined as follows:

```python
class FullyConnectedNetwork(mindspore.nn.Cell):
     def __init__(self, input_size, hidden_size, output_size):
         self.linear1 = mindspore.nn.Dense(input_size, hidden_size)
         self.linear2 = mindspore.nn.Dense(hidden_size, output_size)
         self.relu = mindspore.nn.ReLU()
```

The DQN algorithm uses a loss function to optimize the weights of the neural networks. At this point, a user must define a neural network used to compute the loss function. This network is specified as a nested class of `DQNPolicy`. In addition, an optimizer is required to train the network. The optimizer and the loss function are defined as follows:

```python
class DQNPolicy():
     def __init__(self, params):
        ...
        class PolicyNetWithLossCell(mindspore.nn.Cell):
            def __init__(self, backbone, loss_fn):
            ...
            loss_fn = mindspore.nn.MSELoss()
            optimizer =  mindspore.nn.Adam(self.policy_net.trainable_params(),
                                           learning_rate=params['lr'])
            loss_Q_net = self.PolicyNetWithLossCell(self.policy_network, loss_fn)
            self.policy_network_train = mindspore.nn.TrainOneStepCell(loss_Q_net, otimizer)
            self.policy_network_train.set_train(mode=True)
```

The DQN algorithm is an *off-policy* algorithm that learns using a greedy policy. It uses different behavioural policies for acting on the environment and collecting data. In this example, we use the `RandomPolicy` to initialize the training, the `EpsilonGreedyPolicy` to collect the experience during the training, and the `GreedyPolicy` to evaluate:

```python
class DQNPolicy():
     def __init__(self, params):
         ...
         self.init_policy = RandomPolicy(params['action_space_dim'])
         self.collect_policy = EpsilonGreedyPolicy(self.policy_network,
                               (1,1),params['epsi_high'],
                               params['epsi_low'], params['decay'],
                               params['action_space_dim'])
         self.evaluation_policy = GreedyPolicy(self.policy_network)
```

Since the above three behavioural policies are common for a range of RL algorithms, MindSpore Reinforcement provides them as reusable building blocks. Users may also define their own algorithm-specific behavioural policies.

Note that the names of the methods and the keys of the parameter dictionary must be consistent with the algorithm configuration defined earlier.

### Defining the DQNActor class

To implement the `DQNActor`, a user defines a new actor component that inherits from the `Actor` class provided by MindSpore Reinforcement. They must then overload the methods used by the trainer:

```python
class DQNActor(Actor):
      ...
     def act_init(self, state):
          # Initialise reply buffer
          action = self.init_policy()
          new_state, reward, done = self._environment.step(action)
          self.replay_buffer.insert([state, action, my_reward, new_state])
          return done, reward, new_state

     def act(self, state):
          # Experience collection
          action = self.collect_policy(state)
          new_state, reward, done = self._environment.step(action)
          self.replay_buffer.insert([state, action, reward, new_state])
          return done, reward, new_state

     def evaluate(self, state):
          # Evaluate policies
          action = self.evaluation_policy(state)
          new_state, reward, done = self._environment.step(action)
          return done, reward, new_state
```

The three methods act on the specified environment with different policies, which map states to actions. The methods take as input a tensor-typed value and return the trajectory from the environment.

To interact with the environment, the actor uses the `step(action)` method defined in the `Environment` class. This method reacts to an action applied to the specified `environment and returns a triplet. The triplet includes a new state after applying the previous action, a obtained reward as a float type, and a boolean flag to terminate an episode and reset the environment.

The replay buffer class, `ReplayBuffer`, defines an `insert` method, which is called by `DQNActor` objects to store the experience data in the replay buffer.

The  `Environment` class and the `ReplayBuffer` class are provided by the MindSpore Reinforcement API.

The constructor of the `DQNActor` class defines the environment, the reply buffer, the polices, and the networks. It takes as input the dictionary-typed parameters, which were defined in the algorithm configuration. Below, we only show the initialisation of the environment, other attributes are assigned in the similar way:

```python
class DQNActor(Actor):
     def __init__(self, params):
         self._environment = params['environment']
         ...
```

### Defining the DQNLearner class

To implement the `DQNLearner`, a class must inherit from the `Learner` class in the MindSpore Reinforcement API and overload the `learn` method:

```python
class DQNLearner(Learner):
     ...
     def learn(self, samples):
         state_0, action_0, reward_, state_1 = samples
         next_state_values = self.target_network(state1)
         y_true = reward_1 + self.gamma * next_state_values
         success = self.policy_network_train(state_0, action_0, y_true)
         return success
```

Here, the `learn` method takes as input the trajectory (sampled from a reply buffer) to train the policy network. The constructor assigns the network, the policy, and the discount rate to the DQNLearner, by receiving a dictionary-typed configuration from the algorithm configuration:

```python
class DQNLearner(Learner):
        def __init__(self, params=None):
             self.target_network = params['target_network']
             self.policy_network_train = params['policy_network_train']
             self.gamma = Tensor(params['gamma'], mindspore.float32)
```

## Execute and view results

Execute script `train.py` to start DQN model training.

```python
cd example/dqn/
python train.py
```

The execution results are shown below:

```text
buffer init start...
buffer init time: 10.17387819290161
-----------------------------------------
Evaluation result in episode 0 is 9.399999618530273
-----------------------------------------
Episode time: 0.05016589164733887, steps: 127.0, reward: 11.0, average time: 4560.535604303534us
Episode time: 0.04132795333862305, steps: 136.0, reward: 9.0, average time: 4591.994815402561us
Episode time: 0.04570341110229492, steps: 146.0, reward: 10.0, average time: 4570.341110229492us
Episode time: 0.04680299758911133, steps: 156.0, reward: 10.0, average time: 4680.299758911133us
Episode time: 0.03685498237609863, steps: 164.0, reward: 8.0, average time: 4606.872797012329us
Episode time: 0.04632306098937988, steps: 174.0, reward: 10.0, average time: 4632.306098937988us
Episode time: 0.04598045349121094, steps: 184.0, reward: 10.0, average time: 4598.045349121094us
Episode time: 0.04650545120239258, steps: 194.0, reward: 10.0, average time: 4650.545120239258us
Episode time: 0.04139065742492676, steps: 203.0, reward: 9.0, average time: 4598.961936102973us
Episode time: 0.05491757392883301, steps: 215.0, reward: 12.0, average time: 4576.464494069417us
-----------------------------------------
Evaluation result in episode 10 is 9.800000190734863
----------------------------------------
```
