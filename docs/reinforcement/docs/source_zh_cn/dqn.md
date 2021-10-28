# 使用MindSpore Reinforcement实现深度Q学习（DQN）

<!-- TOC -->

- [使用MindSpore Reinforcement实现深度Q学习（DQN）](#使用mindspore-reinforcement实现深度q学习dqn)
    - [摘要](#摘要)
    - [指定DQN的Actor-Learner-Environment抽象](#指定dqn的actor-learner-environment抽象)
        - [定义DQNTrainer类](#定义dqntrainer类)
        - [定义DQNPolicy类](#定义dqnpolicy类)
        - [定义DQNActor类](#定义dqnactor类)
        - [定义DQNLearner类](#定义dqnlearner类)
    - [执行并查看结果](#执行并查看结果)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/reinforcement/docs/source_zh_cn/dqn.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>
&nbsp;&nbsp;

## 摘要

为了使用MindSpore Reinforcement实现强化学习算法，用户需要：

- 提供算法配置，将算法的实现与其部署细节分开；
- 基于Actor-Learner-Environment抽象实现算法；
- 创建一个执行已实现的算法的会话对象。

本教程展示了使用MindSpore Reinforcement API实现深度Q学习（DQN）算法。注：为保证清晰性和可读性，仅显示与API相关的代码，不相关的代码已省略。点击[此处](https://gitee.com/mindspore/reinforcement/tree/master/example/dqn)获取MindSpore Reinforcement实现完整DQN的源代码。

## 指定DQN的Actor-Learner-Environment抽象

DQN算法需要两个深度神经网络，一个*策略网络*用于近似动作值函数(Q函数)，另一个*目标网络*用于稳定训练。策略网络指如何对环境采取行动的策略，DQN算法的目标是训练策略网络以获得最大的奖励。此外，DQN算法使用*经验回放*技术来维护先前的观察结果，进行off-policy学习。其中Actor使用不同的行为策略来对环境采取行动。

MindSpore Reinforcement使用*算法配置*指定DQN算法所需的逻辑组件（Agent、Actor、Learner、Environment）和关联的超参数。根据提供的配置，它使用不同的策略执行算法，以便用户可以专注于算法设计。

算法配置是一个Python字典，指定如何构造DQN算法的不同组件。每个组件的超参数在单独的Python字典中配置。DQN算法配置定义如下：

```python
    'actor': {
        'number': 1,
        'type': DQNActor,
        'params': None,
        'policies': ['init_policy', 'collect_policy', 'evaluate_policy'],
        'networks': ['policy_network', 'target_network'],
        'environment': True,
        'eval_environment': True,
        'replay_buffer': {'capacity': 100000, 'shape': [(4,), (1,), (1,), (4,)],
                          'sample_size': 64, 'type': [ms.float32, ms.int32, ms.float32, ms.float32]},
    },
    'learner': {
        'number': 1,
        'type': DQNLearner,
        'params': learner_params,
        'networks': ['target_network', 'policy_network_train']
    },
    'policy_and_network': {
        'type': DQNPolicy,
        'params': policy_params
    },
    'environment': {
        'type': GymEnvironment,
        'params': env_params
    },
    'eval_environment': {
        'type': GymEnvironment,
        'params': eval_env_params
    }
}
```

以上配置定义了四个顶层项，每个配置对应一个算法组件：*actor、learner、policy*和*environment*。每个项对应一个类，该类必须由用户定义，以实现DQN算法的逻辑。

顶层项具有描述组件的子项。*number*定义算法使用的组件的实例数。*class*表示必须定义的Python类的名称，用于实现组件。*parameters*为组件提供必要的超参数。*policy*定义组件使用的策略。*networks*列出了此组件使用的所有神经网络。*environment*说明组件是否与环境交互。在DQN示例中，只有Actor与环境交互。*reply_buffer*定义回放缓冲区的*容量、形状、样本大小和数据类型*。

对于DQN算法，我们配置了一个Actor `'number': 1`，三个行为策略`'policies': ['init_policy', 'collect_policy', 'evaluation_policy']`，两个神经网络`'networks': ['policy_network', 'target_network']`，环境`'environment': True`，和回放缓冲区`'replay_buffer':{'capacity':100000,'shape':[...],'sample_size':64,'type':[..]}`。

回放缓冲区的容量设置为100,000，其样本大小为64。它存储shape为`[(4,), (1,), (1,), (4,)]`的张量数据。第二个维度的类型为int32，其他维度的类型为float32。这两种类型都由MindSpore提供：`'type': [mindspore.float32, mindspore.int32, mindspore.float32, mindspore.float32]}`。

其他组件也以类似的方式定义。有关更多详细信息，请参阅[完整代码示例](https://gitee.com/mindspore/reinforcement/tree/master/example/dqn)和[https://www.mindspore.cn/reinforcement/api/zh-CN/master/index.html]。

请注意，MindSpore Reinforcement使用单个*policy*类来定义算法使用的所有策略和神经网络。通过这种方式，它隐藏了策略和神经网络之间数据共享和通信的复杂性。

MindSpore Reinforcement在*session*的上下文中执行算法。会话分配资源（在一台或多台群集计算机上）并执行编译后的计算图。用户传入算法配置以实例化Session类：

```python
dqn_session = Session(dqn_algorithm_config)
```

调用Session对象上的run方法执行DQN算法：

```python
dqn_session.run(class_type=DQNTrainer, episode=650, parameters=trainer_parameters)
```

`run`方法将DQNTrainer类作为输入。下面描述了用于DQN算法的训练循环。

为使用MindSpore的计算图功能，将执行模式设置为`GRAPH_MODE`。

```python
from mindspore import context
context.set_context(mode=context.GRAPH_MODE)
```

`GRAPH_MODE`允许以`@ms_function`注释的函数和方法编译到[MindSopre计算图](https://www.mindspore.cn/docs/programming_guide/en/master/api_structure.html)用于自动并行和加速。在本教程中，我们使用此功能来实现一个高效的`DQNTrainer`类。

### 定义DQNTrainer类

`DQN训练器`类表示训练循环，该循环迭代地从回放缓冲区收集经验并训练目标模型。它必须继承自`Trainer`类，该类是MindSpore Reinforcement API的一部分。

`Trainer`基类包含`MSRL`(MindSpore Reinforcement Learning)对象，该对象允许算法实现与MindSpore Reinforcement交互，以实现训练逻辑。`MSRL`类根据先前定义的算法配置实例化RL算法组件。它提供了函数处理程序，这些处理程序透明地绑定到用户定义的Actor、Learner或回放缓冲区对象的方法。因此，`MSRL`类让用户能够专注于算法逻辑，同时它透明地处理一个或多个worker上不同算法组件之间的对象创建、数据共享和通信。用户通过使用算法配置创建上文提到的`Session`对象来实例化`MSRL`对象。

`DQNTrainer`必须重载训练方法。在本教程中，它的定义如下：

```python
class DQNTrainer(Trainer):
    ...
    def train(self, episode):
        self.init_training()
        for i in range(episode):
           reward, episode_steps = self.train_one_episode(self.update_period)
        reward = self.evaluation()
```

`train`方法首先调用`init_training`初始化训练。然后，它为指定数量的episode（iteration）训练模型，每个episode调用用户定义的`train_one_episode`方法。最后，train方法通过调用`evaluation`方法来评估策略以获得奖励值。

在训练循环的每次迭代中，调用`tre_one_episode`方法来训练一个episode：

```python
@ms_function
def train_one_episode(self, update_period=5):
    """Train one episode"""
    state, done = self.msrl.agent_reset_collect()
    total_reward = self.zero
    steps = self.zero
    while not done:
        done, r, new_state, action, my_reward = self.msrl.agent_act(state)
        self.msrl.replay_buffer_insert([state, action, my_reward, new_state])
        state = new_state
        r = self.squeeze(r)
        self.msrl.agent_learn(self.msrl.replay_buffer_sample())
        total_reward += r
        steps += 1
        if not self.mod(steps, update_period):
            self.msrl.agent_update()
    return total_reward, steps
```

`@ms_function`注解表示此方法将被编译为MindSpore计算图用于加速。所有标量值都必须定义为张量类型，例如`self.zero_value = Tensor(0, mindspore.float32)`。

`train_one_episode`方法首先调用`msrl.agent_reset_collect`函数（由MindSpore Reinforcement API提供）来重置环境。然后，它使用`msrl.agent_act`函数处理程序从环境中收集经验，并使用`msrl.agent_learn`函数训练目标模型。`msrl.agent_learn`的输入是`msrl.sample_replay_buffer`返回的采样结果。

回放缓存`ReplayBuffer`由MindSpore Reinfocement提供。它定义了`insert`和`sample`方法，分别用于对经验数据进行存储和采样。

`init_training`和`evaluation`方法的实现类似。详细信息，请参阅[完整的DQN代码示例](https://gitee.com/mindspore/reinforcement/tree/master/example/dqn)。

### 定义DQNPolicy类

定义`DQNPolicy`类，用于实现神经网络并定义策略。

```python
class DQNPolicy():
     def __init__(self, params):
        self.policy_network = FullyConnectedNet(
            params['state_space_dim'],
            params['hidden_size'],
            params['action_space_dim'])
        self.target_network = FullyConnectedNet(
            params['state_space_dim'],
            params['hidden_size'],
            params['action_space_dim'])
```

构造函数将先前定义的Python字典类型的超参数`policy_parameters`作为输入。

在定义策略网络和目标网络之前，用户必须使用MindSpore算子定义神经网络的结构。例如，它们可能是`FullyConnectedNetwork`类的对象，该类定义如下：

```python
class FullyConnectedNetwork(mindspore.nn.Cell):
     def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnectedNet, self).__init__()
        self.linear1 = nn.Dense(
            input_size,
            hidden_size,
            weight_init="XavierUniform")
        self.linear2 = nn.Dense(
            hidden_size,
            output_size,
            weight_init="XavierUniform")
        self.relu = nn.ReLU()
```

DQN算法使用损失函数来优化神经网络的权重。此时，用户必须定义一个用于计算损失函数的神经网络。此网络被指定为`DQNPolicy`的嵌套类。此外，还需要优化器来训练网络。优化器和损失函数定义如下：

```python
class DQNPolicy():
     def __init__(self, params):
        ...
        loss_fn = mindspore.nn.MSELoss()
        optimizer =  mindspore.nn.Adam(self.policy_net.trainable_params(),
                                       learning_rate=params['lr'])
        loss_Q_net = self.PolicyNetWithLossCell(self.policy_network, loss_fn)
        self.policy_network_train = mindspore.nn.TrainOneStepCell(loss_Q_net, otimizer)
        self.policy_network_train.set_train(mode=True)
```

DQN算法是一种*off-policy*算法，使用贪婪策略学习。它使用不同的行为策略来对环境采取行动和收集数据。在本示例中，我们用`RandomPolicy`初始化训练，用`EpsilonGreedyPolicy`收集训练期间的经验，用`GreedyPolicy`进行评估：

```python
class DQNPolicy():
     def __init__(self, params):
         ...
        self.init_policy = RandomPolicy(params['action_space_dim'])
        self.collect_policy = EpsilonGreedyPolicy(self.policy_network, (1, 1), params['epsi_high'],
                                                  params['epsi_low'], params['decay'], params['action_space_dim'])
        self.evaluate_policy = GreedyPolicy(self.policy_network)
```

由于上述三种行为策略在一系列RL算法中非常常见，MindSpore Reinforcement将它们作为可重用的构建块提供。用户还可以自定义特定算法的行为策略。

请注意，参数字典的方法名称和键必须与前面定义的算法配置一致。

### 定义DQNActor类

定义一个新的Actor组件用于实现`DQNActor`，该组件继承了MindSpore Reinforcement提供的`Actor`类。然后，必须重载trainer使用的方法：

```python
class DQNActor(Actor):
     ...
    def act_init(self, state):
        """Fill the replay buffer"""
        action = self.init_policy()
        new_state, reward, done = self._environment.step(action)
        action = self.reshape(action, (1,))
        my_reward = self.select(done, self.penalty, self.reward)
        return done, reward, new_state, action, my_reward

    def act(self, state):
        """Experience collection"""
        self.step += 1

        ts0 = self.expand_dims(state, 0)
        step_tensor = self.ones((1, 1), ms.float32) * self.step

        action = self.collect_policy(ts0, step_tensor)
        new_state, reward, done = self._environment.step(action)
        action = self.reshape(action, (1,))
        my_reward = self.select(done, self.penalty, self.reward)
        return done, reward, new_state, action, my_reward

    def evaluate(self, state):
        """Evaluate the trained policy"""
        ts0 = self.expand_dims(state, 0)
        action = self.evaluate_policy(ts0)
        new_state, reward, done = self._eval_env.step(action)
        return done, reward, new_state
```

这三种方法使用不同的策略作用于指定的环境，这些策略将状态映射到操作。这些方法将张量类型的值作为输入，并从环境返回轨迹。

为了与环境交互，Actor使用`Environment`类中定义的`step(action)`方法。对于应用到指定环境的操作，此方法会做出反应并返回三元组。三元组包括应用上一个操作后的新状态、作为浮点类型获得的奖励以及用于终止episode和重置环境的布尔标志。

回放缓冲区类`ReplayBuffer`定义了一个`insert`方法，`DQNActor`对象调用该方法将经验数据存储在回放缓冲区中。

`Environment`类和`ReplayBuffer`类由MindSpore Reinforcement API提供。

`DQNActor`类的构造函数定义了环境、回放缓冲区、策略和网络。它将字典类型的参数作为输入，这些参数在算法配置中定义。下面，我们只展示环境的初始化，其他属性以类似的方式分配：

```python
class DQNActor(Actor):
     def __init__(self, params):
         self._environment = params['environment']
         ...
```

### 定义DQNLearner类

为了实现`DQNLearner`，类必须继承MindSpore Reinforcement API中的`Learner`类，并重载`learn`方法：

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

在这里，`learn`方法将轨迹（从回放缓冲区采样）作为输入来训练策略网络。构造函数通过从算法配置接收字典类型的配置，将网络、策略和折扣率分配给DQNLearner：

```python
class DQNLearner(Learner):
        def __init__(self, params=None):
             self.target_network = params['target_network']
             self.policy_network_train = params['policy_network_train']
             self.gamma = Tensor(params['gamma'], mindspore.float32)
```

## 执行并查看结果

执行脚本`train.py`以启动DQN模型训练。

```python
cd example/dqn/
python train.py
```

执行结果如下：

```text
-----------------------------------------
Evaluation result in episode 0 is 95.300
-----------------------------------------
Episode 0, steps: 33.0, reward: 33.000
Episode 1, steps: 45.0, reward: 12.000
Episode 2, steps: 54.0, reward: 9.000
Episode 3, steps: 64.0, reward: 10.000
Episode 4, steps: 73.0, reward: 9.000
Episode 5, steps: 82.0, reward: 9.000
Episode 6, steps: 91.0, reward: 9.000
Episode 7, steps: 100.0, reward: 9.000
Episode 8, steps: 109.0, reward: 9.000
Episode 9, steps: 118.0, reward: 9.000
...
...
Episode 200, steps: 25540.0, reward: 200.000
Episode 201, steps: 25740.0, reward: 200.000
Episode 202, steps: 25940.0, reward: 200.000
Episode 203, steps: 26140.0, reward: 200.000
Episode 204, steps: 26340.0, reward: 200.000
Episode 205, steps: 26518.0, reward: 178.000
Episode 206, steps: 26718.0, reward: 200.000
Episode 207, steps: 26890.0, reward: 172.000
Episode 208, steps: 27090.0, reward: 200.000
Episode 209, steps: 27290.0, reward: 200.000
-----------------------------------------
Evaluation result in episode 210 is 200.000
-----------------------------------------
```
