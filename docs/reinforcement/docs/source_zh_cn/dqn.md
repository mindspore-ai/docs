# 使用MindSpore Reinforcement实现深度Q学习（DQN）

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/reinforcement/docs/source_zh_cn/dqn.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>
&nbsp;&nbsp;

## 摘要

为了使用MindSpore Reinforcement实现强化学习算法，用户需要：

- 提供算法配置，将算法的实现与其部署细节分开；
- 基于Actor-Learner-Environment抽象实现算法；
- 创建一个执行已实现的算法的会话对象。

本教程展示了使用MindSpore Reinforcement API实现深度Q学习（DQN）算法。注：为保证清晰性和可读性，仅显示与API相关的代码，不相关的代码已省略。点击[此处](https://gitee.com/mindspore/reinforcement/tree/r0.3/example/dqn)获取MindSpore Reinforcement实现完整DQN的源代码。

## 指定DQN的Actor-Learner-Environment抽象

DQN算法需要两个深度神经网络，一个*策略网络*用于近似动作值函数(Q函数)，另一个*目标网络*用于稳定训练。策略网络指如何对环境采取行动的策略，DQN算法的目标是训练策略网络以获得最大的奖励。此外，DQN算法使用*经验回放*技术来维护先前的观察结果，进行off-policy学习。其中Actor使用不同的行为策略来对环境采取行动。

MindSpore Reinforcement使用*算法配置*指定DQN算法所需的逻辑组件（Actor、Learner、Policy and Network、 Collect Environment、Eval Environment、Replayuffer）和关联的超参数。根据提供的配置，它使用不同的策略执行算法，以便用户可以专注于算法设计。

算法配置是一个Python字典，指定如何构造DQN算法的不同组件。每个组件的超参数在单独的Python字典中配置。DQN算法配置定义如下：

```python
algorithm_config = {
    'actor': {
        'number': 1,                                                        # Actor实例的数量
        'type': DQNActor,                                                   # 需要创建的Actor类
        'policies': ['init_policy', 'collect_policy', 'evaluate_policy'],   # Actor需要用到的选择动作的策略
    },
    'learner': {
        'number': 1,                                                        # Learner实例的数量
        'type': DQNLearner,                                                 # 需要创建的Learner类
        'params': learner_params,                                           # Learner需要用到的参数
        'networks': ['policy_network', 'target_network']                    # Learner中需要用到的网络
    },
    'policy_and_network': {
        'type': DQNPolicy,                                                  # 需要创建的Policy类
        'params': policy_params                                             # Policy中需要用到的参数
    },
    'collect_environment': {
        'number': 1,                                                        # Collect Environment实例的数量
        'type': GymEnvironment,                                             # 需要创建的Collect Environment类
        'params': collect_env_params                                        # Collect Environment中需要用到的参数
    },
    'eval_environment': {
        'number': 1,                                                        # 同Collect Environment
        'type': GymEnvironment,
        'params': eval_env_params
    },
    'replay_buffer': {'number': 1,                                          # ReplayBuffer实例的数量
                      'type': ReplayBuffer,                                 # 需要创建的ReplayBuffer类
                      'capacity': 100000,                                   # ReplayBuffer大小
                      'data_shape': [(4,), (1,), (1,), (4,)],               # ReplayBuffer中的数据Shape
                      'data_type': [ms.float32, ms.int32, ms.float32, ms.float32],  # ReplayBuffer中的数据Type
                      'sample_size': 64},                                   # ReplayBuffer单次采样的数据量
}
```

以上配置定义了六个顶层项，每个配置对应一个算法组件：*actor、learner、policy*、*replaybuffer*和两个*environment*。每个项对应一个类，该类必须由用户定义或者使用MIndSpore Reinforcement提供的组件，以实现DQN算法的逻辑。

顶层项具有描述组件的子项。*number*定义算法使用的组件的实例数。*type*表示必须定义的Python类的名称，用于实现组件。*params*为组件提供必要的超参数。*actor*中的*policies*定义组件使用的策略。*learner*中的*networks*列出了此组件使用的所有神经网络。在DQN示例中，只有Actor与环境交互。*replay_buffer*定义回放缓冲区的*容量、形状、样本大小和数据类型*。

对于DQN算法，我们配置了一个Actor `'number': 1`，它的Python类`'type': DQNActor`，以及三个行为策略`'policies': ['init_policy', 'collect_policy', 'evaluate_policy']`。

其他组件也以类似的方式定义。有关更多详细信息，请参阅[完整代码示例](https://gitee.com/mindspore/reinforcement/tree/r0.3/example/dqn)和[API](https://www.mindspore.cn/reinforcement/docs/zh-CN/r0.3/reinforcement.html)。

请注意，MindSpore Reinforcement使用单个*policy*类来定义算法使用的所有策略和神经网络。通过这种方式，它隐藏了策略和神经网络之间数据共享和通信的复杂性。

在train.py文件中，需要通过调用MindSpore Reinforcement的*session*来执行算法。*Session*在一台或多台群集计算机上分配资源并执行编译后的计算图。用户传入算法配置以实例化Session类：

```python
from mindspore_rl.core import Session
dqn_session = Session(dqn_algorithm_config)
```

调用Session对象上的run方法，并传入对应的参数来执行DQN算法。其中*class_type*是我们定义的Trainer类在这里是DQNTrainer（后面会介绍如何实现Trainer类），episode为需要运行的循环次数，params为在config文件中定义的trainer所需要用到的参数具体可查看完整代码中*config.py*的内容，callbacks定义了需要用到的统计方法等具体请参考API中的Callback相关内容。

```python
from src.dqn_trainer import DQNTrainer
from mindspore_rl.utils.callback import CheckpointCallback, LossCallback, EvaluateCallback
loss_cb = LossCallback()
ckpt_cb = CheckpointCallback(50, config.trainer_params['ckpt_path'])
eval_cb = EvaluateCallback(10)
cbs = [loss_cb, ckpt_cb, eval_cb]
dqn_session.run(class_type=DQNTrainer, episode=episode, params=config.trainer_params, callbacks=cbs)
```

为使用MindSpore的计算图功能，将执行模式设置为`GRAPH_MODE`。

```python
from mindspore import context
context.set_context(mode=context.GRAPH_MODE)
```

`@ms_function`注释的函数和方法将会编译到MindSpore计算图用于自动并行和加速。在本教程中，我们使用此功能来实现一个高效的`DQNTrainer`类。

### 定义DQNTrainer类

`DQNTrainer`类表示算法的流程编排，主要流程为循环迭代地与环境交互将经验内存入*ReplayBuffer*中，然后从*ReplayBuffer*获取经验并训练目标模型。它必须继承自`Trainer`类，该类是MindSpore Reinforcement API的一部分。

`Trainer`基类包含`MSRL`(MindSpore Reinforcement)对象，该对象允许算法实现与MindSpore Reinforcement交互，以实现训练逻辑。`MSRL`类根据先前定义的算法配置实例化RL算法组件。它提供了函数处理程序，这些处理程序透明地绑定到用户定义的Actor、Learner或ReplayBuffer的方法。因此，`MSRL`类让用户能够专注于算法逻辑，同时它透明地处理一个或多个worker上不同算法组件之间的对象创建、数据共享和通信。用户通过使用算法配置创建上文提到的`Session`对象来实例化`MSRL`对象。

`DQNTrainer`必须重载`train_one_episode`用于训练，`evaluate`用于评估以及`trainable_variable`用于保存断点。在本教程中，它的定义如下：

```python
class DQNTrainer(Trainer):
    def __init__(self, msrl, params):
        ...
        super(DQNTrainer, self).__init__(msrl)

    def trainable_variables(self):
        """Trainable variables for saving."""
        trainable_variables = {"policy_net": self.msrl.learner.policy_network}
        return trainable_variables

    @ms_function
    def init_training(self):
        """Initialize training"""
        state = self.msrl.collect_environment.reset()
        done = self.false
        i = self.zero_value
        while self.less(i, self.fill_value):
            done, _, new_state, action, my_reward = self.msrl.agent_act(
                trainer.INIT, state)
            self.msrl.replay_buffer_insert(
                [state, action, my_reward, new_state])
            state = new_state
            if done:
                state = self.msrl.collect_environment.reset()
                done = self.false
            i += 1
        return done

    @ms_function
    def evaluate(self):
        """Policy evaluate"""
        total_reward = self.zero_value
        eval_iter = self.zero_value
        while self.less(eval_iter, self.num_evaluate_episode):
            episode_reward = self.zero_value
            state = self.msrl.eval_environment.reset()
            done = self.false
            while not done:
                done, r, state = self.msrl.agent_act(trainer.EVAL, state)
                r = self.squeeze(r)
                episode_reward += r
            total_reward += episode_reward
            eval_iter += 1
        avg_reward = total_reward / self.num_evaluate_episode
        return avg_reward
```

用户调用`train`方法会调用Trainer基类的`train`。然后，为它指定数量的episode（iteration）训练模型，每个episode调用用户定义的`train_one_episode`方法。最后，train方法通过调用`evaluate`方法来评估策略以获得奖励值。

在训练循环的每次迭代中，调用`train_one_episode`方法来训练一个episode：

```python
@ms_function
def train_one_episode(self):
    """Train one episode"""
    if not self.inited:
        self.init_training()
        self.inited = self.true
    state = self.msrl.collect_environment.reset()
    done = self.false
    total_reward = self.zero
    steps = self.zero
    loss = self.zero
    while not done:
        done, r, new_state, action, my_reward = self.msrl.agent_act(
            trainer.COLLECT, state)
        self.msrl.replay_buffer_insert(
            [state, action, my_reward, new_state])
        state = new_state
        r = self.squeeze(r)
        loss = self.msrl.agent_learn(self.msrl.replay_buffer_sample())
        total_reward += r
        steps += 1
        if not self.mod(steps, self.update_period):
            self.msrl.learner.update()
    return loss, total_reward, steps
```

`@ms_function`注解表示此方法将被编译为MindSpore计算图用于加速。所有标量值都必须定义为张量类型，例如`self.zero_value = Tensor(0, mindspore.float32)`。

`train_one_episode`方法首先调用环境的`reset`方法，`self.msrl.collect_environment.reset()`函数来重置环境。然后，它使用`self.msrl.agent_act`函数处理程序从环境中收集经验，并通过`self.msrl.replay_buffer_insert`把经验存入到回放缓存中。在收集完经验后，使用`msrl.agent_learn`函数训练目标模型。`self.msrl.agent_learn`的输入是`self.msrl.replay_buffer_sample`返回的采样结果。

回放缓存`ReplayBuffer`由MindSpore Reinfocement提供。它定义了`insert`和`sample`方法，分别用于对经验数据进行存储和采样。详细信息，请参阅[完整的DQN代码示例](https://gitee.com/mindspore/reinforcement/tree/r0.3/example/dqn)。

### 定义DQNPolicy类

定义`DQNPolicy`类，用于实现神经网络并定义策略。

```python
class DQNPolicy:
    def __init__(self, params):
        self.policy_network = FullyConnectedNet(
            params['state_space_dim'],
            params['hidden_size'],
            params['action_space_dim'],
            params['compute_type'])
        self.target_network = FullyConnectedNet(
            params['state_space_dim'],
            params['hidden_size'],
            params['action_space_dim'],
            params['compute_type'])
```

构造函数将先前在config.py中定义的Python字典类型的超参数`policy_params`作为输入。

在定义策略网络和目标网络之前，用户必须使用MindSpore算子定义神经网络的结构。例如，它们可能是`FullyConnectedNetwork`类的对象，该类定义如下：

```python
class FullyConnectedNetwork(mindspore.nn.Cell):
     def __init__(self, input_size, hidden_size, output_size, compute_type=mstype.float32):
        super(FullyConnectedNet, self).__init__()
        self.linear1 = nn.Dense(
            input_size,
            hidden_size,
            weight_init="XavierUniform").to_float(compute_type)
        self.linear2 = nn.Dense(
            hidden_size,
            output_size,
            weight_init="XavierUniform").to_float(compute_type)
        self.relu = nn.ReLU()
```

DQN算法使用损失函数来优化神经网络的权重。此时，用户必须定义一个用于计算损失函数的神经网络。此网络被指定为`DQNLearner`的嵌套类。此外，还需要优化器来训练网络。优化器和损失函数定义如下：

```python
class DQNLearner(Learner):
    """DQN Learner"""

    class PolicyNetWithLossCell(nn.Cell):
        """DQN policy network with loss cell"""

        def __init__(self, backbone, loss_fn):
            super(DQNLearner.PolicyNetWithLossCell,
                  self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn
            self.gather = P.GatherD()

        def construct(self, x, a0, label):
            """constructor for Loss Cell"""
            out = self._backbone(x)
            out = self.gather(out, 1, a0)
            loss = self._loss_fn(out, label)
            return loss
    def __init__(self, params=None):
        super(DQNLearner, self).__init__()
        ...
        optimizer = nn.Adam(
            self.policy_network.trainable_params(),
            learning_rate=params['lr'])
        loss_fn = nn.MSELoss()
        loss_q_net = self.PolicyNetWithLossCell(self.policy_network, loss_fn)
        self.policy_network_train = nn.TrainOneStepCell(loss_q_net, optimizer)
        self.policy_network_train.set_train(mode=True)
        ...
```

DQN算法是一种*off-policy*算法，使用epsilon-贪婪策略学习。它使用不同的行为策略来对环境采取行动和收集数据。在本示例中，我们用`RandomPolicy`初始化训练，用`EpsilonGreedyPolicy`收集训练期间的经验，用`GreedyPolicy`进行评估：

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

定义一个新的Actor组件用于实现`DQNActor`，该组件继承了MindSpore Reinforcement提供的`Actor`类。然后，必须重载Actor中的方法：

```python
class DQNActor(Actor):
     ...
    def act(self, phase, params):
        if phase == 1:
            # Fill the replay buffer
            action = self.init_policy()
            new_state, reward, done = self._environment.step(action)
            action = self.reshape(action, (1,))
            my_reward = self.select(done, self.penalty, self.reward)
            return done, reward, new_state, action, my_reward
        if phase == 2:
            # Experience collection
            self.step += 1
            ts0 = self.expand_dims(params, 0)
            step_tensor = self.ones((1, 1), ms.float32) * self.step

            action = self.collect_policy(ts0, step_tensor)
            new_state, reward, done = self._environment.step(action)
            action = self.reshape(action, (1,))
            my_reward = self.select(done, self.penalty, self.reward)
            return done, reward, new_state, action, my_reward
        if phase == 3:
            # Evaluate the trained policy
            ts0 = self.expand_dims(params, 0)
            action = self.evaluate_policy(ts0)
            new_state, reward, done = self._eval_env.step(action)
            return done, reward, new_state
        self.print("Phase is incorrect")
        return 0
```

这三种方法使用不同的策略作用于指定的环境，这些策略将状态映射到操作。这些方法将张量类型的值作为输入，并从环境返回轨迹。

为了与环境交互，Actor使用`Environment`类中定义的`step(action)`方法。对于应用到指定环境的操作，此方法会做出反应并返回三元组。三元组包括应用上一个操作后的新状态、作为浮点类型获得的奖励以及用于终止episode和重置环境的布尔标志。

回放缓冲区类`ReplayBuffer`定义了一个`insert`方法，`DQNActor`对象调用该方法将经验数据存储在回放缓冲区中。

`Environment`类和`ReplayBuffer`类由MindSpore Reinforcement API提供。

`DQNActor`类的构造函数定义了环境、回放缓冲区、策略和网络。它将字典类型的参数作为输入，这些参数在算法配置中定义。下面，我们只展示环境的初始化，其他属性以类似的方式分配：

```python
class DQNActor(Actor):
     def __init__(self, params):
         self._environment = params['collect_environment']
         self._eval_env = params['eval_environment']
         ...
```

### 定义DQNLearner类

为了实现`DQNLearner`，类必须继承MindSpore Reinforcement API中的`Learner`类，并重载`learn`方法：

```python
class DQNLearner(Learner):
     ...
    def learn(self, experience):
        """Model update"""
        s0, a0, r1, s1 = experience
        next_state_values = self.target_network(s1)
        next_state_values = next_state_values.max(axis=1)
        r1 = self.reshape(r1, (-1,))

        y_true = r1 + self.gamma * next_state_values

        # Modify last step reward
        one = self.ones_like(r1)
        y_true = self.select(r1 == -one, one, y_true)
        y_true = self.expand_dims(y_true, 1)

        success = self.policy_network_train(s0, a0, y_true)
        return success
```

在这里，`learn`方法将轨迹（从回放缓冲区采样）作为输入来训练策略网络。构造函数通过从算法配置接收字典类型的配置，将网络、策略和折扣率分配给DQNLearner：

```python
class DQNLearner(Learner):
        def __init__(self, params=None):
            super(DQNLearner, self).__init__()
            self.policy_network = params['policy_network']
            self.target_network = params['target_network']
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
