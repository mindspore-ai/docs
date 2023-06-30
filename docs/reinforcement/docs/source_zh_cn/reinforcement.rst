mindspore_rl
========================

MindSpore强化学习框架的组件。

mindspore_rl.agent
------------------------

agent、actor、learner、trainer的组件。

.. include:: agent/mindspore_rl.agent.Actor.txt

.. include:: agent/mindspore_rl.agent.Learner.txt

.. include:: agent/mindspore_rl.agent.Trainer.txt

.. include:: agent/mindspore_rl.agent.Agent.txt

.. automodule:: mindspore_rl.agent
    :members:

mindspore_rl.core
-------------------------

用于实现 RL 算法的Helper程序组件。

.. include:: core/mindspore_rl.core.MSRL.txt

.. include:: core/mindspore_rl.core.Session.txt

.. include:: core/mindspore_rl.core.UniformReplayBuffer.txt

.. include:: core/mindspore_rl.core.PriorityReplayBuffer.txt

.. automodule:: mindspore_rl.core
    :members:

mindspore_rl.environment
-------------------------

用于实现自定义环境的组件。

.. include:: environment/mindspore_rl.environment.GymEnvironment.txt

.. include:: environment/mindspore_rl.environment.MultiEnvironmentWrapper.txt

.. include:: environment/mindspore_rl.environment.Environment.txt

.. include:: environment/mindspore_rl.environment.Space.txt

.. include:: environment/mindspore_rl.environment.MsEnvironment.txt

.. include:: environment/mindspore_rl.environment.EnvironmentProcess.txt

.. include:: environment/mindspore_rl.environment.StarCraft2Environment.txt

.. include:: environment/mindspore_rl.environment.TicTacToeEnvironment.txt

.. automodule:: mindspore_rl.environment
    :members:

mindspore_rl.network
-------------------------

用于实现策略的网络组件。

.. include:: network/mindspore_rl.network.FullyConnectedLayers.txt

.. include:: network/mindspore_rl.network.GruNet.txt

.. automodule:: mindspore_rl.network
    :members:

mindspore_rl.policy
-------------------------

RL 算法中使用的策略。

.. include:: policy/mindspore_rl.policy.Policy.txt

.. include:: policy/mindspore_rl.policy.RandomPolicy.txt

.. include:: policy/mindspore_rl.policy.GreedyPolicy.txt

.. include:: policy/mindspore_rl.policy.EpsilonGreedyPolicy.txt

.. automodule:: mindspore_rl.policy
    :members:
