mindspore_rl
========================

MindSpore强化学习框架的组件。

mindspore_rl.agent
------------------------

agent、actor、learner、trainer的组件。

.. include:: agent/mindspore_rl.agent.Actor.rst

.. include:: agent/mindspore_rl.agent.Learner.rst

.. include:: agent/mindspore_rl.agent.Trainer.rst

.. include:: agent/mindspore_rl.agent.Agent.rst

.. automodule:: mindspore_rl.agent
    :members:

mindspore_rl.core
-------------------------

用于实现 RL 算法的Helper程序组件。

.. include:: core/mindspore_rl.core.MSRL.rst

.. include:: core/mindspore_rl.core.Session.rst

.. include:: core/mindspore_rl.core.UniformReplayBuffer.rst

.. include:: core/mindspore_rl.core.PriorityReplayBuffer.rst

.. automodule:: mindspore_rl.core
    :members:

mindspore_rl.environment
-------------------------

用于实现自定义环境的组件。

.. include:: environment/mindspore_rl.environment.GymEnvironment.rst

.. include:: environment/mindspore_rl.environment.MultiEnvironmentWrapper.rst

.. include:: environment/mindspore_rl.environment.Environment.rst

.. include:: environment/mindspore_rl.environment.Space.rst

.. include:: environment/mindspore_rl.environment.MsEnvironment.rst

.. include:: environment/mindspore_rl.environment.EnvironmentProcess.rst

.. include:: environment/mindspore_rl.environment.StarCraft2Environment.rst

.. include:: environment/mindspore_rl.environment.TicTacToeEnvironment.rst

.. include:: environment/mindspore_rl.environment.DeepMindControlEnvironment.rst

.. automodule:: mindspore_rl.environment
    :members:

mindspore_rl.network
-------------------------

用于实现策略的网络组件。

.. include:: network/mindspore_rl.network.FullyConnectedLayers.rst

.. include:: network/mindspore_rl.network.GruNet.rst

.. automodule:: mindspore_rl.network
    :members:

mindspore_rl.policy
-------------------------

RL 算法中使用的策略。

.. include:: policy/mindspore_rl.policy.Policy.rst

.. include:: policy/mindspore_rl.policy.RandomPolicy.rst

.. include:: policy/mindspore_rl.policy.GreedyPolicy.rst

.. include:: policy/mindspore_rl.policy.EpsilonGreedyPolicy.rst

.. automodule:: mindspore_rl.policy
    :members:

mindspore_rl.utils
-------------------------

RL 算法中工具组件。

.. include:: utils/mindspore_rl.utils.DiscountedReturn.rst

.. include:: utils/mindspore_rl.utils.OUNoise.rst

.. include:: utils/mindspore_rl.utils.SoftUpdate.rst

.. include:: utils/mindspore_rl.utils.Callback.rst

.. include:: utils/mindspore_rl.utils.utils.rst

.. include:: utils/mindspore_rl.utils.MCTS.rst

.. include:: utils/mindspore_rl.utils.VanillaFunc.rst

.. include:: utils/mindspore_rl.utils.AlgorithmFunc.rst

.. include:: utils/mindspore_rl.utils.BatchWrite.rst

.. include:: utils/mindspore_rl.utils.TensorArray.rst

.. include:: utils/mindspore_rl.utils.TensorsQueue.rst

.. automodule:: mindspore_rl.utils
    :members:
