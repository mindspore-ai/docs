Debugger
==================================

MindSpore Debugger is a debugging tool for training in Graph Mode. It can be applied to visualize and analyze the intermediate computation results of the computational graph.

In `Graph Mode` training, the computation results of intermediate nodes in the computational graph can not be acquired conveniently, which makes it difficult for users to do the debugging. By applying MindSpore Debugger, users can:

- Visualize the computational graph on the UI and analyze the output of the graph node.
- Set watchpoints to monitor training exceptions (for example, tensor overflow) and trace error causes.
- Visualize and analyze the change of parameters, such as weights.
- Visualize the nodes and code mapping relationship.

Debugger can be used in two modes: online mode and offline mode. Online Debugger can visualize the training during the training process, which is easy to use, and can visualize the results of every step in the training. It is available for small and medium sized, too large network may lead to out of memory. The offline debugger can connect to offline dump data for visualized analysis. It solves the out of memory problem in online debugger when the training network is too large. Dump data of specified steps need to be prepared in advance，and  then the data of these steps can be analysed.

.. toctree::
   :maxdepth: 1

   debugger_online
   debugger_offline
