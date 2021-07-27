"""model accu
Construct model accu
"""
import math
from mindspore.train.callback import RunContext
from mindspore import context
from mindspore.context import ParallelMode
from mindspore import Model, connect_network_with_dataset
from mindspore import pytype_to_dtype
from mindspore._c_expression import init_exec_dataset
from mindspore.train.train_thor.dataset_helper import DatasetHelper
from mindspore.parallel._utils import _need_to_full, _to_full_tensor


def _convert_type(types):
    """
    Convert from numpy type to tensor type.

    Args:
        types (list): Numpy type list of element in dataset.

    Returns:
        list, list of element in dataset.
    """
    ms_types = []
    for np_type in types:
        ms_type = pytype_to_dtype(np_type)
        ms_types.append(ms_type)
    return ms_types


def _get_types_and_shapes(dataset):
    """Get dataset types and shapes."""
    dataset_types = _convert_type(dataset.output_types())
    dataset_shapes = dataset.output_shapes()
    return dataset_types, dataset_shapes


def _exec_datagraph(exec_dataset, dataset_size, phase='dataset'):
    """Initialize and execute the dataset graph."""
    batch_size = exec_dataset.get_batch_size()
    input_indexs = exec_dataset.input_indexs

    # transform data format
    dataset_types, dataset_shapes = _get_types_and_shapes(exec_dataset)
    init_exec_dataset(exec_dataset.__transfer_dataset__.queue_name,
                      dataset_size,
                      batch_size,
                      dataset_types,
                      dataset_shapes,
                      input_indexs,
                      phase=phase,
                      need_run=False)


class Model_ACCU(Model):
    """"Construct Model_ACCU"""
    def __init__(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None,
                 eval_indexes=None, amp_level="O0", **kwargs):
        super(Model_ACCU, self).__init__(network, loss_fn, optimizer, metrics, eval_network,
                                         eval_indexes, amp_level, **kwargs)
        self._frequency = context.get_auto_parallel_context("grad_accumulation_step")
        # used to stop training for early stop, such as stopAtTIme or stopATStep
        self.should_stop = False
        self.switch_branch_one = True
        self.index_second_order = 0
        self.train_network_init_flag = True
        self.has_do_dataset_init = False
        self._train_network = self._build_train_network()

    def _exec_preprocess(self, network, is_train, phase, dataset, dataset_sink_mode, sink_size=-1,
                         epoch_num=1, iter_update_order=1):
        """Initializes dataset."""
        if dataset_sink_mode and not is_train:
            dataset.__loop_size__ = 1
        dataset_helper = DatasetHelper(dataset, dataset_sink_mode, sink_size, epoch_num, iter_update_order)

        if dataset_sink_mode and context.get_context("device_target") != "GPU":
            network = connect_network_with_dataset(network, dataset_helper)
        network.set_train(is_train)
        network.phase = phase

        if self._parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            network.set_auto_parallel()

        return dataset_helper, network

    def _train_gpu_sink_step(self, cb_params, inputs, list_callback, iter_accu_order, run_context):
        """train gpu sink step"""
        if self.switch_branch_one:
            cb_params.cur_step_num += 1
            if self.train_network_init_flag:
                self._train_network.add_flags_recursive(accumulation=True)
            self._train_network.phase = 'train0'
            outputs = self._train_network(*inputs)
            cb_params.net_outputs = outputs
            self.index_second_order += 1
            if self.index_second_order == iter_accu_order:
                self.index_second_order = 0
                self.switch_branch_one = not self.switch_branch_one
                list_callback.step_end(run_context)
        else:
            cb_params.cur_step_num += 1
            if self.train_network_init_flag:
                self._train_network.add_flags_recursive(accumulation=False)
                self.train_network_init_flag = False
            self._train_network.phase = 'train1'
            self.switch_branch_one = not self.switch_branch_one
            outputs = self._train_network(*inputs)
            cb_params.net_outputs = outputs
            list_callback.step_end(run_context)

    def _train_ascend_sink_step(self, cb_params, train_dataset, iter_accu_order, inputs, list_callback, run_context):
        """train ascend sink step"""
        if self.switch_branch_one:
            cb_params.cur_step_num += iter_accu_order
            if self.train_network_init_flag:
                self._train_network.add_flags_recursive(accumulation=True)
            self._train_network.phase = 'train0'
        else:
            cb_params.cur_step_num += 1
            if self.train_network_init_flag:
                self._train_network.add_flags_recursive(accumulation=False)
                self.train_network_init_flag = False
            self._train_network.phase = 'train1'
            if not self.has_do_dataset_init:
                _exec_datagraph(train_dataset, 1, phase='train1_dataset')
                self.has_do_dataset_init = True
        self.switch_branch_one = not self.switch_branch_one
        outputs = self._train_network(*inputs)
        cb_params.net_outputs = outputs
        list_callback.step_end(run_context)

    def _train_dataset_sink_process(self, epoch, train_dataset, list_callback=None, cb_params=None, sink_size=-1):
        """
        Training process. The data would be passed to network through dataset channel.

        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            list_callback (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.
            sink_size (int): Control the amount of data in each sink. Default: -1.
        """
        if sink_size == -1:
            epoch_num = epoch
        else:
            epoch_num = math.ceil(epoch * sink_size / train_dataset.get_dataset_size())

        iter_update_order = 1
        iter_accu_order = self._frequency - 1
        if context.get_context("device_target") == "GPU":
            train_dataset.__loop_size__ = 1
        else:
            train_dataset.__loop_size__ = iter_accu_order
        dataset_helper, train_network = self._exec_preprocess(self._train_network,
                                                              is_train=True,
                                                              phase='train',
                                                              dataset=train_dataset,
                                                              dataset_sink_mode=True,
                                                              sink_size=sink_size,
                                                              epoch_num=epoch_num,
                                                              iter_update_order=iter_update_order)

        self._train_network = train_network
        cb_params.train_network = self._train_network
        cb_params.cur_step_num = 0

        run_context = RunContext(cb_params)
        list_callback.begin(run_context)

        for i in range(epoch):
            cb_params.cur_epoch_num = i + 1
            list_callback.epoch_begin(run_context)
            # for data sink dataset_helper only iter once, other wise iter epoch_size times.
            for inputs in dataset_helper:
                if _need_to_full() and context.get_context("device_target") == "GPU":
                    inputs = _to_full_tensor(inputs, self._device_number, self._global_rank)
                list_callback.step_begin(run_context)
                if context.get_context("device_target") == "GPU":
                    self._train_gpu_sink_step(cb_params, inputs, list_callback, iter_accu_order, run_context)
                else:
                    self._train_ascend_sink_step(cb_params, train_dataset, iter_accu_order, inputs, list_callback,
                                                 run_context)
            list_callback.epoch_end(run_context)
            self.should_stop = self.should_stop or run_context.get_stop_requested()
            if self.should_stop:
                break
        dataset_helper.stop_send()

        list_callback.end(run_context)
