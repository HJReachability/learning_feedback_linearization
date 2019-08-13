u"""

Some simple logging functionality, inspired by rllab's logging.

Logs to a tab-separated-values file (path/to/output_directory/progress.txt)

"""
from __future__ import with_statement
from __future__ import absolute_import
import json
import joblib
import shutil
import numpy as np
import tensorflow as tf
import os.path as osp, time, atexit, os
from spinup2.utils.mpi_tools import proc_id, mpi_statistics_scalar
from spinup2.utils.serialization_utils import convert_json
from io import open
from itertools import imap

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    u"""
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(unicode(num))
    if bold: attr.append(u'1')
    return u'\x1b[%sm%s\x1b[0m' % (u';'.join(attr), string)

def restore_tf_graph(sess, fpath):
    u"""
    Loads graphs saved by Logger.

    Will output a dictionary whose keys and values are from the 'inputs' 
    and 'outputs' dict you specified with logger.setup_tf_saver().

    Args:
        sess: A Tensorflow session.
        fpath: Filepath to save directory.

    Returns:
        A dictionary mapping from keys to tensors in the computation graph
        loaded from ``fpath``. 
    """
    tf.saved_model.loader.load(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                fpath
            )
    model_info = joblib.load(osp.join(fpath, u'model_info.pkl'))
    graph = tf.get_default_graph()
    model = dict()
    model.update(dict((k, graph.get_tensor_by_name(v)) for k,v in model_info[u'inputs'].items()))
    model.update(dict((k, graph.get_tensor_by_name(v)) for k,v in model_info[u'outputs'].items()))
    return model

class Logger(object):
    u"""
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    """

    def __init__(self, output_dir=None, output_fname=u'progress.txt', exp_name=None):
        u"""
        Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to. If 
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.

            output_fname (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        if proc_id()==0:
            self.output_dir = output_dir or u"/tmp/experiments/%i"%int(time.time())
            if osp.exists(self.output_dir):
                print u"Warning: Log dir %s already exists! Storing info there anyway."%self.output_dir
            else:
                os.makedirs(self.output_dir)
            self.output_file = open(osp.join(self.output_dir, output_fname), u'w')
            atexit.register(self.output_file.close)
            print colorize(u"Logging data to %s"%self.output_file.name, u'green', bold=True)
        else:
            self.output_dir = None
            self.output_file = None
        self.first_row=True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log(self, msg, color=u'green'):
        u"""Print a colorized message to stdout."""
        if proc_id()==0:
            print colorize(msg, color, bold=True)

    def log_tabular(self, key, val):
        u"""
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, u"Trying to introduce a new key %s that you didn't include in the first iteration"%key
        assert key not in self.log_current_row, u"You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
        self.log_current_row[key] = val

    def save_config(self, config):
        u"""
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible). 

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json[u'exp_name'] = self.exp_name
        if proc_id()==0:
            output = json.dumps(config_json, separators=(u',',u':\t'), indent=4, sort_keys=True)
            print colorize(u'Saving config:\n', color=u'cyan', bold=True)
            print output
            with open(osp.join(self.output_dir, u"config.json"), u'w') as out:
                out.write(output)

    def save_state(self, state_dict, itr=None):
        u"""
        Saves the state of an experiment.

        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function. This function
        will save whatever is in ``state_dict``---usually just a copy of the
        environment---and the most recent parameters for the model you 
        previously set up saving for with ``setup_tf_saver``. 

        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent 
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.

        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.

            itr: An int, or None. Current iteration of training.
        """
        if proc_id()==0:
            fname = u'vars.pkl' if itr is None else u'vars%d.pkl'%itr
            try:
                joblib.dump(state_dict, osp.join(self.output_dir, fname))
            except:
                self.log(u'Warning: could not pickle state_dict.', color=u'red')
            if hasattr(self, u'tf_saver_elements'):
                self._tf_simple_save(itr)

    def setup_tf_saver(self, sess, inputs, outputs):
        u"""
        Set up easy model saving for tensorflow.

        Call once, after defining your computation graph but before training.

        Args:
            sess: The Tensorflow session in which you train your computation
                graph.

            inputs (dict): A dictionary that maps from keys of your choice
                to the tensorflow placeholders that serve as inputs to the 
                computation graph. Make sure that *all* of the placeholders
                needed for your outputs are included!

            outputs (dict): A dictionary that maps from keys of your choice
                to the outputs from your computation graph.
        """
        self.tf_saver_elements = dict(session=sess, inputs=inputs, outputs=outputs)
        self.tf_saver_info = {u'inputs': dict((k, v.name) for k,v in inputs.items()),
                              u'outputs': dict((k, v.name) for k,v in outputs.items())}

    def _tf_simple_save(self, itr=None):
        u"""
        Uses simple_save to save a trained model, plus info to make it easy
        to associated tensors to variables after restore. 
        """
        if proc_id()==0:
            assert hasattr(self, u'tf_saver_elements'), \
                u"First have to setup saving with self.setup_tf_saver"
            fpath = u'simple_save' + (u'%d'%itr if itr is not None else u'')
            fpath = osp.join(self.output_dir, fpath)
            if osp.exists(fpath):
                # simple_save refuses to be useful if fpath already exists,
                # so just delete fpath if it's there.
                shutil.rmtree(fpath)
            tf.saved_model.simple_save(export_dir=fpath, **self.tf_saver_elements)
            joblib.dump(self.tf_saver_info, osp.join(fpath, u'model_info.pkl'))
    
    def dump_tabular(self):
        u"""
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        if proc_id()==0:
            vals = []
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15,max(key_lens))
            keystr = u'%'+u'%d'%max_key_len
            fmt = u"| " + keystr + u"s | %15s |"
            n_slashes = 22 + max_key_len
            print u"-"*n_slashes
            for key in self.log_headers:
                val = self.log_current_row.get(key, u"")
                valstr = u"%8.3g"%val if hasattr(val, u"__float__") else val
                print fmt%(key, valstr)
                vals.append(val)
            print u"-"*n_slashes
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write(u"\t".join(self.log_headers)+u"\n")
                self.output_file.write(u"\t".join(imap(unicode,vals))+u"\n")
                self.output_file.flush()
        self.log_current_row.clear()
        self.first_row=False

class EpochLogger(Logger):
    u"""
    A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to 
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use 

    .. code-block:: python

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you 
    would use 

    .. code-block:: python

        epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super(EpochLogger, self).__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        u"""
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical 
        values.
        """
        for k,v in kwargs.items():
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        u"""
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with 
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the 
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super(EpochLogger, self).log_tabular(key,val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
            stats = mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super(EpochLogger, self).log_tabular(key if average_only else u'Average' + key, stats[0])
            if not(average_only):
                super(EpochLogger, self).log_tabular(u'Std'+key, stats[1])
            if with_min_and_max:
                super(EpochLogger, self).log_tabular(u'Max'+key, stats[3])
                super(EpochLogger, self).log_tabular(u'Min'+key, stats[2])
        self.epoch_dict[key] = []

    def get_stats(self, key):
        u"""
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
        return mpi_statistics_scalar(vals)