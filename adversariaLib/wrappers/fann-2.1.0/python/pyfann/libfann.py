# This file was automatically generated by SWIG (http://www.swig.org).
# Version 2.0.11
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2,6,0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_libfann', [dirname(__file__)])
        except ImportError:
            import _libfann
            return _libfann
        if fp is not None:
            try:
                _mod = imp.load_module('_libfann', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _libfann = swig_import_helper()
    del swig_import_helper
else:
    import _libfann
del version_info
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError(name)

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0


class SwigPyIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SwigPyIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SwigPyIterator, name)
    def __init__(self, *args, **kwargs): raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _libfann.delete_SwigPyIterator
    __del__ = lambda self : None;
    def value(self): return _libfann.SwigPyIterator_value(self)
    def incr(self, n=1): return _libfann.SwigPyIterator_incr(self, n)
    def decr(self, n=1): return _libfann.SwigPyIterator_decr(self, n)
    def distance(self, *args): return _libfann.SwigPyIterator_distance(self, *args)
    def equal(self, *args): return _libfann.SwigPyIterator_equal(self, *args)
    def copy(self): return _libfann.SwigPyIterator_copy(self)
    def next(self): return _libfann.SwigPyIterator_next(self)
    def __next__(self): return _libfann.SwigPyIterator___next__(self)
    def previous(self): return _libfann.SwigPyIterator_previous(self)
    def advance(self, *args): return _libfann.SwigPyIterator_advance(self, *args)
    def __eq__(self, *args): return _libfann.SwigPyIterator___eq__(self, *args)
    def __ne__(self, *args): return _libfann.SwigPyIterator___ne__(self, *args)
    def __iadd__(self, *args): return _libfann.SwigPyIterator___iadd__(self, *args)
    def __isub__(self, *args): return _libfann.SwigPyIterator___isub__(self, *args)
    def __add__(self, *args): return _libfann.SwigPyIterator___add__(self, *args)
    def __sub__(self, *args): return _libfann.SwigPyIterator___sub__(self, *args)
    def __iter__(self): return self
SwigPyIterator_swigregister = _libfann.SwigPyIterator_swigregister
SwigPyIterator_swigregister(SwigPyIterator)

ERRORFUNC_LINEAR = _libfann.ERRORFUNC_LINEAR
ERRORFUNC_TANH = _libfann.ERRORFUNC_TANH
STOPFUNC_MSE = _libfann.STOPFUNC_MSE
STOPFUNC_BIT = _libfann.STOPFUNC_BIT
TRAIN_INCREMENTAL = _libfann.TRAIN_INCREMENTAL
TRAIN_BATCH = _libfann.TRAIN_BATCH
TRAIN_RPROP = _libfann.TRAIN_RPROP
TRAIN_QUICKPROP = _libfann.TRAIN_QUICKPROP
LINEAR = _libfann.LINEAR
THRESHOLD = _libfann.THRESHOLD
THRESHOLD_SYMMETRIC = _libfann.THRESHOLD_SYMMETRIC
SIGMOID = _libfann.SIGMOID
SIGMOID_STEPWISE = _libfann.SIGMOID_STEPWISE
SIGMOID_SYMMETRIC = _libfann.SIGMOID_SYMMETRIC
SIGMOID_SYMMETRIC_STEPWISE = _libfann.SIGMOID_SYMMETRIC_STEPWISE
GAUSSIAN = _libfann.GAUSSIAN
GAUSSIAN_SYMMETRIC = _libfann.GAUSSIAN_SYMMETRIC
GAUSSIAN_STEPWISE = _libfann.GAUSSIAN_STEPWISE
ELLIOT = _libfann.ELLIOT
ELLIOT_SYMMETRIC = _libfann.ELLIOT_SYMMETRIC
LINEAR_PIECE = _libfann.LINEAR_PIECE
LINEAR_PIECE_SYMMETRIC = _libfann.LINEAR_PIECE_SYMMETRIC
SIN_SYMMETRIC = _libfann.SIN_SYMMETRIC
COS_SYMMETRIC = _libfann.COS_SYMMETRIC
LAYER = _libfann.LAYER
SHORTCUT = _libfann.SHORTCUT
class training_data_parent(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, training_data_parent, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, training_data_parent, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _libfann.new_training_data_parent(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _libfann.delete_training_data_parent
    __del__ = lambda self : None;
    def destroy_train(self): return _libfann.training_data_parent_destroy_train(self)
    def read_train_from_file(self, *args): return _libfann.training_data_parent_read_train_from_file(self, *args)
    def save_train(self, *args): return _libfann.training_data_parent_save_train(self, *args)
    def save_train_to_fixed(self, *args): return _libfann.training_data_parent_save_train_to_fixed(self, *args)
    def shuffle_train_data(self): return _libfann.training_data_parent_shuffle_train_data(self)
    def merge_train_data(self, *args): return _libfann.training_data_parent_merge_train_data(self, *args)
    def length_train_data(self): return _libfann.training_data_parent_length_train_data(self)
    def num_input_train_data(self): return _libfann.training_data_parent_num_input_train_data(self)
    def num_output_train_data(self): return _libfann.training_data_parent_num_output_train_data(self)
    def get_input(self): return _libfann.training_data_parent_get_input(self)
    def get_output(self): return _libfann.training_data_parent_get_output(self)
    def set_train_data(self, *args): return _libfann.training_data_parent_set_train_data(self, *args)
    def create_train_from_callback(self, *args): return _libfann.training_data_parent_create_train_from_callback(self, *args)
    def scale_input_train_data(self, *args): return _libfann.training_data_parent_scale_input_train_data(self, *args)
    def scale_output_train_data(self, *args): return _libfann.training_data_parent_scale_output_train_data(self, *args)
    def scale_train_data(self, *args): return _libfann.training_data_parent_scale_train_data(self, *args)
    def subset_train_data(self, *args): return _libfann.training_data_parent_subset_train_data(self, *args)
training_data_parent_swigregister = _libfann.training_data_parent_swigregister
training_data_parent_swigregister(training_data_parent)

class neural_net_parent(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, neural_net_parent, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, neural_net_parent, name)
    __repr__ = _swig_repr
    def __init__(self): 
        this = _libfann.new_neural_net_parent()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _libfann.delete_neural_net_parent
    __del__ = lambda self : None;
    def destroy(self): return _libfann.neural_net_parent_destroy(self)
    def create_standard(self, *args): return _libfann.neural_net_parent_create_standard(self, *args)
    def create_standard_array(self, *args): return _libfann.neural_net_parent_create_standard_array(self, *args)
    def create_sparse(self, *args): return _libfann.neural_net_parent_create_sparse(self, *args)
    def create_sparse_array(self, *args): return _libfann.neural_net_parent_create_sparse_array(self, *args)
    def create_shortcut(self, *args): return _libfann.neural_net_parent_create_shortcut(self, *args)
    def create_shortcut_array(self, *args): return _libfann.neural_net_parent_create_shortcut_array(self, *args)
    def run(self, *args): return _libfann.neural_net_parent_run(self, *args)
    def randomize_weights(self, *args): return _libfann.neural_net_parent_randomize_weights(self, *args)
    def init_weights(self, *args): return _libfann.neural_net_parent_init_weights(self, *args)
    def print_connections(self): return _libfann.neural_net_parent_print_connections(self)
    def create_from_file(self, *args): return _libfann.neural_net_parent_create_from_file(self, *args)
    def save(self, *args): return _libfann.neural_net_parent_save(self, *args)
    def save_to_fixed(self, *args): return _libfann.neural_net_parent_save_to_fixed(self, *args)
    def train(self, *args): return _libfann.neural_net_parent_train(self, *args)
    def train_epoch(self, *args): return _libfann.neural_net_parent_train_epoch(self, *args)
    def train_on_data(self, *args): return _libfann.neural_net_parent_train_on_data(self, *args)
    def train_on_file(self, *args): return _libfann.neural_net_parent_train_on_file(self, *args)
    def test(self, *args): return _libfann.neural_net_parent_test(self, *args)
    def test_data(self, *args): return _libfann.neural_net_parent_test_data(self, *args)
    def get_MSE(self): return _libfann.neural_net_parent_get_MSE(self)
    def reset_MSE(self): return _libfann.neural_net_parent_reset_MSE(self)
    def set_callback(self, *args): return _libfann.neural_net_parent_set_callback(self, *args)
    def print_parameters(self): return _libfann.neural_net_parent_print_parameters(self)
    def get_training_algorithm(self): return _libfann.neural_net_parent_get_training_algorithm(self)
    def set_training_algorithm(self, *args): return _libfann.neural_net_parent_set_training_algorithm(self, *args)
    def get_learning_rate(self): return _libfann.neural_net_parent_get_learning_rate(self)
    def set_learning_rate(self, *args): return _libfann.neural_net_parent_set_learning_rate(self, *args)
    def get_activation_function(self, *args): return _libfann.neural_net_parent_get_activation_function(self, *args)
    def set_activation_function(self, *args): return _libfann.neural_net_parent_set_activation_function(self, *args)
    def set_activation_function_layer(self, *args): return _libfann.neural_net_parent_set_activation_function_layer(self, *args)
    def set_activation_function_hidden(self, *args): return _libfann.neural_net_parent_set_activation_function_hidden(self, *args)
    def set_activation_function_output(self, *args): return _libfann.neural_net_parent_set_activation_function_output(self, *args)
    def get_activation_steepness(self, *args): return _libfann.neural_net_parent_get_activation_steepness(self, *args)
    def set_activation_steepness(self, *args): return _libfann.neural_net_parent_set_activation_steepness(self, *args)
    def set_activation_steepness_layer(self, *args): return _libfann.neural_net_parent_set_activation_steepness_layer(self, *args)
    def set_activation_steepness_hidden(self, *args): return _libfann.neural_net_parent_set_activation_steepness_hidden(self, *args)
    def set_activation_steepness_output(self, *args): return _libfann.neural_net_parent_set_activation_steepness_output(self, *args)
    def get_train_error_function(self): return _libfann.neural_net_parent_get_train_error_function(self)
    def set_train_error_function(self, *args): return _libfann.neural_net_parent_set_train_error_function(self, *args)
    def get_quickprop_decay(self): return _libfann.neural_net_parent_get_quickprop_decay(self)
    def set_quickprop_decay(self, *args): return _libfann.neural_net_parent_set_quickprop_decay(self, *args)
    def get_quickprop_mu(self): return _libfann.neural_net_parent_get_quickprop_mu(self)
    def set_quickprop_mu(self, *args): return _libfann.neural_net_parent_set_quickprop_mu(self, *args)
    def get_rprop_increase_factor(self): return _libfann.neural_net_parent_get_rprop_increase_factor(self)
    def set_rprop_increase_factor(self, *args): return _libfann.neural_net_parent_set_rprop_increase_factor(self, *args)
    def get_rprop_decrease_factor(self): return _libfann.neural_net_parent_get_rprop_decrease_factor(self)
    def set_rprop_decrease_factor(self, *args): return _libfann.neural_net_parent_set_rprop_decrease_factor(self, *args)
    def get_rprop_delta_min(self): return _libfann.neural_net_parent_get_rprop_delta_min(self)
    def set_rprop_delta_min(self, *args): return _libfann.neural_net_parent_set_rprop_delta_min(self, *args)
    def get_rprop_delta_max(self): return _libfann.neural_net_parent_get_rprop_delta_max(self)
    def set_rprop_delta_max(self, *args): return _libfann.neural_net_parent_set_rprop_delta_max(self, *args)
    def get_num_input(self): return _libfann.neural_net_parent_get_num_input(self)
    def get_num_output(self): return _libfann.neural_net_parent_get_num_output(self)
    def get_total_neurons(self): return _libfann.neural_net_parent_get_total_neurons(self)
    def get_total_connections(self): return _libfann.neural_net_parent_get_total_connections(self)
    def get_network_type(self): return _libfann.neural_net_parent_get_network_type(self)
    def get_connection_rate(self): return _libfann.neural_net_parent_get_connection_rate(self)
    def get_num_layers(self): return _libfann.neural_net_parent_get_num_layers(self)
    def get_layer_array(self, *args): return _libfann.neural_net_parent_get_layer_array(self, *args)
    def get_bias_array(self, *args): return _libfann.neural_net_parent_get_bias_array(self, *args)
    def get_connection_array(self, *args): return _libfann.neural_net_parent_get_connection_array(self, *args)
    def set_weight_array(self, *args): return _libfann.neural_net_parent_set_weight_array(self, *args)
    def set_weight(self, *args): return _libfann.neural_net_parent_set_weight(self, *args)
    def get_learning_momentum(self): return _libfann.neural_net_parent_get_learning_momentum(self)
    def set_learning_momentum(self, *args): return _libfann.neural_net_parent_set_learning_momentum(self, *args)
    def get_train_stop_function(self): return _libfann.neural_net_parent_get_train_stop_function(self)
    def set_train_stop_function(self, *args): return _libfann.neural_net_parent_set_train_stop_function(self, *args)
    def get_bit_fail_limit(self): return _libfann.neural_net_parent_get_bit_fail_limit(self)
    def set_bit_fail_limit(self, *args): return _libfann.neural_net_parent_set_bit_fail_limit(self, *args)
    def get_bit_fail(self): return _libfann.neural_net_parent_get_bit_fail(self)
    def cascadetrain_on_data(self, *args): return _libfann.neural_net_parent_cascadetrain_on_data(self, *args)
    def cascadetrain_on_file(self, *args): return _libfann.neural_net_parent_cascadetrain_on_file(self, *args)
    def get_cascade_output_change_fraction(self): return _libfann.neural_net_parent_get_cascade_output_change_fraction(self)
    def set_cascade_output_change_fraction(self, *args): return _libfann.neural_net_parent_set_cascade_output_change_fraction(self, *args)
    def get_cascade_output_stagnation_epochs(self): return _libfann.neural_net_parent_get_cascade_output_stagnation_epochs(self)
    def set_cascade_output_stagnation_epochs(self, *args): return _libfann.neural_net_parent_set_cascade_output_stagnation_epochs(self, *args)
    def get_cascade_candidate_change_fraction(self): return _libfann.neural_net_parent_get_cascade_candidate_change_fraction(self)
    def set_cascade_candidate_change_fraction(self, *args): return _libfann.neural_net_parent_set_cascade_candidate_change_fraction(self, *args)
    def get_cascade_candidate_stagnation_epochs(self): return _libfann.neural_net_parent_get_cascade_candidate_stagnation_epochs(self)
    def set_cascade_candidate_stagnation_epochs(self, *args): return _libfann.neural_net_parent_set_cascade_candidate_stagnation_epochs(self, *args)
    def get_cascade_weight_multiplier(self): return _libfann.neural_net_parent_get_cascade_weight_multiplier(self)
    def set_cascade_weight_multiplier(self, *args): return _libfann.neural_net_parent_set_cascade_weight_multiplier(self, *args)
    def get_cascade_candidate_limit(self): return _libfann.neural_net_parent_get_cascade_candidate_limit(self)
    def set_cascade_candidate_limit(self, *args): return _libfann.neural_net_parent_set_cascade_candidate_limit(self, *args)
    def get_cascade_max_out_epochs(self): return _libfann.neural_net_parent_get_cascade_max_out_epochs(self)
    def set_cascade_max_out_epochs(self, *args): return _libfann.neural_net_parent_set_cascade_max_out_epochs(self, *args)
    def get_cascade_max_cand_epochs(self): return _libfann.neural_net_parent_get_cascade_max_cand_epochs(self)
    def set_cascade_max_cand_epochs(self, *args): return _libfann.neural_net_parent_set_cascade_max_cand_epochs(self, *args)
    def get_cascade_num_candidates(self): return _libfann.neural_net_parent_get_cascade_num_candidates(self)
    def get_cascade_activation_functions_count(self): return _libfann.neural_net_parent_get_cascade_activation_functions_count(self)
    def get_cascade_activation_functions(self): return _libfann.neural_net_parent_get_cascade_activation_functions(self)
    def set_cascade_activation_functions(self, *args): return _libfann.neural_net_parent_set_cascade_activation_functions(self, *args)
    def get_cascade_activation_steepnesses_count(self): return _libfann.neural_net_parent_get_cascade_activation_steepnesses_count(self)
    def get_cascade_activation_steepnesses(self): return _libfann.neural_net_parent_get_cascade_activation_steepnesses(self)
    def set_cascade_activation_steepnesses(self, *args): return _libfann.neural_net_parent_set_cascade_activation_steepnesses(self, *args)
    def get_cascade_num_candidate_groups(self): return _libfann.neural_net_parent_get_cascade_num_candidate_groups(self)
    def set_cascade_num_candidate_groups(self, *args): return _libfann.neural_net_parent_set_cascade_num_candidate_groups(self, *args)
    def scale_train(self, *args): return _libfann.neural_net_parent_scale_train(self, *args)
    def descale_train(self, *args): return _libfann.neural_net_parent_descale_train(self, *args)
    def set_input_scaling_params(self, *args): return _libfann.neural_net_parent_set_input_scaling_params(self, *args)
    def set_output_scaling_params(self, *args): return _libfann.neural_net_parent_set_output_scaling_params(self, *args)
    def set_scaling_params(self, *args): return _libfann.neural_net_parent_set_scaling_params(self, *args)
    def clear_scaling_params(self): return _libfann.neural_net_parent_clear_scaling_params(self)
    def scale_input(self, *args): return _libfann.neural_net_parent_scale_input(self, *args)
    def scale_output(self, *args): return _libfann.neural_net_parent_scale_output(self, *args)
    def descale_input(self, *args): return _libfann.neural_net_parent_descale_input(self, *args)
    def descale_output(self, *args): return _libfann.neural_net_parent_descale_output(self, *args)
    def set_error_log(self, *args): return _libfann.neural_net_parent_set_error_log(self, *args)
    def get_errno(self): return _libfann.neural_net_parent_get_errno(self)
    def reset_errno(self): return _libfann.neural_net_parent_reset_errno(self)
    def reset_errstr(self): return _libfann.neural_net_parent_reset_errstr(self)
    def get_errstr(self): return _libfann.neural_net_parent_get_errstr(self)
    def print_error(self): return _libfann.neural_net_parent_print_error(self)
neural_net_parent_swigregister = _libfann.neural_net_parent_swigregister
neural_net_parent_swigregister(neural_net_parent)

class training_data(training_data_parent):
    __swig_setmethods__ = {}
    for _s in [training_data_parent]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, training_data, name, value)
    __swig_getmethods__ = {}
    for _s in [training_data_parent]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, training_data, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _libfann.new_training_data(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _libfann.delete_training_data
    __del__ = lambda self : None;
    def get_input(self): return _libfann.training_data_get_input(self)
    def get_output(self): return _libfann.training_data_get_output(self)
    def set_train_data(self, *args): return _libfann.training_data_set_train_data(self, *args)
training_data_swigregister = _libfann.training_data_swigregister
training_data_swigregister(training_data)

class neural_net(neural_net_parent):
    __swig_setmethods__ = {}
    for _s in [neural_net_parent]: __swig_setmethods__.update(getattr(_s,'__swig_setmethods__',{}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, neural_net, name, value)
    __swig_getmethods__ = {}
    for _s in [neural_net_parent]: __swig_getmethods__.update(getattr(_s,'__swig_getmethods__',{}))
    __getattr__ = lambda self, name: _swig_getattr(self, neural_net, name)
    __repr__ = _swig_repr
    def __init__(self): 
        this = _libfann.new_neural_net()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _libfann.delete_neural_net
    __del__ = lambda self : None;
    def create_standard_array(self, *args): return _libfann.neural_net_create_standard_array(self, *args)
    def create_sparse_array(self, *args): return _libfann.neural_net_create_sparse_array(self, *args)
    def create_shortcut_array(self, *args): return _libfann.neural_net_create_shortcut_array(self, *args)
    def run(self, *args): return _libfann.neural_net_run(self, *args)
    def train(self, *args): return _libfann.neural_net_train(self, *args)
    def test(self, *args): return _libfann.neural_net_test(self, *args)
    def get_layer_array(self, *args): return _libfann.neural_net_get_layer_array(self, *args)
    def get_bias_array(self, *args): return _libfann.neural_net_get_bias_array(self, *args)
    def get_connection_array(self, *args): return _libfann.neural_net_get_connection_array(self, *args)
    def set_weight_array(self, *args): return _libfann.neural_net_set_weight_array(self, *args)
    def get_cascade_activation_steepnesses(self): return _libfann.neural_net_get_cascade_activation_steepnesses(self)
    def set_cascade_activation_steepnesses(self, *args): return _libfann.neural_net_set_cascade_activation_steepnesses(self, *args)
neural_net_swigregister = _libfann.neural_net_swigregister
neural_net_swigregister(neural_net)

# This file is compatible with both classic and new-style classes.


