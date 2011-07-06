# Abstract feed-forward neural net
# 
# Author: Alex Krizhevsky (akrizhevsky@gmail.com)

import numpy as n
import numpy.random as nr
from util import *
from data import *
from options import *
from gpumodel import *
import sys
import math as m
from layer import LayerParser
from convdp import *   

class GPUModel(IGPUModel):
    def __init__(self, model_name, op, load_dic, dp_params=None):
        filename_options = []
        if dp_params is None:
            dp_params={'minibatch_size': op.get_value('minibatch_size')}
        IGPUModel.__init__(self, model_name, op, load_dic, filename_options, dp_params=dp_params)
        
    def init_model_lib(self):
        self.libmodel.initModel(self.layers, self.minibatch_size, self.device_ids[0])
        
    def init_model_state(self):
        ms = self.model_state
        if self.load_file:
            ms['layers'] = LayerParser.parse_layers(self.layer_def, self.layer_params, self, ms['layers'])
        else:
            ms['layers'] = LayerParser.parse_layers(self.layer_def, self.layer_params, self)
            
    def fill_excused_options(self):
        if self.op.get_value('save_path') is None:
            self.op.set_value('save_path', "")
            self.op.set_value('train_batch_range', '0')
            self.op.set_value('test_batch_range', '0')
            self.op.set_value('data_path', '')
            
    # Not necessary here
    def parse_batch_data(self, batch_data, train=True):
        return batch_data

    def start_batch(self, batch_data, train=True):
        data = batch_data[2]
        if self.check_grads:
            self.libmodel.checkGradients(data)
        else:
            self.libmodel.startBatch(data, not train)
        
    def print_iteration(self):
        print "%d.%d..." % (self.epoch, self.batchnum),
        
    def print_train_time(self, compute_time_py):
        print "(%.3f sec)" % (compute_time_py)
        
    def print_train_results(self):
        for errname, errvals in self.train_outputs[-1].items():
            print "%s: " % errname,
            print ", ".join("%6f" % v for v in errvals),
            if sum(m.isnan(v) for v in errvals) > 0 or sum(m.isinf(v) for v in errvals):
                print "^ got nan or inf!"
                sys.exit(1)

    def print_test_status(self):
        pass
    
    def print_test_results(self):
        print ""
        print "======================Test output======================"
        for errname, errvals in self.test_outputs[-1].items():
            print "%s: " % errname,
            print ", ".join("%6f" % v for v in errvals),
        print ""
        print "-------------------------------------------------------", 
        for i,l in enumerate(self.layers): # This is kind of hacky but will do for now.
            if 'weights' in l:
                if type(l['weights']) == n.ndarray:
                    print "\nLayer '%s' weights: %e [%e]" % (l['name'], n.mean(n.abs(l['weights'])), n.mean(n.abs(l['weightsInc']))),
                elif type(l['weights']) == list:
                    print ""
                    print "\n".join("Layer '%s' weights[%d]: %e [%e]" % (l['name'], i, n.mean(n.abs(w)), n.mean(n.abs(wi))) for i,(w,wi) in enumerate(zip(l['weights'],l['weightsInc']))),
        print ""
        
    def conditional_save(self):
        self.save_state()
        print "-------------------------------------------------------"
        print "Saved checkpoint to %s" % os.path.join(self.save_path, self.save_file)
        print "=======================================================",
        
    def aggregate_test_outputs(self, test_outputs):
        for i in xrange(1 ,len(test_outputs)):
            for k,v in test_outputs[i].items():
                for j in xrange(len(v)):
                    test_outputs[0][k][j] += test_outputs[i][k][j]
        return test_outputs[0]
    
    @classmethod
    def get_options_parser(cls):
        op = IGPUModel.get_options_parser()
        op.add_option("mini", "minibatch_size", IntegerOptionParser, "Minibatch size", default=128)
        op.add_option("layer-def", "layer_def", StringOptionParser, "Layer definition file", set_once=True)
        op.add_option("layer-params", "layer_params", StringOptionParser, "Layer parameter file")
        op.add_option("check-grads", "check_grads", BooleanOptionParser, "Check gradients and quit?", default=0, excuses=['data_path','save_path','train_batch_range','test_batch_range'])
        
        op.delete_option('max_test_err')
        op.options["max_filesize_mb"].default = 0
        op.options["testing_freq"].default = 50
        op.options["num_epochs"].default = 50000
        op.options['dp_type'].default = None
        
        return op
    
if __name__ == "__main__":
    #nr.seed(5)
    op = GPUModel.get_options_parser()
    
    DataProvider.register_data_provider('cifar', 'CIFAR', CIFARDataProvider)
    DataProvider.register_data_provider('dummy-cn-n', 'Dummy ConvNet', DummyConvNetDataProvider)

    op, load_dic = IGPUModel.parse_options(op)
    model = GPUModel("ConvNet", op, load_dic)
    model.start()
