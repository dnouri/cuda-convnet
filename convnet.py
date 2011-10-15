# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as n
import numpy.random as nr
from util import *
from data import *
from options import *
from gpumodel import *
import sys
import math as m
import layer as lay
from convdata import *
from os import linesep as NL

class GPUModel(IGPUModel):
    def __init__(self, model_name, op, load_dic, dp_params={}):
        filename_options = []
        dp_params['multiview_test'] = op.get_value('multiview_test')
        dp_params['crop_border'] = op.get_value('crop_border')
        IGPUModel.__init__(self, model_name, op, load_dic, filename_options, dp_params=dp_params)
        
    def init_model_lib(self):
        self.libmodel.initModel(self.layers, self.minibatch_size, self.device_ids[0])
        
    def init_model_state(self):
        ms = self.model_state
        if self.load_file:
            ms['layers'] = lay.LayerParser.parse_layers(self.layer_def, self.layer_params, self, ms['layers'])
        else:
            ms['layers'] = lay.LayerParser.parse_layers(self.layer_def, self.layer_params, self)
        
        logreg_name = self.op.get_value('logreg_name')
        if logreg_name:
            self.logreg_idx = self.get_layer_idx(logreg_name, check_type='cost.logreg')
        
        # Convert convolutional layers to local
        if self.op.get_value('unshare_conv'):
            for layer in ms['layers']:
                if layer['type'] == 'conv':
                    lay.LocalLayerParser.conv_to_local(layer)
                        
    def get_layer_idx(self, layer_name, check_type=None):
        try:
            layer_idx = [l['name'] for l in self.model_state['layers']].index(layer_name)
            if check_type:
                layer_type = self.model_state['layers'][layer_idx]['type']
                if layer_type != check_type:
                    raise ModelStateException("Layer with name '%s' has type '%s'; should be '%s'." % (layer_name, layer_type, check_type))
            return layer_idx
        except ValueError:
            raise ModelStateException("Layer with name '%s' not defined." % layer_name)
            
    def fill_excused_options(self):
        if self.op.get_value('check_grads'):
            self.op.set_value('save_path', '')
            self.op.set_value('train_batch_range', '0')
            self.op.set_value('test_batch_range', '0')
            self.op.set_value('data_path', '')
            
    # Make sure the data provider returned data in proper format
    def parse_batch_data(self, batch_data, train=True):
        if max(d.dtype != n.single for d in batch_data[2]):
            raise DataProviderException("All matrices returned by data provider must consist of single-precision floats.")
        return batch_data

    def start_batch(self, batch_data, train=True):
        data = batch_data[2]
        if self.check_grads:
            self.libmodel.checkGradients(data)
        elif not train and self.multiview_test:
            self.libmodel.startMultiviewTest(data, self.train_data_provider.num_views, self.logreg_idx)
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
                    print "%sLayer '%s' weights: %e [%e]" % (NL, l['name'], n.mean(n.abs(l['weights'])), n.mean(n.abs(l['weightsInc']))),
                elif type(l['weights']) == list:
                    print ""
                    print NL.join("Layer '%s' weights[%d]: %e [%e]" % (l['name'], i, n.mean(n.abs(w)), n.mean(n.abs(wi))) for i,(w,wi) in enumerate(zip(l['weights'],l['weightsInc']))),
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
        op.add_option("multiview-test", "multiview_test", BooleanOptionParser, "Cropped DP: test on multiple patches?", default=0, requires=['logreg_name'])
        op.add_option("crop-border", "crop_border", IntegerOptionParser, "Cropped DP: crop border size", default=4, set_once=True)
        op.add_option("logreg-name", "logreg_name", StringOptionParser, "Cropped DP: logreg layer name", default="")
        op.add_option("unshare-conv", "unshare_conv", BooleanOptionParser, "Convert all convolutional layers to unshared locally-connected?", default=False)
        
        op.delete_option('max_test_err')
        op.options["max_filesize_mb"].default = 0
        op.options["testing_freq"].default = 50
        op.options["num_epochs"].default = 50000
        op.options['dp_type'].default = None
        
        DataProvider.register_data_provider('cifar', 'CIFAR', CIFARDataProvider)
        DataProvider.register_data_provider('dummy-cn-n', 'Dummy ConvNet', DummyConvNetDataProvider)
        DataProvider.register_data_provider('cifar-cropped', 'Cropped CIFAR', CroppedCIFARDataProvider)
        
        return op
    
if __name__ == "__main__":
    #nr.seed(5)
    op = GPUModel.get_options_parser()

    op, load_dic = IGPUModel.parse_options(op)
    model = GPUModel("ConvNet", op, load_dic)
    model.start()
