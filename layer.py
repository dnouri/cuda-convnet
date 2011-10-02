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

import options as op
import sys
import ConfigParser as cfg
import os
import numpy as n
import numpy.random as nr
from math import ceil
from ordereddict import OrderedDict
from os import linesep as NL
from options import OptionsParser
import re

class LayerParsingError(Exception):
    pass

# A neuron that doesn't take parameters
class NeuronParser:
    def __init__(self, type, func_str):
        self.type = type
        self.func_str = func_str
        
    def parse(self, type):
        if type == self.type:
            return {'type': self.type,
                    'params': {}}
        return None
    
# A neuron that takes parameters
class ParamNeuronParser(NeuronParser):
    neuron_regex = re.compile(r'^\s*(\w+)\s*\[\s*(\w+(\s*,\w+)*)\s*\]\s*$')
    def __init__(self, type, func_str):
        NeuronParser.__init__(self, type, func_str)
        m = self.neuron_regex.match(type)
        self.base_type = m.group(1)
        self.param_names = m.group(2).split(',')
        assert len(set(self.param_names)) == len(self.param_names)
        
    def parse(self, type):
        m = re.match(r'^%s\s*\[([\d,\.\s\-e]*)\]\s*$' % self.base_type, type)
        if m:
            try:
                param_vals = [float(v.strip()) for v in m.group(1).split(',')]
                if len(param_vals) == len(self.param_names):
                    return {'type': self.base_type,
                            'params': dict(zip(self.param_names, param_vals))}
            except TypeError:
                pass
        return None

class AbsTanhNeuronParser(ParamNeuronParser):
    def __init__(self):
        ParamNeuronParser.__init__(self, 'abstanh[a,b]', 'f(x) = a * |tanh(b * x)|')
        
    def parse(self, type):
        dic = ParamNeuronParser.parse(self, type)
        # Make b positive, since abs(tanh(bx)) = abs(tanh(-bx)) and the C++ code
        # assumes b is positive.
        if dic:
            dic['params']['b'] = abs(dic['params']['b'])
        return dic

# Subclass that throws more convnet-specific exceptions than the default
class MyConfigParser(cfg.SafeConfigParser):
    def safe_get(self, section, option, f=cfg.SafeConfigParser.get, typestr=None, default=None):
        try:
            return f(self, section, option)
        except cfg.NoOptionError, e:
            if default is not None:
                return default
            raise LayerParsingError("Layer '%s': required parameter '%s' missing" % (section, option))
        except ValueError, e:
            if typestr is None:
                raise e
            raise LayerParsingError("Layer '%s': parameter '%s' must be %s" % (section, option, typestr))
        
    def safe_get_list(self, section, option, f, typestr=None):
        v = self.safe_get(section, option)
        try:
            return [f(x) for x in v.split(',')]
        except:
            raise LayerParsingError("Layer '%s': parameter '%s' must be ','-delimited list of %s" % (section, option, typestr))
        
    def safe_get_int(self, section, option, default=None):
        return self.safe_get(section, option, f=cfg.SafeConfigParser.getint, typestr='int', default=default)
        
    def safe_get_float(self, section, option, default=None):
        return self.safe_get(section, option, f=cfg.SafeConfigParser.getfloat, typestr='float', default=default)
    
    def safe_get_bool(self, section, option, default=None):
        return self.safe_get(section, option, f=cfg.SafeConfigParser.getboolean, typestr='bool', default=default)
    
    def safe_get_float_list(self, section, option):
        return self.safe_get_list(section, option, float, typestr='floats')
    
    def safe_get_int_list(self, section, option):
        return self.safe_get_list(section, option, int, typestr='ints')
    
    def safe_get_bool_list(self, section, option):
        return self.safe_get_list(section, option, bool, typestr='bools')
                     
class LayerParser:
    def requires_params(self):
        return False
        
    def add_params(self, name, mcp, dic):
        pass
    
    def parse(self, name, mcp, prev_layers, model):
        dic = {}
        dic['name'] = name
        dic['type'] = mcp.safe_get(name, 'type')

        return dic
    
    @staticmethod
    def parse_neuron(layer_name, neuron_str):
        for n in neuron_parsers:
            p = n.parse(neuron_str)
            if p: # Successfully parsed neuron, return it
                return p
        # Could not parse neuron
        # Print available neuron types
        colnames = ['Neuron type', 'Function']
        m = max(len(colnames[0]), OptionsParser._longest_value(neuron_parsers, key=lambda x:x.type)) + 2
        ntypes = [OptionsParser._bold(colnames[0].ljust(m))] + [n.type.ljust(m) for n in neuron_parsers]
        fnames = [OptionsParser._bold(colnames[1])] + [n.func_str for n in neuron_parsers]
        usage_lines = NL.join(ntype + fname for ntype,fname in zip(ntypes, fnames))
        
        raise LayerParsingError("Layer '%s': unable to parse neuron type '%s'. Valid neuron types: %sWhere neurons have parameters, they must be floats." % (layer_name, neuron_str, NL + usage_lines + NL))
    
    @staticmethod
    def make_weights(numRows, numCols, init, order='C'):
        weights = n.array(init * nr.randn(numRows, numCols), dtype=n.single, order=order)
        weights_inc = n.zeros_like(weights)
        return weights, weights_inc    
    
    @staticmethod
    def verify_int_range(layer_name, v, param_name, _min, _max):
        if _min is not None and _max is not None and (v < _min or v > _max):
            raise LayerParsingError("Layer '%s': parameter '%s' must be in the range %d-%d" % (layer_name, param_name, _min, _max))
        elif _min is not None and v < _min:
            raise LayerParsingError("Layer '%s': parameter '%s' must be greater than %d" % (layer_name, param_name, _min))
        elif _max is not None and v > _max:
            raise LayerParsingError("Layer '%s': parameter '%s' must be smaller than %d" % (layer_name, param_name, _max))
    
    @staticmethod
    def verify_divisible(layer_name, value, div, value_name, div_name=None):
        if value % div != 0:
            raise LayerParsingError("Layer '%s': parameter '%s' must be divisible by %s" % (layer_name, value_name, str(div) if div_name is None else "'%s'" % div_name))
        
    @staticmethod
    def verify_str_in(layer_name, value, lst):
        if value not in lst:
            raise LayerParsingError("Layer '%s': parameter '%s' must be one of %s" % (layer_name, value_name, ", ".join("'%s'" % s for s in lst)))

    @staticmethod
    def parse_layers(layer_cfg_path, param_cfg_path, model, layers=[]):
        try:
            if not os.path.exists(layer_cfg_path):
                raise LayerParsingError("Layer definition file '%s' does not exist" % layer_cfg_path)
            if not os.path.exists(param_cfg_path):
                raise LayerParsingError("Layer parameter file '%s' does not exist" % param_cfg_path)
            if len(layers) == 0:
                mcp = MyConfigParser(dict_type=OrderedDict)
                mcp.read([layer_cfg_path])
                for name in mcp.sections():
                    if not mcp.has_option(name, 'type'):
                        raise LayerParsingError("Layer '%s': no type given" % name)
                    ltype = mcp.safe_get(name, 'type')
                    if ltype not in layer_parsers:
                        raise LayerParsingError("Layer '%s': Unknown layer type: '%s'" % (name, ltype))
                    layers += [layer_parsers[ltype].parse(name, mcp, layers, model)]
            
            for l in layers:
                if not l['type'].startswith('cost.'):
                    found = max(l['name'] in [layers[n]['name'] for n in l2['inputs']] for l2 in layers if 'inputs' in l2)
                    if not found:
                        raise LayerParsingError("Layer '%s' of type '%s' is unused" % (l['name'], l['type']))

            mcp = MyConfigParser(dict_type=OrderedDict)
            mcp.read([param_cfg_path])
            
            for i in xrange(len(layers)):
                name = layers[i]['name']
                ltype = layers[i]['type']
                lp = layer_parsers[ltype]
                if not mcp.has_section(name) and lp.requires_params():
                    raise LayerParsingError("Layer '%s' of type '%s' requires extra parameters, but none given in file '%s'." % (name, ltype, param_cfg_path))
                lp.add_params(name, mcp, layers[i])
        except LayerParsingError, e:
            print e
            sys.exit(1)
         
        return layers
        
    @staticmethod
    def register_layer_parser(ltype, cls):
        if ltype in layer_parsers:
            raise LayerParsingError("Layer type '%s' already registered" % ltype)
        layer_parsers[ltype] = cls

# Any layer that takes an input (i.e. non-data layer)
class LayerWithInputParser(LayerParser):
    def __init__(self, num_inputs=-1):
        self.num_inputs = num_inputs
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerParser.parse(self, name, mcp, prev_layers, model)
        
        dic['inputs'] = [inp.strip() for inp in mcp.safe_get(name, 'inputs').split(',')]
        prev_names = [p['name'] for p in prev_layers]
        for inp in dic['inputs']:
            if inp not in prev_names:
                raise LayerParsingError("Layer '%s': input '%s' not defined" % (name, inp))
        dic['inputs'] = [prev_names.index(inp) for inp in dic['inputs']]
        dic['numInputs'] = [prev_layers[i]['outputs'] for i in dic['inputs']]
        
        if self.num_inputs > 0 and len(dic['numInputs']) != self.num_inputs:
            raise LayerParsingError("Layer '%s': number of inputs must be %d", name, self.num_inputs) 
        
        return dic

class FCLayerParser(LayerWithInputParser):
    def requires_params(self):
        return True
    
    @classmethod
    def verify_num_params(cls, dic, param):
        if len(dic[param]) != len(dic['inputs']):
            raise LayerParsingError("Layer '%s': %s list length does not match number of inputs" % (dic['name'], param))
    
    def add_params(self, name, mcp, dic):
        dic['epsW'] = mcp.safe_get_float_list(name, 'epsW')
        dic['epsB'] = mcp.safe_get_float(name, 'epsB')
        dic['momW'] = mcp.safe_get_float_list(name, 'momW')
        dic['momB'] = mcp.safe_get_float(name, 'momB')
        dic['wc'] = mcp.safe_get_float_list(name, 'wc')
        
        FCLayerParser.verify_num_params(dic, 'epsW')
        FCLayerParser.verify_num_params(dic, 'momW')
        FCLayerParser.verify_num_params(dic, 'wc')
    
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)

        dic['outputs'] = mcp.safe_get_int(name, 'outputs')
        dic['neuron'] = LayerParser.parse_neuron(name, mcp.safe_get(name, 'neuron'))
        dic['initW'] = mcp.safe_get_float_list(name, 'initW')
        
        LayerParser.verify_int_range(name, dic['outputs'], 'outputs', 1, None)
        FCLayerParser.verify_num_params(dic, 'initW')
        
        weights = [LayerParser.make_weights(numIn, dic['outputs'], init, order='F') for numIn,init in zip(dic['numInputs'], dic['initW'])]
        biases = LayerParser.make_weights(1, dic['outputs'], 0, order='F')
        dic['weights'] = [w[0] for w in weights]
        dic['weightsInc'] = [w[1] for w in weights]
        dic['biases'] = biases[0]
        dic['biasesInc'] = biases[1]
        
        print "Initialized fully-connected layer '%s', producing %d outputs" % (name, dic['outputs'])
        return dic

class LocalLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
    
    def requires_params(self):
        return True
    
    def add_params(self, name, mcp, dic):
        dic['epsW'] = mcp.safe_get_float(name, 'epsW')
        dic['epsB'] = mcp.safe_get_float(name, 'epsB')
        dic['momW'] = mcp.safe_get_float(name, 'momW')
        dic['momB'] = mcp.safe_get_float(name, 'momB')
        dic['wc'] = mcp.safe_get_float(name, 'wc')
        
    # Returns (groups, filterChannels) array that represents the set
    # of image channels to which each group is connected
    def gen_rand_conns(self, groups, channels, filterChannels):
        overSample = groups * filterChannels / channels
        return [x for i in xrange(overSample) for x in nr.permutation(range(channels))]
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        
        dic['padding'] = mcp.safe_get_int(name, 'padding', default=0)
        dic['stride'] = mcp.safe_get_int(name, 'stride', default=1)
        dic['filterSize'] = mcp.safe_get_int(name, 'filterSize')
        dic['filterPixels'] = dic['filterSize']**2
        dic['modulesX'] = 1 + int(ceil((2 * dic['padding'] + dic['imgSize'] - dic['filterSize']) / float(dic['stride'])))
        dic['modules'] = dic['modulesX']**2
        dic['filters'] = mcp.safe_get_int(name, 'filters')
        dic['groups'] = mcp.safe_get_int(name, 'groups', default=1)
        dic['randSparse'] = mcp.safe_get_bool(name, 'randSparse', default=False)
        dic['filters'] *= dic['groups']
        dic['outputs'] = dic['modules'] * dic['filters']

        LayerParser.verify_int_range(name, dic['stride'], 'stride', 1, None)
        LayerParser.verify_int_range(name, dic['filterSize'],'filterSize', 1, None)
        LayerParser.verify_int_range(name, dic['padding'], 'padding', 0, None)
        LayerParser.verify_int_range(name, dic['channels'], 'channels', 1, None)
        LayerParser.verify_int_range(name, dic['imgSize'], 'imgSize', 1, None)
        LayerParser.verify_int_range(name, dic['groups'], 'groups', 1, None)
        
        dic['filterChannels'] = dic['channels'] / dic['groups']
        
        if dic['numInputs'][0] % dic['imgPixels'] != 0 or dic['imgSize'] * dic['imgSize'] != dic['imgPixels']:
            raise LayerParsingError("Layer '%s': has %-d dimensional input, not interpretable as square %d-channel images" % (name, dic['numInputs'][0], dic['channels']))
        if dic['channels'] > 3 and dic['channels'] % 4 != 0:
            raise LayerParsingError("Layer '%s': number of channels must be smaller than 4 or divisible by 4" % name)
        if dic['filterSize'] > 2 * dic['padding'] + dic['imgSize']:
            raise LayerParsingError("Layer '%s': filter size (%d) greater than image size + 2 * padding (%d)" % (name, dic['filterSize'], 2 * dic['padding'] + dic['imgSize']))
        
        if dic['randSparse']: # Random sparse connectivity requires some extra checks
            if dic['groups'] == 1:
                raise LayerParsingError("Layer '%s': number of groups must be greater than 1 when using random sparse connectivity" % name)
            dic['filterChannels'] = mcp.safe_get_int(name, 'filterChannels', default=dic['filterChannels'])
            LayerParser.verify_divisible(name, dic['channels'], dic['filterChannels'], 'channels', 'filterChannels')
            LayerParser.verify_divisible(name, dic['filterChannels'], 4, 'filterChannels')
            LayerParser.verify_divisible(name, dic['groups']*dic['filterChannels'], dic['channels'], 'groups * filterChannels', 'channels')
            dic['filterConns'] = self.gen_rand_conns(dic['groups'], dic['channels'], dic['filterChannels'])
        else:
            if dic['groups'] > 1:
                LayerParser.verify_divisible(name, dic['channels'], 4*dic['groups'], 'channels', '4 * groups')
            LayerParser.verify_divisible(name, dic['channels'], dic['groups'], 'channels', 'groups')

        LayerParser.verify_divisible(name, dic['filters']/dic['groups'], 16, 'filters')
        
        dic['padding'] = -dic['padding']
        dic['neuron'] = LayerParser.parse_neuron(name, mcp.safe_get(name, 'neuron'))
        dic['initW'] = mcp.safe_get_float(name, 'initW')
          
        return dic    

class ConvLayerParser(LocalLayerParser):
    def parse(self, name, mcp, prev_layers, model):
        dic = LocalLayerParser.parse(self, name, mcp, prev_layers, model)
        
        dic['partialSum'] = mcp.safe_get_int(name, 'partialSum')
        dic['sharedBiases'] = mcp.safe_get_bool(name, 'sharedBiases', default=True)

        if dic['partialSum'] != 0 and dic['modules'] % dic['partialSum'] != 0:
            raise LayerParsingError("Layer '%s': convolutional layer produces %d outputs per filter, but given partialSum parameter (%d) does not divide this number" % (name, dic['modules'], dic['partialSum']))

        num_biases = dic['filters'] if dic['sharedBiases'] else dic['modules']*dic['filters']
        dic['weights'], dic['weightsInc'] = LayerParser.make_weights(dic['filterPixels']*dic['filterChannels'], \
                                                                     dic['filters'], dic['initW'], order='C')
        dic['biases'], dic['biasesInc'] = LayerParser.make_weights(num_biases, 1, 0, order='C')
        
        print "Initialized convolutional layer '%s', producing %d groups of %dx%d %d-channel output" % \
              (name, dic['groups'], dic['modulesX'], dic['modulesX'], dic['filters']/dic['groups'])
  
        return dic    
    
class LocalUnsharedLayerParser(LocalLayerParser):
    def parse(self, name, mcp, prev_layers, model):
        dic = LocalLayerParser.parse(self, name, mcp, prev_layers, model)

        dic['weights'], dic['weightsInc'] = LayerParser.make_weights(dic['modules'] * dic['filterPixels'] * dic['filterChannels'], \
                                                                     dic['filters'], dic['initW'], order='C')
        dic['biases'], dic['biasesInc'] = LayerParser.make_weights(dic['modules'] * dic['filters'], 1, 0, order='C')
        
        print "Initialized locally-connected layer '%s', producing %d groups of %dx%d %d-channel output" % \
              (name, dic['groups'], dic['modulesX'], dic['modulesX'], dic['filters']/dic['groups'])
  
        return dic  
    
class DataLayerParser(LayerParser):
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerParser.parse(self, name, mcp, prev_layers, model)
        dic['dataIdx'] = mcp.safe_get_int(name, 'dataIdx')
        dic['outputs'] = model.train_data_provider.get_data_dims(idx=dic['dataIdx'])
        
        print "Initialized data layer '%s', producing %d outputs" % (name, dic['outputs'])
        return dic

class SoftmaxLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['outputs'] = prev_layers[dic['inputs'][0]]['outputs']
        print "Initialized softmax layer '%s', producing %d outputs" % (name, dic['outputs'])
        return dic

class PoolLayerParser(LayerWithInputParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=1)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        dic['sizeX'] = mcp.safe_get_int(name, 'sizeX')
        dic['start'] = mcp.safe_get_int(name, 'start', default=0)
        dic['stride'] = mcp.safe_get_int(name, 'stride')
        dic['outputsX'] = mcp.safe_get_int(name, 'outputsX', default=0)
        dic['stride'] = mcp.safe_get_int(name, 'stride')
        dic['pool'] = mcp.safe_get(name, 'pool')
        
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        
        LayerParser.verify_int_range(name, dic['sizeX'], 'sizeX', 1, dic['imgSize'])
        LayerParser.verify_int_range(name, dic['stride'], 'stride', 1, dic['sizeX'])
        LayerParser.verify_int_range(name, dic['outputsX'], 'outputsX', 0, None)
        LayerParser.verify_int_range(name, dic['channels'], 'channels', 1, None)
        
        LayerParser.verify_divisible(name, dic['channels'], 16, 'channels')
        LayerParser.verify_str_in(name, dic['pool'], ['max', 'avg'])
        
        if dic['numInputs'][0] % dic['imgPixels'] != 0 or dic['imgSize'] * dic['imgSize'] != dic['imgPixels']:
            raise LayerParsingError("Layer '%s': has %-d dimensional input, not interpretable as %d-channel images" % (name, dic['numInputs'][0], dic['channels']))
        if dic['outputsX'] <= 0:
            dic['outputsX'] = int(ceil((dic['imgSize'] - dic['start'] - dic['sizeX']) / float(dic['stride']))) + 1;
        dic['outputs'] = dic['outputsX']**2 * dic['channels']
        
        print "Initialized %s-pooling layer '%s', producing %dx%d %d-channel output" % (dic['pool'], name, dic['outputsX'], dic['outputsX'], dic['channels'])
        return dic
    
class NormLayerParser(LayerWithInputParser):
    def __init__(self, norm_type):
        LayerWithInputParser.__init__(self, num_inputs=1)
        self.norm_type = norm_type
        
    def parse(self, name, mcp, prev_layers, model):
        dic = LayerWithInputParser.parse(self, name, mcp, prev_layers, model)
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        dic['sizeX'] = mcp.safe_get_int(name, 'sizeX')
        dic['pow'] = mcp.safe_get_float(name, 'pow')
        dic['scale'] = mcp.safe_get_float(name, 'scale') / (dic['sizeX']**2)
        
        dic['imgPixels'] = dic['numInputs'][0] / dic['channels']
        dic['imgSize'] = int(n.sqrt(dic['imgPixels']))
        
        LayerParser.verify_int_range(name, dic['sizeX'], 'sizeX', 1, dic['imgSize'])
        LayerParser.verify_int_range(name, dic['channels'], 'channels', 1, None)
        
        if dic['channels'] > 3 and dic['channels'] % 4 != 0:
            raise LayerParsingError("Layer '%s': number of channels must be smaller than 4 or divisible by 4" % name)
        
        if dic['numInputs'][0] % dic['imgPixels'] != 0 or dic['imgSize'] * dic['imgSize'] != dic['imgPixels']:
            raise LayerParsingError("Layer '%s': has %-d dimensional input, not interpretable as %d-channel images" % (name, dic['numInputs'][0], dic['channels']))
        dic['outputs'] = dic['imgPixels'] * dic['channels']
        print "Initialized %s-normalization layer '%s', producing %dx%d %d-channel output" % (self.norm_type, name, dic['imgSize'], dic['imgSize'], dic['channels'])
        return dic

class CostParser(LayerWithInputParser):
    def __init__(self, num_inputs=-1):
        LayerWithInputParser.__init__(self, num_inputs=num_inputs)
        
    def requires_params(self):
        return True    
    
    def add_params(self, name, mcp, dic):
        dic['coeff'] = mcp.safe_get_float(name, 'coeff')
            
class LogregCostParser(CostParser):
    def __init__(self):
        LayerWithInputParser.__init__(self, num_inputs=2)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = CostParser.parse(self, name, mcp, prev_layers, model)
        if dic['numInputs'][0] != 1: # first input must be labels
            raise LayerParsingError("Layer '%s': Dimensionality of first input must be 1" % name)
        if prev_layers[dic['inputs'][1]]['type'] != 'softmax':
            raise LayerParsingError("Layer '%s': Second input must be softmax layer" % name)
        if dic['numInputs'][1] != model.train_data_provider.get_num_classes():
            raise LayerParsingError("Layer '%s': Softmax input '%s' must produce %d outputs, because that is the number of classes in the dataset" \
                                    % (name, prev_layers[dic['inputs'][1]]['name'], model.train_data_provider.get_num_classes()))
        
        print "Initialized logistic regression cost '%s'" % name
        return dic

# All the layer parsers
layer_parsers = {'data': DataLayerParser(),
                 'fc': FCLayerParser(),
                 'conv': ConvLayerParser(),
                 'local': LocalUnsharedLayerParser(),
                 'softmax': SoftmaxLayerParser(),
                 'pool': PoolLayerParser(),
                 'rnorm': NormLayerParser('response'),
                 'cnorm': NormLayerParser('contrast'),
                 'cost.logreg': LogregCostParser()}
 
# All the neuron parsers
# This isn't a name --> parser mapping as the layer parsers above because neurons don't have fixed names.
# A user may write tanh[0.5,0.25], etc.
neuron_parsers = sorted([NeuronParser('ident', 'f(x) = x'),
                         NeuronParser('logistic', 'f(x) = 1 / (1 + e^-x)'),
                         NeuronParser('abs', 'f(x) = |x|'),
                         NeuronParser('relu', 'f(x) = max(0, x)'),
                         NeuronParser('softrelu', 'f(x) = log(1 + e^x)'),
                         ParamNeuronParser('tanh[a,b]', 'f(x) = a * tanh(b * x)'),
                         ParamNeuronParser('brelu[a]', 'f(x) = min(a, max(0, x))'),
                         AbsTanhNeuronParser()],
                        key=lambda x:x.type)
