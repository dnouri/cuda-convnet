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

import numpy
import sys
import getopt as opt
from util import *
from math import sqrt, ceil, floor
import os
from gpumodel import IGPUModel

try:
    from pylab import *
except:
    print "This script requires the matplotlib python library (Ubuntu/Fedora package name python-matplotlib). Please install it."
    sys.exit(1)

FILTERS_PER_ROW = 16
MAX_FILTERS = FILTERS_PER_ROW * 16

class ShowNetError(Exception):
    pass

def draw_filters(filters, filter_start, fignum, _title, num_filters, combine_chans):
    num_colors = filters.shape[0]
    f_per_row = int(ceil(FILTERS_PER_ROW / float(1 if combine_chans else num_colors)))
    filter_end = min(filter_start+MAX_FILTERS, num_filters)
    filter_rows = int(ceil(float(filter_end - filter_start) / f_per_row))

    filter_pixels = filters.shape[1]
    filter_size = int(sqrt(filters.shape[1]))
    fig = figure(fignum)
    fig.text(.5, .95, '%s %dx%d filters %d-%d' % (_title, filter_size, filter_size, filter_start, filter_end-1), horizontalalignment='center') 
    num_filters = filter_end - filter_start
    if not combine_chans:
        bigpic = n.zeros((filter_size * filter_rows + filter_rows + 1, filter_size*num_colors * f_per_row + f_per_row + 1), dtype=n.single)
    else:
        bigpic = n.zeros((3, filter_size * filter_rows + filter_rows + 1, filter_size * f_per_row + f_per_row + 1), dtype=n.single)

    for m in xrange(filter_start,filter_end ):
        filter = filters[:,:,m]
        y, x = (m - filter_start) / f_per_row, (m - filter_start) % f_per_row
        if not combine_chans:
            for c in xrange(num_colors):
                filter_pic = filter[c,:].reshape((filter_size,filter_size))
                bigpic[1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                       1 + (1 + filter_size*num_colors) * x + filter_size*c:1 + (1 + filter_size*num_colors) * x + filter_size*(c+1)] = filter_pic
        else:
            filter_pic = filter.reshape((3, filter_size,filter_size))
            bigpic[:,
                   1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                   1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic
            
    xticks([])
    yticks([])
    if not combine_chans:
        imshow(bigpic, cmap=cm.gray, interpolation='nearest')
    else:
        bigpic = bigpic.swapaxes(0,2).swapaxes(0,1)
        imshow(bigpic, interpolation='nearest')

def print_usage():
    print "%s usage:" % sys.argv[0]
    print "-f <file>"
    print "[-e <cost name>] -- plot given cost function value"
    print "[-l <layer name>] -- draw filters in given layer"
    print "[-c <num channels>] -- number of channels in given layer name (for fully-connected layers only)"
    print "[-i <input idx>] -- input index for which to draw filters (for fully-connected layers only)"
    print "[-o] -- don't combine channels into RGB when there are exactly 3 of them"
    print "\nSee http://code.google.com/p/cuda-convnet/wiki/ViewingNet for more thorough documentation, with usage examples."

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_usage()
        sys.exit(0)

    try:
        (options, args) = opt.getopt(sys.argv[1:], "f:l:e:c:i:o")
        options = dict(options)
    
        net_file = options["-f"]
        err_name = options["-e"] if "-e" in options else None
        layer_name = options["-l"] if "-l" in options else None

        dic = IGPUModel.load_checkpoint(net_file)
    
        dic["op"].print_values()
        dic.update(dic["model_state"])
        dic.update(dict((v.name, v.value) for v in dic["op"].options.itervalues()))
    
        # Plot error
        if err_name:
            if err_name not in dic['train_outputs'][0]:
                raise ShowNetError("Cost function with name '%s' not defined by given convnet." % err_name)
            train_errors = [o[err_name][0] for o in dic["train_outputs"]]
            test_errors = [o[err_name][0] for o in dic["test_outputs"]]
            testing_freq = dic["testing_freq"]
    
            numbatches = len(dic["train_batch_range"])
            test_errors = numpy.row_stack(test_errors)
            test_errors = numpy.tile(test_errors, (1, testing_freq))
            test_errors = list(test_errors.flatten())
            test_errors += [test_errors[-1]] * (len(train_errors) - len(test_errors))
    
            numepochs = len(train_errors) / float(numbatches)
            figure(1)
            x = range(0, len(train_errors))
            plot(x, train_errors, 'k-', label='Training set')
            plot(x, test_errors, 'r-', label='Test set')
            legend()
            ticklocs = range(numbatches, len(train_errors) - len(train_errors) % numbatches + 1, numbatches)
            epoch_label_gran = int(ceil(numepochs / 20.)) # aim for about 20 labels
            epoch_label_gran = int(ceil(float(epoch_label_gran) / 10) * 10) # but round to nearest 10
            ticklabels = map(lambda x: str((x[1] / numbatches)) if x[0] % epoch_label_gran == epoch_label_gran-1 else '', enumerate(ticklocs))
    
            xticks(ticklocs, ticklabels)
            xlabel('Epoch')
            ylabel(err_name)
            title(err_name)
        
        # Draw some filters
        if layer_name:
            layer_names = [l['name'] for l in dic['layers']]
            if layer_name not in layer_names:
                raise ShowNetError("Layer with name '%s' not defined by given convnet." % layer_name)
            layer = dic['layers'][layer_names.index(layer_name)]
            filters = layer['weights']
            if type(filters) == list: # Fully-connected layer
                input_idx = int(options["-i"])
                filters = filters[input_idx]
                num_filters = layer['numOutputs']
                channels = int(options["-c"])
            else: # Conv layer
                num_filters = layer['numFilters']
                channels = layer['channels']
            combine_chans = "-o" not in options and channels == 3

            filters = filters.reshape(channels, filters.shape[0]/channels, filters.shape[1])
            filters -= filters.min()
            filters /= filters.max()
    
            draw_filters(filters, 0, 2, 'Layer %s' % layer_name, num_filters, combine_chans)
    
        show()
    except (UnpickleError, ShowNetError, opt.GetoptError), e:
        print "----------------"
        print "Error:"
        print e 

