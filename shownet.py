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
import random as r
import numpy.random as nr
from convnet import GPUModel
from options import *

try:
    import pylab as pl
except:
    print "This script requires the matplotlib python library (Ubuntu/Fedora package name python-matplotlib). Please install it."
    sys.exit(1)

class ShowNetError(Exception):
    pass

class ShowGPUModel(GPUModel):
    def __init__(self, model_name, op, load_dic):
        GPUModel.__init__(self, model_name, op, load_dic)
    
    def import_model(self):
        if self.op.get_value('show_preds'):
            GPUModel.import_model(self)
        
    def init_model_lib(self):
        if self.op.get_value('show_preds'):
            GPUModel.init_model_lib(self)
        
    def get_gpus(self):
        if self.op.get_value('show_preds'):
            GPUModel.get_gpus(self)
    
    def plot_cost(self):
        if self.show_cost not in self.train_outputs[0]:
            raise ShowNetError("Cost function with name '%s' not defined by given convnet." % self.show_cost)
        train_errors = [o[self.show_cost][0] for o in self.train_outputs]
        test_errors = [o[self.show_cost][0] for o in self.test_outputs]

        numbatches = len(self.train_batch_range)
        test_errors = numpy.row_stack(test_errors)
        test_errors = numpy.tile(test_errors, (1, self.testing_freq))
        test_errors = list(test_errors.flatten())
        test_errors += [test_errors[-1]] * (len(train_errors) - len(test_errors))

        numepochs = len(train_errors) / float(numbatches)
        pl.figure(1)
        x = range(0, len(train_errors))
        pl.plot(x, train_errors, 'k-', label='Training set')
        pl.plot(x, test_errors, 'r-', label='Test set')
        pl.legend()
        ticklocs = range(numbatches, len(train_errors) - len(train_errors) % numbatches + 1, numbatches)
        epoch_label_gran = int(ceil(numepochs / 20.)) # aim for about 20 labels
        epoch_label_gran = int(ceil(float(epoch_label_gran) / 10) * 10) # but round to nearest 10
        ticklabels = map(lambda x: str((x[1] / numbatches)) if x[0] % epoch_label_gran == epoch_label_gran-1 else '', enumerate(ticklocs))

        pl.xticks(ticklocs, ticklabels)
        pl.xlabel('Epoch')
        pl.ylabel(self.show_cost)
        pl.title(self.show_cost)
        
    def make_filter_fig(self, filters, filter_start, fignum, _title, num_filters, combine_chans):
        FILTERS_PER_ROW = 16
        MAX_ROWS = 16
        MAX_FILTERS = FILTERS_PER_ROW * MAX_ROWS
        num_colors = filters.shape[0]
        f_per_row = int(ceil(FILTERS_PER_ROW / float(1 if combine_chans else num_colors)))
        filter_end = min(filter_start+MAX_FILTERS, num_filters)
        filter_rows = int(ceil(float(filter_end - filter_start) / f_per_row))
    
        filter_pixels = filters.shape[1]
        filter_size = int(sqrt(filters.shape[1]))
        fig = pl.figure(fignum)
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
                
        pl.xticks([])
        pl.yticks([])
        if not combine_chans:
            pl.imshow(bigpic, cmap=pl.cm.gray, interpolation='nearest')
        else:
            bigpic = bigpic.swapaxes(0,2).swapaxes(0,1)
            pl.imshow(bigpic, interpolation='nearest')        
        
    def plot_filters(self):
        filter_start = 0 # First filter to show
        layer_names = [l['name'] for l in self.layers]
        if self.show_filters not in layer_names:
            raise ShowNetError("Layer with name '%s' not defined by given convnet." % self.show_filters)
        layer = self.layers[layer_names.index(self.show_filters)]
        filters = layer['weights']
        if layer['type'] == 'fc': # Fully-connected layer
            filters = filters[self.input_idx]
            num_filters = layer['outputs']
            channels = self.channels
        elif layer['type'] in ('conv', 'local'): # Conv layer
            num_filters = layer['filters']
            channels = layer['filterChannels']
            if layer['type'] == 'local':
                filters = filters.reshape((layer['modules'], layer['filterPixels'] * channels, layer['filters']))
                filter_start = r.randint(0, layer['modules']-1)*layer['filters'] # pick out some random modules
                filters = filters.swapaxes(0,1).reshape(channels * layer['filterPixels'], layer['filters'] * layer['modules'])
                num_filters *= layer['modules']

        filters = filters.reshape(channels, filters.shape[0]/channels, filters.shape[1])
        combine_chans = not self.no_rgb and channels == 3
        
        filters -= filters.min()
        filters /= filters.max()

        self.make_filter_fig(filters, filter_start, 2, 'Layer %s' % self.show_filters, num_filters, combine_chans)
    
    def plot_predictions(self):
        if self.test_data_provider.num_colors != 3:
            raise ShowNetError("Can only show color images")
        data = self.get_next_batch(train=False)[2] # get a test batch
        num_classes = self.test_data_provider.get_num_classes()
        NUM_ROWS = 2
        NUM_COLS = 4
        NUM_IMGS = NUM_ROWS * NUM_COLS
        NUM_TOP_CLASSES = min(num_classes, 4) # show this many top labels
        
        img_size = int(sqrt(self.test_data_provider.get_data_dims() / 3))
        label_names = self.test_data_provider.batch_meta['label_names']
        if self.only_errors:
            preds = n.zeros((data[0].shape[1], num_classes), dtype=n.single)
        else:
            preds = n.zeros((NUM_IMGS, num_classes), dtype=n.single)
            img_indices = nr.randint(0, data[0].shape[1], NUM_IMGS)
            data[0] = n.require(data[0][:,img_indices], requirements='C')
            data[1] = n.require(data[1][:,img_indices], requirements='C')
        data += [preds]
        self.libmodel.startLabeler(data, self.logreg_idx)
        self.finish_batch()
        fig = pl.figure(3)
        fig.text(.4, .95, 'Random test case predictions')
        if self.only_errors:
            err_idx = n.where(preds.argmax(axis=1) != data[1][0,:])[0] # what the net got wrong
            data[0], data[1], preds = data[0][:,err_idx], data[1][:,err_idx], preds[err_idx,:]
        data[0] += self.test_data_provider.data_mean
        data[0] /= 255.0
        #print [sum(x == i for x in data[1][0,:]) for i in range(16)]
        for r in xrange(NUM_ROWS):
            for c in xrange(NUM_COLS):
                img_idx = nr.randint(data[0].shape[1]) if self.only_errors else r * NUM_COLS + c
                pl.subplot(NUM_ROWS*2, NUM_COLS, r * 2 * NUM_COLS + c + 1)
                pl.xticks([])
                pl.yticks([])
                img = data[0][:, img_idx].reshape(3, img_size, img_size).swapaxes(0,2).swapaxes(0,1)
                pl.imshow(img, interpolation='nearest')
                true_label = int(data[1][0,img_idx])

                img_labels = sorted(zip(list(preds[img_idx,:]), label_names), key=lambda x: x[0], reverse=True)[:NUM_TOP_CLASSES]
                img_labels.reverse()
                pl.subplot(NUM_ROWS*2, NUM_COLS, (r * 2 + 1) * NUM_COLS + c + 1, aspect='equal')
                #pl.pie([l[0] for l in img_labels],
                #       labels=[l[1] for l in img_labels])
                ylocs = n.array(range(NUM_TOP_CLASSES)) + 0.5
                height = 0.5
                width = max(ylocs)
                pl.barh(ylocs, [l[0]*width for l in img_labels], height=height)
                pl.title(label_names[true_label])
                pl.yticks(ylocs + height/2, [l[1] for l in img_labels])
                pl.xticks([width/2.0, width], ['50%', '100%'])
                pl.ylim(0, ylocs[-1] + height*2)
    
    def start(self):
        if self.show_cost:
            self.plot_cost()
        if self.show_filters:
            self.plot_filters()
        if self.show_preds:
            self.plot_predictions()
        pl.show()
        sys.exit(0)
            
    @classmethod
    def get_options_parser(cls):
        op = GPUModel.get_options_parser()
        op.add_option("show-cost", "show_cost", StringOptionParser, "Show specified objective function", default="")
        op.add_option("show-filters", "show_filters", StringOptionParser, "Show learned filters in specified layer", default="")
        op.add_option("input-idx", "input_idx", IntegerOptionParser, "Input index for layer given to --show-filters (fully-connected layers only)", default=0)
        op.add_option("no-rgb", "no_rgb", BooleanOptionParser, "Don't combine filter channels into RGB in layer given to --show-filters", default=False)
        op.add_option("channels", "channels", IntegerOptionParser, "Number of channels in layer given to --show-filters (fully-connected layers only)", default=0)
        op.add_option("show-preds", "show_preds", BooleanOptionParser, "Show predictions made on test set", default=False, requires=['logreg_name'])
        op.add_option("only-errors", "only_errors", BooleanOptionParser, "Show only mistaken predictions", default=False, requires=['show_preds'])

        return op
    
if __name__ == "__main__":
    try:
        op = ShowGPUModel.get_options_parser()
        op, load_dic = IGPUModel.parse_options(op)
        model = ShowGPUModel("ConvNet", op, load_dic)
        model.start()
    except (UnpickleError, ShowNetError, opt.GetoptError), e:
        print "----------------"
        print "Error:"
        print e 
