from data import *

class CIFARDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.minibatch_size = dp_params['minibatch_size']
        
    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        data_mean = self.batch_meta['data_mean']
        
        if 'processed' not in datadic:
            datadic['data'] = n.require((datadic['data'].astype(n.single) - data_mean), dtype=n.single, requirements='C')# / 255
            datadic['labels'] = datadic['labels'].reshape((1, datadic['data'].shape[1]))
            datadic['num_real_cases'] = datadic['data'].shape[1]
            # Pad data to size of minibatch
            if datadic['data'].shape[1] % self.minibatch_size != 0:
                datadic['data'] = n.require(n.c_[datadic['data'], n.zeros((datadic['data'].shape[0], self.minibatch_size - datadic['data'].shape[1] % self.minibatch_size), dtype=n.single)], requirements='C', dtype=n.single)
                datadic['labels'] = n.require(n.c_[datadic['labels'], -1*n.ones((1, self.minibatch_size - datadic['labels'].shape[1] % self.minibatch_size), dtype=n.single)], requirements='C', dtype=n.single)
                
            datadic['processed'] = True

        return epoch, batchnum, datadic['num_real_cases'], [datadic['data'], datadic['labels']]

    def get_data_dims(self, idx=0):
        return 3072 if idx == 0 else 1
    
class DummyConvNetDataProvider(LabeledDummyDataProvider):
    def __init__(self, data_dim):
        LabeledDummyDataProvider.__init__(self, data_dim)
        
    def get_next_batch(self):
        epoch, batchnum, dic = LabeledDummyDataProvider.get_next_batch(self)
        
        dic['data'] = n.require(dic['data'].T, requirements='C')
        dic['labels'] = n.require(dic['labels'].T, requirements='C')
        dic['num_real_cases'] = dic['data'].shape[1]
        
        return epoch, batchnum, dic

    def get_data_dims(self, idx=0):
        return self.batch_meta['num_vis'] if idx == 0 else 1