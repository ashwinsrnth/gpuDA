from theano import shared
import numpy as np
import pygpu

class TheanoNDArray:
    def __init__(self, size):        
        self.size = size
        self.vec = shared(np.empty(size))

    def set(self, values):
        self.vec.set_value(values)

    def set_value(self, value):
        self.vec.set_value(np.ones(self.size)*value)

    def get(self):
        return self.vec.get_value()

    def getGPU(self):
        return self.vec.container.data

    @property
    def itemsize(self):
        return self.vec.container.data.dtype.itemsize

    @property
    def shape(self):
        return self.vec.container.data.shape

    @property
    def nbytes(self):
        return self.vec.container.data.size*self.vec.container.data.dtype.itemsize

    @property
    def strides(self):
        return self.vec.container.data.strides

    @property
    def gpudata(self):
        try:
            return pygpu.gpuarray.get_raw_ptr(self.vec.container.data.gpudata)
        except:
            raise AttributeError('can''t get gpudata (are you using the GPU?)')
