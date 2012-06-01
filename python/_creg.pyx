import heapq

include "linear.pxi"
include "logistic.pxi"
include "ordinal.pxi"

cdef class Bias:
    def __repr__(self):
        return '***BIAS***'

BIAS = Bias()

cdef class Intercept:
    def __cinit__(self, level):
        self.level = level

    def __repr__(self):
        return 'y>=%d' % self.level


cdef class Dataset:
    def __cinit__(self):
        self.instances = new vector[TrainingInstance]()

    def __init__(self, data, categorical):
        cdef TrainingInstance* instance
        cdef SparseVector[float]* fv
        for features, response in data:
            instance = new TrainingInstance()
            if categorical:
                instance.y.label = self.get_label(response)
            else:
                instance.y.value = response
            fv = fvector(features)
            instance.x = fv[0]
            self.instances.push_back(instance[0])
            del instance
            del fv
        Freeze()

    def __dealloc__(self):
        del self.instances

    property num_features:
        def __get__(self):
            return num_features()

    def __repr__(self):
        return '<Dataset: %d instances, %d features>' %\
                (self.instances[0].size(), num_features())
        

cdef class Weights:

    def __len__(self):
        return num_features()

    def top(self, k):
        return heapq.nlargest(k, self, key=lambda kv: abs(kv[1]))

    property df:
        def __get__(self):
            return sum(1 for _, w in self if w != 0)

    def __repr__(self):
        return '<Weights: %d values, %d non-zero>' % (len(self), self.df)


cdef class Model:

    def __cinit__(self):
        self.weight_vector = new vector[double]()

    def __dealloc__(self):
        del self.weight_vector
