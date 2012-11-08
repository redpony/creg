import heapq
from cython.operator import preincrement as pinc

cdef class Bias:
    def __repr__(self):
        return '***BIAS***'

BIAS = Bias()

cdef class Intercept:
    cdef unsigned level

    def __cinit__(self, level):
        self.level = level

    def __repr__(self):
        return 'y>=%d' % self.level

cdef char* as_str(name):
    if isinstance(name, bytes):
        return name
    elif isinstance(name, unicode):
        name = name.encode('utf8')
        return name
    raise TypeError('Cannot convert %s to string.' % type(name))

cdef vector[pair[int, float]]* feature_vector(fmap):
    cdef vector[pair[int, float]]* fvector = new vector[pair[int, float]]()
    cdef pair[int, float]* fpair
    for key in fmap:
        fpair = new pair[int, float](Convert(as_str(key)), fmap[key])
        fvector.push_back(fpair[0])
        del fpair
    return fvector

cdef class Dataset:
    cdef vector[TrainingInstance]* instances
    cdef FeatureMapStorage* fms
    cdef bint categorical

    def __cinit__(self):
        self.instances = new vector[TrainingInstance]()
        self.fms = new FeatureMapStorage()

    def __init__(self, data, categorical):
        self.categorical = categorical
        cdef TrainingInstance* instance
        cdef pair[int, float] *fpair, *front, *back
        cdef vector[pair[int, float]]* featmap
        for features, response in data:
            instance = new TrainingInstance()
            if categorical:
                instance.y.label = self._get_label(response)
            else:
                instance.y.value = response
            featmap = feature_vector(features)
            front = &featmap.front()
            back = &featmap.back()+1
            instance.x = self.fms.AddFeatureMap(front, back)
            del featmap
            self.instances.push_back(instance[0])
            del instance
        Freeze()

    def __dealloc__(self):
        del self.instances
        del self.fms

    property num_features:
        def __get__(self):
            return num_features()

    def __repr__(self):
        return '<Dataset: %d instances, %d features>' %\
                (len(self), num_features())

    def __iter__(self):
        cdef unsigned i
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return self.instances.size()

    def __getitem__(self, int i):
        if not 0 <= i < len(self):
            raise KeyError('training set index out of range')
        cdef TrainingInstance* instance = &self.instances[0][i]
        y = self._get_response(instance.y.label) if self.categorical else instance.y.value
        x = {}
        cdef const_pair_int_float* xptr = instance.x.begin()
        cdef int f
        cdef float fval
        while xptr != instance.x.end():
            f = xptr[0].first
            fval = xptr[0].second
            fname = unicode(Convert(f).c_str(), 'utf8')
            x[fname] = fval
            pinc(xptr)
        return (x, y)

cdef class Weights:
    def __len__(self):
        return num_features()

    def top(self, k):
        """
        top(k) -> top k largest weights in absolute value
        """
        return heapq.nlargest(k, self, key=lambda kv: abs(kv[1]))

    property df:
        def __get__(self):
            return sum(1 for _, w in self if w != 0)

    def __repr__(self):
        return '<Weights: %d values, %d non-zero>' % (len(self), self.df)


cdef class Model:
    cdef vector[double]* weight_vector

    def __cinit__(self):
        self.weight_vector = new vector[double]()

    def __dealloc__(self):
        del self.weight_vector

include "linear.pxi"
include "logistic.pxi"
include "ordinal.pxi"
