from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcreg cimport *

cdef class Dataset:
    cdef vector[TrainingInstance]* instances

cdef class Model:
    cdef vector[double]* weight_vector

cdef class Weights:
    pass

cdef class Bias:
    pass

cdef class Intercept:
    cdef unsigned level

cdef inline unsigned num_features():
    return NumFeats()

cdef inline SparseVector[float]* fvector(features):
    cdef pair[int, float] *fpair, *front, *back
    cdef vector[pair[int, float]]* featmap = new vector[pair[int, float]]()
    for fname, fval in features.iteritems():
        fname = fname.encode('utf8')
        fpair = new pair[int, float](Convert(<char*> fname), fval)
        featmap.push_back(fpair[0])
        del fpair
    front = &featmap.front()
    back = &featmap.back()+1
    cdef SparseVector[float]* result = new SparseVector[float](front, back)
    del featmap
    return result
