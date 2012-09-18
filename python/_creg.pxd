from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "creg/fdict.h" namespace "FD":
    int NumFeats()
    string& Convert(int)
    int Convert(char*)
    void Freeze()

cdef extern from "creg/feature_map.h":
    cdef cppclass const_pair_int_float "const std::pair<int,float>":
        int first
        float second

    cdef cppclass FrozenFeatureMap:
        const_pair_int_float* begin()
        const_pair_int_float* end()

    cdef cppclass FeatureMapStorage:
        FrozenFeatureMap AddFeatureMap(pair[int, float]* begin,
                pair[int, float]* end)

cdef extern from "creg/creg.cc":
    cdef union InstanceValue:
        unsigned label
        float value

    cdef cppclass TrainingInstance:
        FrozenFeatureMap x
        InstanceValue y

    cdef cppclass BaseLoss:
        BaseLoss()

    cdef cppclass MulticlassLogLoss(BaseLoss):
        MulticlassLogLoss(
          vector[TrainingInstance]& tr,
          unsigned k,
          unsigned numfeats,
          double l2) # double t = 0.0

        pair[unsigned, double] Predict(FrozenFeatureMap& fx, vector[double]& w)
        double Evaluate(vector[TrainingInstance]& test, vector[double]& w, double thresh_p)

    cdef cppclass OrdinalLogLoss(BaseLoss):
        OrdinalLogLoss(
          vector[TrainingInstance]& tr,
          unsigned k,
          unsigned numfeats,
          double l2)

        unsigned Predict(FrozenFeatureMap& fx, vector[double]& w)
        double Evaluate(vector[TrainingInstance]& test, vector[double]& w)

    cdef cppclass UnivariateSquaredLoss(BaseLoss):
        UnivariateSquaredLoss(
          vector[TrainingInstance]& tr,
          unsigned numfeats,
          double l2)

        double Predict(FrozenFeatureMap& fx, vector[double]& w)
        double Evaluate(vector[TrainingInstance]& test, vector[double]& w)

    double LearnParameters(BaseLoss loss,
            double l1,
            unsigned l1_start,
            unsigned memory_buffers,
            double epsilon,
            double delta,
            vector[double]* px)

cdef extern from *:
    ctypedef char* const_char_ptr "const char*"

cdef inline unsigned num_features():
    return NumFeats()

"""
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
"""
