from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.vector cimport vector

cdef extern from "creg/fdict.h" namespace "FD":
    int NumFeats()
    string Convert(int)
    int Convert(char*)
    void Freeze()

cdef extern from "creg/fast_sparse_vector.h":
    cdef cppclass SparseVector[T]:
        SparseVector(pair[int, T]* first, pair[int, T]* last)
        unsigned size()

cdef extern from "creg/creg.cc":
    cdef union InstanceValue:
        unsigned label
        float value

    cdef cppclass TrainingInstance:
        SparseVector[float] x
        InstanceValue y

    cdef cppclass BaseLoss:
        BaseLoss()

        double Evaluate(vector[TrainingInstance] test, vector[double] w)

    cdef cppclass MulticlassLogLoss(BaseLoss):
        MulticlassLogLoss(
          vector[TrainingInstance] tr,
          unsigned k,
          unsigned numfeats,
          double l2)

        unsigned Predict(SparseVector[float] fx, vector[double] w)

    cdef cppclass OrdinalLogLoss(BaseLoss):
        OrdinalLogLoss(
          vector[TrainingInstance] tr,
          unsigned k,
          unsigned numfeats,
          double l2)

        unsigned Predict(SparseVector[float] fx, vector[double] w)

    cdef cppclass UnivariateSquaredLoss(BaseLoss):
        UnivariateSquaredLoss(
          vector[TrainingInstance] tr,
          unsigned numfeats,
          double l2)

        double Predict(SparseVector[float] fx, vector[double] w)

    double LearnParameters(BaseLoss loss,
            double l1,
            unsigned l1_start,
            unsigned memory_buffers,
            double epsilon,
            double delta,
            vector[double]* px)
