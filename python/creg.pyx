from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcreg cimport *
import heapq

cdef extern from *:
    ctypedef char* const_char_ptr "const char*"

cdef SparseVector[float]* fvector(features):
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

cdef unsigned num_features():
    return NumFeats()

cdef class Dataset:

    cdef vector[TrainingInstance]* instances

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

    def __str__(self):
        return '<Dataset: %d instances, %d features>' %\
                (self.instances[0].size(), num_features())
        
cdef class CategoricalDataset(Dataset):

    cdef readonly list labels
    cdef dict label_map

    def __init__(self, data):
        """
        Dataset with categorical response
        data: iterator of (dict, *) pairs
        """
        self.labels = []
        self.label_map = {}
        super(CategoricalDataset, self).__init__(data, True)

    def get_label(self, response):
        if response in self.label_map:
            label = self.label_map[response]
        else:
            label = self.label_map.setdefault(response, len(self.label_map))
            self.labels.append(response)
        return label
        
cdef class RealvaluedDataset(Dataset):

    def __init__(self, data):
        """
        Dataset with real-valued response
        data: iterator of (dict, float) pairs
        """
        super(RealvaluedDataset, self).__init__(data, False)


cdef class Weights:

    def __len__(self):
        return num_features()

    def top(self, k):
        return heapq.nlargest(k, self, key=lambda kv: abs(kv[1]))

    property df:
        def __get__(self):
            return sum(1 for _, w in self if w != 0)

    def __str__(self):
        return '<Weights: %d values, %d non-zero>' % (len(self), self.df)

cdef class LinearRegressionWeights(Weights):
    
    cdef LinearRegression model

    def __cinit__(self, LinearRegression model):
        self.model = model

    def __iter__(self):
        yield BIAS, self.model.weight_vector[0][0]
        cdef double fval
        cdef unsigned f
        cdef const_char_ptr fname
        for f in range(1, num_features()):
            fval = self.model.weight_vector[0][1+f]
            fname = Convert(f).c_str()
            yield fname.decode('utf8'), fval

    def __getitem__(self, char* fname):
        cdef unsigned u = (0 if fname == BIAS else 1+Convert(<char*> fname))
        return self.model.weight_vector[0][u]

cdef class Model:

    cdef vector[double]* weight_vector

    def __cinit__(self):
        self.weight_vector = new vector[double]()

    def __dealloc__(self):
        del self.weight_vector

BIAS = '***BIAS***'

cdef class LogisticRegression(Model):

    cdef MulticlassLogLoss* loss
    cdef CategoricalDataset data

    def __dealloc__(self):
        if self.loss != NULL:
            del self.loss

    property weights:
        def __get__(self):
            assert (self.weight_vector.size() > 0)
            ret_weights = {}
            cdef double w
            cdef const_char_ptr fname
            cdef unsigned y, f
            cdef unsigned K = len(self.data.labels)
            for y in range(K-1):
                label = self.data.labels[y]
                ret_weights[label] = {BIAS: self.weight_vector[0][y]}
                for f in range(1, num_features()):
                    w = self.weight_vector[0][(K-1) + y * num_features() + f]
                    fname = Convert(f).c_str()
                    ret_weights[label][fname.decode('utf8')] = w
            return ret_weights

    def fit(self, CategoricalDataset data,
            double l1=0, double l2=0, unsigned memory_buffers=40,
            double epsilon=1e-4, double delta=0):
        self.data = data
        cdef unsigned K = len(data.labels)
        if self.loss == NULL:
            self.weight_vector.resize((1 + num_features()) * (K - 1), 0.0)
        else:
            del self.loss
        self.loss = new MulticlassLogLoss(data.instances[0], K, num_features(), l2)
        LearnParameters(self.loss[0], l1, K-1, memory_buffers, epsilon, delta, self.weight_vector)

    def predict(self, features):
        assert (self.loss != NULL)
        cdef vector[double] dotprods
        cdef unsigned K = len(self.data.labels)
        dotprods.resize(K - 1, 0.0)
        cdef SparseVector[float]* fv = fvector(features)
        self.loss.ComputeDotProducts(fv[0], self.weight_vector[0], &dotprods);
        cdef double best = 0
        cdef unsigned y, besty = dotprods.size()
        for y in range(dotprods.size()):
            if dotprods[y] > best:
                best = dotprods[y]
                besty = y
        del fv
        return self.data.labels[besty]

    def evaluate(self, CategoricalDataset data):
        """ Returns accuracy of the predictions for the dataset"""
        assert (self.loss != NULL)
        return self.loss.Evaluate(data.instances[0], self.weight_vector[0])

    def _load(self, labels, unsigned num_features, weights):
        assert (self.loss == NULL)
        self.data = CategoricalDataset([])
        # Initialize labels
        self.data.labels = list(labels)
        for (lid, label) in enumerate(self.data.labels):
            self.data.label_map[label] = lid
        cdef unsigned K = len(self.data.labels)

        # Initialize weights
        self.weight_vector.resize((1 + num_features) * (K - 1), 0.0)
        cdef unsigned y, f, u
        for label, label_weights in weights.iteritems():
            y = self.data.label_map[label]
            for fname, fval in label_weights.iteritems():
                if fname == BIAS:
                    u = y
                else:
                    fname = fname.encode('utf8')
                    f = Convert(<char*> fname)
                    assert (f < num_features)
                    u = (K-1) + y * num_features + f
                self.weight_vector[0][u] = fval
        Freeze()
        self.loss = new MulticlassLogLoss(self.data.instances[0], K, num_features, 0)


cdef class LinearRegression(Model):

    cdef UnivariateSquaredLoss* loss

    def __dealloc__(self):
        if self.loss != NULL:
            del self.loss

    property weights:
        def __get__(self):
            assert (self.weight_vector.size() > 0)
            return LinearRegressionWeights(self)

    def fit(self, RealvaluedDataset data,
            double l1=0, double l2=0, unsigned memory_buffers=40,
            double epsilon=1e-4, double delta=1e-5):
        if self.loss == NULL:
            self.weight_vector.resize((1 + num_features()), 0.0)
        else:
            del self.loss
        self.loss = new UnivariateSquaredLoss(data.instances[0], num_features(), l2)
        LearnParameters(self.loss[0], l1, 1, memory_buffers, epsilon, delta, self.weight_vector)

    def predict(self, features):
        assert (self.loss != NULL)
        cdef vector[double] dotprods
        dotprods.resize(1, 0.0)
        cdef SparseVector[float]* fv = fvector(features)
        self.loss.ComputeDotProducts(fv[0], self.weight_vector[0], &dotprods);
        cdef double value = dotprods[0]
        del fv
        return value

    def evaluate(self, RealvaluedDataset data):
        """ Returns RMSE of the predictions for the dataset"""
        assert (self.loss != NULL)
        return self.loss.Evaluate(data.instances[0], self.weight_vector[0])

    def _load(self, unsigned num_features, weights):
        assert (self.loss == NULL)
        # Initialize weights
        self.weight_vector.resize(1 + num_features, 0.0)
        cdef unsigned u
        for fname, fval in weights.iteritems():
            fname = fname.encode('utf8')
            u = (0 if fname == BIAS else 1+Convert(<char*> fname))
            self.weight_vector[0][u] = fval
        Freeze()
        cdef vector[TrainingInstance] instances
        self.loss = new UnivariateSquaredLoss(instances, num_features, 0)
