from libc.math cimport log

cdef class OrdinalDataset(Dataset):

    cdef readonly set levels

    def __init__(self, data):
        """
        Dataset with categorical response
        data: iterator of (dict, int) pairs
        """
        self.levels = set()
        super(OrdinalDataset, self).__init__(data, True)
        assert self.levels == set(range(len(self.levels)))

    def get_label(self, response):
        level = int(response)
        self.levels.add(level)
        return level

    def get_response(self, label):
        return label


cdef class OrdinalRegressionWeights(Weights):
    
    cdef OrdinalRegression model

    def __cinit__(self, OrdinalRegression model):
        self.model = model

    def __len__(self):
        cdef unsigned K = len(self.model.data.levels)
        return num_features() + K - 2

    def __iter__(self):
        cdef unsigned K = len(self.model.data.levels)
        cdef unsigned k
        for k in range(1, K):
            yield Intercept(k), self.model.weight_vector[0][k-1]
        cdef double fval
        cdef unsigned f
        for f in range(1, num_features()):
            fval = self.model.weight_vector[0][K-1+f]
            fname = unicode(Convert(f).c_str(), 'utf8')
            yield fname, fval

    def __getitem__(self, fname):
        cdef unsigned K = len(self.model.data.levels)
        cdef unsigned u = (fname.level-1 if isinstance(fname, Intercept) 
            else K-1+Convert(<char*> fname))
        return self.model.weight_vector[0][u]


cdef class OrdinalRegression(Model):

    cdef OrdinalLogLoss* loss
    cdef OrdinalDataset data

    def __dealloc__(self):
        if self.loss != NULL:
            del self.loss

    property weights:
        def __get__(self):
            assert (self.weight_vector.size() > 0)
            return OrdinalRegressionWeights(self)

    def fit(self, OrdinalDataset data,
            double l1=0, double l2=0, unsigned memory_buffers=40,
            double epsilon=1e-4, double delta=0):
        self.data = data
        cdef unsigned K = len(data.levels)
        cdef unsigned k
        if self.loss == NULL:
            self.weight_vector.resize(K - 1 + num_features(), 0.0)
            for k in range(K-1):
              self.weight_vector[0][k] = log(k+1) - log(K)
        else:
            del self.loss
        self.loss = new OrdinalLogLoss(data.instances[0], K, num_features(), l2)
        LearnParameters(self.loss[0], l1, K-1, memory_buffers,
            epsilon, delta, self.weight_vector)

    def predict(self, OrdinalDataset test):
        assert (self.loss != NULL)
        cdef unsigned i
        for i in range(test.instances.size()):
            yield self.loss.Predict(test.instances[0][i].x, self.weight_vector[0])

    def evaluate(self, OrdinalDataset data):
        """ Returns accuracy of the predictions for the dataset"""
        assert (self.loss != NULL)
        return self.loss.Evaluate(data.instances[0], self.weight_vector[0])
