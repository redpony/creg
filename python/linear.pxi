cdef class RealvaluedDataset(Dataset):

    def __init__(self, data):
        """
        RealvaluedDataset(data) -> dataset with real-valued response
        data: iterator of (dict, float) pairs
        """
        super(RealvaluedDataset, self).__init__(data, False)
 

cdef class LinearRegressionWeights(Weights):
    
    cdef LinearRegression model

    def __cinit__(self, LinearRegression model):
        self.model = model

    def __iter__(self):
        yield BIAS, self.model.weight_vector[0][0]
        cdef double fval
        cdef unsigned f
        for f in range(1, num_features()):
            fval = self.model.weight_vector[0][1+f]
            fname = unicode(Convert(f).c_str(), 'utf8')
            yield fname, fval

    def __getitem__(self, fname):
        cdef unsigned u = (0 if fname == BIAS else 1+Convert(<char*> fname))
        return self.model.weight_vector[0][u]


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
            double epsilon=1e-4, double delta=0):
        """
        fit(RealvaluedDataset data, l1=0, l2=0, memory_buffers=40, epsilon=1e-4, delta=0)
        Fit a linear regression model on the training data.
        l1: L1 regularization strength
        l2: L2 regularization strength
        memory_buffers: number of memory buffers for LBFGS
        epsilon: convergence threshold for termination criterion: ||g|| < epsilon * max(1, ||w||))
        delta: convergence threshold for termination criterion (f' - f) / f < delta
        """
        if self.loss == NULL:
            self.weight_vector.resize(1 + num_features(), 0.0)
        else:
            del self.loss
        self.loss = new UnivariateSquaredLoss(data.instances[0], num_features(), l2)
        LearnParameters(self.loss[0], l1, 1, memory_buffers,
            epsilon, delta, self.weight_vector)

    def _predict_dataset(self, RealvaluedDataset test):
        for i in range(len(test)):
            yield self.loss.Predict(test.instances[0][i].x, self.weight_vector[0])

    def _predict_features(self, fmap):
        cdef vector[pair[int, float]]* test_vector = feature_vector(fmap)
        cdef double y = self.loss.Predict(test_vector[0], self.weight_vector[0])
        del test_vector
        return y

    def predict(self, test):
        """
        predict(RealvaluedDataset) -> iterator of predictions
        predict(mapping) -> predicted value
        """
        assert (self.loss != NULL)
        if isinstance(test, RealvaluedDataset):
            return self._predict_dataset(test)
        elif isinstance(test, collections.Mapping):
            return self._predict_features(test)
        else:
            raise TypeError('test has to be a RealvaluedDataset or a mapping')

    def evaluate(self, RealvaluedDataset data):
        """
        evaluate(RealvaluedDataset) -> RMSE of the predictions for the dataset
        """
        assert (self.loss != NULL)
        return self.loss.Evaluate(data.instances[0], self.weight_vector[0])

    def _load(self, unsigned num_features, weights):
        assert (self.loss == NULL)
        # Initialize weights
        self.weight_vector.resize(1 + num_features, 0.0)
        cdef unsigned u
        for fname, fval in weights.iteritems():
            u = (0 if fname == BIAS else 1+Convert(as_str(fname)))
            self.weight_vector[0][u] = fval
        Freeze()
        cdef vector[TrainingInstance] instances
        self.loss = new UnivariateSquaredLoss(instances, num_features, 0)
