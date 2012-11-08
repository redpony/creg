import collections

cdef class CategoricalDataset(Dataset):

    cdef readonly list labels
    cdef readonly dict label_map

    def __init__(self, data):
        """
        CategoricalDataset(data) -> dataset with categorical response
        data: iterator of (dict, *) pairs
        """
        self.labels = []
        self.label_map = {}
        super(CategoricalDataset, self).__init__(data, True)

    def _get_label(self, response):
        if response in self.label_map:
            label = self.label_map[response]
        else:
            label = self.label_map.setdefault(response, len(self.label_map))
            self.labels.append(response)
        return label

    def _get_response(self, label):
        return self.labels[label]


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
            cdef unsigned y, f
            cdef unsigned K = len(self.data.labels)
            for y in range(K-1):
                label = self.data.labels[y]
                ret_weights[label] = {BIAS: self.weight_vector[0][y]}
                for f in range(1, num_features()):
                    w = self.weight_vector[0][(K-1) + y * num_features() + f]
                    fname = unicode(Convert(f).c_str(), 'utf8')
                    ret_weights[label][fname] = w
            return ret_weights

    def fit(self, CategoricalDataset data,
            double l1=0, double l2=0, unsigned memory_buffers=40,
            double epsilon=1e-4, double delta=0):
        """
        fit(CategoricalDataset data, l1=0, l2=0, memory_buffers=40, epsilon=1e-4, delta=0)
        Fit an logistic regression model on the training data.
        l1: L1 regularization strength
        l2: L2 regularization strength
        memory_buffers: number of memory buffers for LBFGS
        epsilon: convergence threshold for termination criterion: ||g|| < epsilon * max(1, ||w||))
        delta: convergence threshold for termination criterion (f' - f) / f < delta
        """
        self.data = data
        cdef unsigned K = len(data.labels)
        if self.loss == NULL:
            self.weight_vector.resize((1 + num_features()) * (K - 1), 0.0)
        else:
            del self.loss
        self.loss = new MulticlassLogLoss(data.instances[0], K, num_features(), l2)
        LearnParameters(self.loss[0], l1, K-1, memory_buffers,
            epsilon, delta, self.weight_vector)

    def _predict_dataset(self, CategoricalDataset test):
        cdef unsigned y
        for i in range(len(test)):
            y = self.loss.Predict(test.instances[0][i].x, self.weight_vector[0], NULL).first
            yield self.data.labels[y]

    def _predict_features(self, fmap):
        cdef vector[pair[int, float]]* test_vector = feature_vector(fmap)
        cdef unsigned y = self.loss.Predict(test_vector[0], self.weight_vector[0], NULL).first
        del test_vector
        return self.data.labels[y]

    def predict(self, test):
        """
        predict(CategoricalDataset) -> iterator of predictions
        predict(mapping) -> predicted label
        """
        assert (self.loss != NULL)
        if isinstance(test, CategoricalDataset):
            return self._predict_dataset(test)
        elif isinstance(test, collections.Mapping):
            return self._predict_features(test)
        else:
            raise TypeError('test has to be a CategoricalDataset or a mapping')

    def _predict_proba_dataset(self, CategoricalDataset test):
        cdef unsigned y
        cdef MulticlassPrediction* pred = new MulticlassPrediction()
        for i in range(len(test)):
            self.loss.Predict(test.instances[0][i].x, self.weight_vector[0], pred)
            yield pred.posterior
        del pred

    def _predict_proba_features(self, fmap):
        cdef vector[pair[int, float]]* test_vector = feature_vector(fmap)
        cdef MulticlassPrediction* pred = new MulticlassPrediction()
        self.loss.Predict(test_vector[0], self.weight_vector[0], pred)
        predictions = list(pred.posterior)
        del test_vector, pred
        return predictions

    def predict_proba(self, test):
        """
        predict_proba(CategoricalDataset) -> iterator of posterior probabilities
        predict_proba(mapping) -> posterior label probabilities
        """
        assert (self.loss != NULL)
        if isinstance(test, CategoricalDataset):
            return self._predict_proba_dataset(test)
        elif isinstance(test, collections.Mapping):
            return self._predict_proba_features(test)
        else:
            raise TypeError('test has to be a CategoricalDataset or a mapping')

    def evaluate(self, CategoricalDataset data, double thresh_p=0):
        """
        evaluate(CategoricalDataset, thresh_p=0) -> accuracy of the predictions for the dataset
        thresh_p: only compute the accuracy on instances where the predicted class posterior
                  is greater than this value
        """
        assert (self.loss != NULL)
        return self.loss.Evaluate(data.instances[0], self.weight_vector[0], thresh_p)

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
                    f = Convert(as_str(fname))
                    assert (f < num_features)
                    u = (K-1) + y * num_features + f
                self.weight_vector[0][u] = fval
        Freeze()
        self.loss = new MulticlassLogLoss(self.data.instances[0], K, num_features, 0)
