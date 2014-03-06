import numpy as np
import random
import math
import sys

INFINITY = float('inf')

def logadd(a,b):
    """
    compute log(exp(a) + exp(b))
    """
    if a == -INFINITY:
        return b
    if b == -INFINITY:
        return a
    if b < a: # b - a < 0
        return a + math.log1p(math.exp(b - a))
    else: # a - b < 0
        return b + math.log1p(math.exp(a - b))

class IOLogisticRegression:
    """
    Logistic regression.
    Minimize regularized log-loss:
        L(x, y|w) = - sum_i log p(y_i|x_i, w) + l2 ||w||^2
        p(y|x, w) = exp(w[y].x) / (sum_y' exp(w[y'].x))

    Parameters
    ----------
    l2: float, default=0
        L2 regularization strength
    """
    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = l1
        self.l2 = l2

    def gradient(self, x, n, y, y_feats, W, G):
        z = -INFINITY
        log_probs = np.zeros(self.num_labels)
        xw = x.dot(W)
        found = False
        for yi in n:
            if yi == y: found = True
            u = xw.dot(y_feats[yi])
            log_probs[yi] = u
            z = logadd(z, u)
        if not found:
            print '[ERROR] for training instance', x, 'gold label', y, 'not found in neighborhood', n
            raise Exception
        loss = -(log_probs[y] - z)
        for yi in n:
            delta = math.exp(log_probs[yi] - z) - (yi == y)
            G += np.outer(x, y_feats[yi]) * delta
        return loss

    def fit(self, infeats, outfeats, X, N, Y, y_feats, num_labels, iterations=300, minibatch_size=1000, eta=1.0):
        minibatch_size = min(minibatch_size, len(X))
        self.num_labels = num_labels
        self.y_feats = y_feats
        self.W = np.zeros(shape=(infeats, outfeats))
        G = np.zeros(shape=(infeats, outfeats))
        H = np.ones(shape=(infeats, outfeats)) * 1e-300
        for i in range(iterations):
            sys.stderr.write('Iteration: %d\n' % i)
            G.fill(0.0)
            loss = 0
            for s in random.sample(range(X.shape[0]), minibatch_size):
                loss += self.gradient(X[s], N[s], Y[s], y_feats, self.W, G)

            #for k in range(self.n_classes - 1):
            #    offset = (self.n_features + 1) * k
            #    for j in range(self.n_features):
            #        loss += self.l2 * self.coef_[offset + j]**2
            #        g[offset + j] += 2 * self.l2 * self.coef_[offset + j]

            sys.stderr.write('  Loss = %f\n' % loss)
            G /= minibatch_size
            H += np.square(G)
            self.W -= np.divide(G, np.sqrt(H)) * eta
        return self

    def predict_(self, x, n, probs):
        probs.fill(0.0)
        z = -INFINITY
        xw = x.dot(self.W)
        for y in n:
            u = xw.dot(self.y_feats[y])
            probs[y] = u
            z = logadd(z, u)
        for y in n:
            probs[y] = math.exp(probs[y] - z)

    def predict(self, X, N):
        post = np.zeros(shape=(len(X),self.num_labels))
        return post

    def predict_proba(self, X, N):
        post = np.zeros(shape=(len(X),self.num_labels))
        for (x, n, p) in zip(X, N, post):
          self.predict_(x, n, p)
        return post
