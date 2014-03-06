import sys
import json
from sklearn import preprocessing
from sklearn import feature_extraction
from iologreg import IOLogisticRegression

features = []
labels = {}
invlabels = {}
# read labels and associated features
for line in open(sys.argv[1]):
  (label, f) = line.strip().split('\t')
  invlabels[len(labels)] = label
  labels[label] = len(labels)
  features.append(json.loads(f))
label_dict = feature_extraction.DictVectorizer()
label_features = label_dict.fit_transform(features).toarray()

sys.stderr.write('        LABELS: %s\n' % ' '.join(labels.keys()))
sys.stderr.write('LABEL-FEATURES: %s\n' % ' '.join(label_dict.get_feature_names()))
out_dim = len(label_dict.get_feature_names())

ids = {}
X = []
N = []
# read training instances and neighborhoods
for line in open(sys.argv[2]):
  (id, xfeats, n) = line.strip().split('\t')
  ids[id] = len(ids)
  X.append(json.loads(xfeats))
  neighborhood = json.loads(n)['N']
  if len(neighborhood) == 0:
    sys.stderr.write('[ERROR] empty neighborhood in line:\n%s' % line)
    sys.exit(1)
  if len(neighborhood) == 1:
    sys.stderr.write('[WARNING] neighborhood for id="%s" is singleton: %s\n' % (id, str(neighborhood)))
  n = [labels[x] for x in neighborhood]
  N.append(n)
X_dict = feature_extraction.DictVectorizer()
X = X_dict.fit_transform(X).toarray()

sys.stderr.write('       rows(X): %d\n' % len(X))
sys.stderr.write('INPUT-FEATURES: %s\n' % ' '.join(X_dict.get_feature_names()))
in_dim = len(X_dict.get_feature_names())

# read gold labels
Y = [0 for x in xrange(len(X))]
for line in open(sys.argv[3]):
  (id, y) = line.strip().split('\t')
  Y[ids[id]] = labels[y]

assert len(X) == len(N)
assert len(Y) == len(X)

model = IOLogisticRegression()
model.fit(in_dim, out_dim, X, N, Y, label_features, len(labels), iterations = 1000, minibatch_size=10)

D = model.predict_proba(X, N)
for row in D:
  dist = {}
  for i in range(len(row)):
    if row[i] > 0.0: dist[invlabels[i]] = row[i]
  print dist

