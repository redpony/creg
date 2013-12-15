import sys
import json
import argparse
import numpy as np
from numpy.linalg import svd

parser = argparse.ArgumentParser(description='Do dimensionality reduction on matrix of inputs.')
parser.add_argument('-n', type=int, default=2,
                   help='Number of features to generate')
parser.add_argument('training_json_file', nargs=1,
                   help='JSON feature map for training set')
parser.add_argument('test_json_files', nargs='*',
                   help='JSON feature map for test set')

args = parser.parse_args()

jd=json.JSONDecoder()
je=json.JSONEncoder()

sys.stderr.write('Reading training instances...\n')
dicts = {}
keys = {}
with open(args.training_json_file[0]) as f:
  for line in f:
    (key, dict) = line.strip().split('\t')
    dict = jd.decode(dict)
    dicts[key] = dict
    for k in dict.keys():
      keys[k] = 1

sys.stderr.write('Creating matrix...\n')
X = np.zeros((len(dicts), len(keys)))

r = 0
for (key, dict) in dicts.items():
  c = 0
  for k in keys.keys():
    v = 0.0
    if k in dict: v = dict[k]
    X[r][c] = v
    c += 1
  r += 1

sys.stderr.write('Doing SVD...\n')
(U,S,V) = svd(X)
sys.stderr.write('Eigenvalues: {}\n'.format(str(S)))
r = 0
outfile = args.training_json_file[0] + '.svd'
sys.stderr.write('Writing {}...\n'.format(outfile))
with open(outfile, 'w') as f:
  for (key, dict) in dicts.items():
    od = {}
    #mU = V.dot(X[r])
    for c in range(args.n):
      k = 'Feature_{}'.format(c)
      od[k] = U[r][c]      # this is equal to mU[c] / S[c]
      # or maybe we should scale by the eigenvalue? this will change the behavior
      # of the regularizer, but have no other effects
    f.write('{}\t{}\n'.format(key, je.encode(od)))
    r += 1

x = np.zeros((len(keys),))
for tname in args.test_json_files:
  outfile = tname + '.svd'
  with open(tname) as f:
    sys.stderr.write('Writing {}...\n'.format(outfile))
    with open(outfile, 'w') as o:
      for line in f:
        (key, dict) = line.strip().split('\t')
        dict = jd.decode(dict)
        c = 0
        x *= 0.0
        for k in keys.keys():
          if k in dict:
            x[c] = dict[k]
          c += 1
        u = V.dot(x)
        od = {}
        #mU = V.dot(X[r])
        for c in range(args.n):
          k = 'Feature_{}'.format(c)
          od[k] = u[c] / S[c]
        o.write('{}\t{}\n'.format(key, je.encode(od)))
