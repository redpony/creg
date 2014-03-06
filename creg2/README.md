
This directory contains some experimental learning code for logistic regression on structured label spaces.

It is possible to
 * define features of inputs and outputs seperately, the classifier will operate on the outer product space.
 * target an empirical distribution over labels, rather than a single gold standard label
 * specify different output spaces for each training instance.

Usage

The inputs are the following:
 * `labels.feat` defines the feature map for the output space; providing an indepedent binary feature for each output reduces the problem to familiar multiclass logistic regression
 * `train.feat` defines the feature maps for the input data and the discriminative neighborhoods for each training instance
 * `train.resp` defines the response variable or the response distribution

Example invocation:

    python creg2.py test_data/iris/labels.feat test_data/iris/iris.trainfeat test_data/iris/iris.trainresp

