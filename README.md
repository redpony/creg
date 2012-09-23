creg
====

Fast regression modeling framework.

Building
--------
You wil need:

* A C++ compiler (g++)
* The [boost libraries](http://www.boost.org) installed somewhere

Run the `./bjam` command in the root source directory.


Examples
--------

Logistic regression example (training only):

	./dist/bin/creg -x test_data/iris.trainfeat -y test_data/iris.trainresp --l1 1.0 > weights.txt

Logistic regression example (training and testing):

	./dist/bin/creg -x test_data/iris.trainfeat -y test_data/iris.trainresp --l1 1.0 \
	     --tx test_data/iris.testfeat --ty test_data/iris.testresp > weights.txt

Logistic regression example (training and prediction):

	./dist/bin/creg -x test_data/iris.trainfeat -y test_data/iris.trainresp --l1 1.0 --tx test_data/iris.testfeat -W -D

*The `-W` option in preceeding example supresses writing the learned weights and the `-D` option causes the full posterior distribution over predicted labels to be written.*

Linear regression example (training and testing):

	./dist/bin/creg -n -x test_data/auto-mpg.trainfeat -y test_data/auto-mpg.trainresp --l2 1000 \
	     --tx test_data/auto-mpg.testfeat --ty test_data/auto-mpg.testresp > weights.txt

Ordinal regression example (training and testing)

	./dist/bin/creg -o -x test_data/shuttle.trainfeat -y test_data/shuttle.trainresp \
	    -t test_data/shuttle.testfeat -s test_data/shuttle.testresp > weights.txt

Note: for ordinal regression, labels have to be consecutive and start from 0 (e.g., 0/1/2 for 3 labels).

Data format
-----------

Training and evaluation data are expected to be in the following format:

* A feature file containing lines of the form:

    	id1\t{"feature1": 1.0, "feature2": -10}
    	id2\t{"feature2": 10,  "feature3": 2.53}

	where the JSON-like map defines a sparse feature vector for each instance

* A response file containing the same number of lines of the form:	

    	id1\t10.1
    	id2\t4.3

	where the response is numeric for linear and ordinal regression and a label for logistic regression

You will find example files for each type of model in the [test\_data](https://github.com/redpony/creg/tree/master/test_data) directory.

Python module
-------------

Quick install: 

    pip install -e git+https://github.com/redpony/creg.git#egg=creg

Some documentation is available [on the wiki](https://github.com/redpony/creg/wiki/Python-module).
