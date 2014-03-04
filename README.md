creg
====

Fast regression modeling framework.

Building
--------
To build `creg`, you will need:

* Download the [2014-02-18 version of the source code](http://demo.clab.cs.cmu.edu/cdec/creg-2014-02-18.tar.gz) (or install from github).
* If you are working with the code from github
  * Run `autoreconf -fvi`
* If you are working with a source tarball
  * Run `tar xzf creg-2014-02-18.tar.gz`
  * Change into the directory `cd creg-2014-02-18`
* Last, run the `./configure` command and then type `make`.

Software requirements:

* A relatively recent C++ compiler ([g++](http://gcc.gnu.org/) or [clang](http://clang.llvm.org/))
* The [Boost C++ libraries](http://www.boost.org) installed somewhere

Examples
--------

Logistic regression example (training only):

	./creg/creg -x test_data/iris.trainfeat -y test_data/iris.trainresp --l1 1.0 > weights.txt

  * To load initial values for weights from a file (warm start), use `-w FILENAME`.

Logistic regression example (training and testing):

	./creg/creg -x test_data/iris.trainfeat -y test_data/iris.trainresp --l1 1.0 \
	     --tx test_data/iris.testfeat --ty test_data/iris.testresp > weights.txt

Logistic regression example (training and prediction):

	./creg/creg -x test_data/iris.trainfeat -y test_data/iris.trainresp --l1 1.0 --tx test_data/iris.testfeat

  * By default, the test set predictions and learned weights are written to stdout.
  * If `-D` is specified, the full posterior distribution over predicted labels will be written.
  * To write weights to a file instead of stdout, specify `--z FILENAME`. To suppress outputting of weights altogether, supply the `-W` flag.

Linear regression example (training and testing):

	./creg/creg -n -x test_data/auto-mpg.trainfeat -y test_data/auto-mpg.trainresp --l2 1000 \
	     --tx test_data/auto-mpg.testfeat --ty test_data/auto-mpg.testresp > weights.txt

Ordinal regression example (training and testing)

	./creg/creg -o -x test_data/shuttle.trainfeat -y test_data/shuttle.trainresp \
	    --tx test_data/shuttle.testfeat --ty test_data/shuttle.testresp > weights.txt

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
