creg
====

Fast regression modeling framework.

Building
--------
You wil need:
* A C++ compiler (g++)
* The [boost libraries](http://www.boost.org) installed somewhere

Instructions:

1. Type `./bjam`

Examples
--------

Logistic regression example (training only):

	$ ./dist/bin/creg -x test_data/iris.trainfeat -y test_data/iris.trainresp --l1 1.0 > weights.txt

Logistic regression example (training and testing):

	$ ./dist/bin/creg -x test_data/iris.trainfeat -y test_data/iris.trainresp --l1 1.0 \
	     -t test_data/iris.testfeat -s test_data/iris.testresp > weights.txt

Linear regression example (training and testing):

	$ ./dist/bin/creg -n -x test_data/auto-mpg.trainfeat -y test_data/auto-mpg.trainresp --l2 1000 \
	     -t test_data/auto-mpg.testfeat -s test_data/auto-mpg.testresp > weights.txt``

Ordinal regression example

	$ ./creg -o -x test_data/shuttle.trainfeat -y test_data/shuttle.trainresp

