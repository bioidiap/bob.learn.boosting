=============================================================================
Generalized Boosting Framework using Stump and Look Up Table (LUT) based Weak Classifier
=============================================================================
The package implements a generalized boosting framework which incorporate different
boosting approaches. The Boosting algorithms implemented in this package are

1) Gradient Boost (generalized version of Adaboost) for univariate cases
2) TaylorBoost for univariante and multivariate cases

The weak classfiers associated with these boosting algorithms are 

1) Stump classifiers
2) LUT based classfiers

Check the following reference for the details: 

1. Viola, Paul, and Michael J. Jones. "Robust real-time face detection." 
International journal of computer vision 57.2 (2004): 137-154.

2. Saberian, Mohammad J., Hamed Masnadi-Shirazi, and Nuno Vasconcelos. "Taylorboost: 
First and second-order boosting algorithms with explicit margin control." Computer 
Vision and Pattern Recognition (CVPR), 2011 IEEE Conference on. IEEE, 2011.

3. Cosmin Atanasoaei, "Multivariate Boosting with Look Up Table for face processing",
PhD thesis (2012).

Testdata:
----------

The test are performed on the MNIST digits dataset. The tests can be mainly divided into
two categories:

1) Univariate Test: It corresponds to binary classification problem. The digits are tested 
one-vs-one and one-vs-all. Both the boosting algorithm (Gradient Boost and Taylor boost)
can be used for testing this scenario.

2) Multivariate Test: It is the multi class classification problem. All the 10 digit classfication
is considered in a single test. Only Multivariate Taylor boosting can be used for testing this scenario.

Installation:
----------

Once you have downloaded the package use the following two commands to install it:

  $ python bootstrap.py 

  $ ./bin/buildout

These 2 commands should download and install all non-installed dependencies and
get you a fully operational test and development environment.

User Guide
----------

This section explains how to use the package in order to: a) test the MNIST dataset for binary clssification
b) test the dataset for multi class classification.

a) The following command will run a single binary test for the digits specified and display the classifcation 
accuracy on the console:

  $ ./bin/mnist_binary_one.py 

if you want to see all the option associated with the command type:

  $ ./bin/mnist_binary_one.py -h

To run the tests for all the combination of of ten digits use the following command:

  $ ./bin/mnist_binary_all.py 

This command tests all the possible comniation of digits which results in 45 different binary tests. The 
accuracy of individual tests and the final average accuracy of all the tests is displayed on the console.

b) The following command can be used for the multivarite digits test:

  $ ./bin/mnist_multi.py 

Because of large number of samples and multivariate problem it requires times in days on a normal system. Use -h 
option to see different option available with this command.  


