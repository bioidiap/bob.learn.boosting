========================================================================================
Generalized Boosting Framework using Stump and Look Up Table (LUT) based Weak Classifier
========================================================================================

The package implements a generalized boosting framework, which incorporates different boosting approaches.
The Boosting algorithms implemented in this package are

1) Gradient Boost (generalized version of Adaboost) for univariate cases
2) TaylorBoost for univariate and multivariate cases

The weak classifiers associated with these boosting algorithms are

1) Stump classifiers
2) LUT based classifiers

Check the following reference for the details:

1. Viola, Paul, and Michael J. Jones. "Robust real-time face detection." International journal of computer vision 57.2 (2004): 137-154.

2. Saberian, Mohammad J., Hamed Masnadi-Shirazi, and Nuno Vasconcelos. "Taylorboost: First and second-order boosting algorithms with explicit margin control." Computer Vision and Pattern Recognition (CVPR), 2011 IEEE Conference on. IEEE, 2011.

3. Cosmin Atanasoaei, "Multivariate Boosting with Look Up Table for face processing", PhD thesis (2012).

Installation:
----------

Once you have downloaded the package use the following two commands to install it:

  $ python bootstrap.py

  $ ./bin/buildout

These 2 commands should download and install all non-installed dependencies and get you a fully operational test and development environment.


Example
-------
To show an exemplary usage of the boosting algorithm, the binary and multi-variate classification of hand-written digits from the MNIST database is performed.
For simplicity, we just use the pixel gray values as (discrete) features to classify the digits.
In each boosting round, a single pixel location is selected.
In case of the stump classifier, this pixel value is compared to a threshold (which is determined during training), and one of the two classes is assigned.
In case of the LUT, for each value of the pixel the most probable digit is determined.

The script ``./bin/boosting_example.py`` is provided to perform all different examples.
This script has several command line parameters, which vary the behavior of the training and/or testing procedure.
All parameters have a long value (starting with ``--``) and a shotcut (starting with a single ``-``).
These parameters are (see also ``./bin/boosting_example.py --help``):

To control the type of training, you can select:

* ``--trainer-type``: Select the type of weak classifier. Possible values are ``stump`` and ``lut``
* ``--loss-type``: Select the loss function. Possible values are ``tan``, ``log`` and ``exp``. By default, a loss function suitable to the trainer type is selected.
* ``--number-of-boosting-rounds``: The number of weak classifiers to select.
* ``--multi-variate`` (only valid for LUT trainer): Perform multi-vatriate classification, or binary (one-to-one) classification.
* ``--feature-selection-style`` (only valid for multi-variate training): Select the feature for each output ``independent``ly or ``shared``?

To control the experimentation, you can choose:

* ``--digits``: The digits to classify. For multi-variate training, one classifier is trained for all given digits, while for uni-variate training all possible one-to-one classifiers are trained.
* ``--all``: Select all 10 digits.
* ``--classifier-file``: Save the trained classifier(s) into the given file and/or read the classifier(s) from this file.
* ``--force``: Overwrite the given classifier file if it already exists.

For information and debugging purposes, it might be interesting to use:

* ``--verbose`` (can be used several times): Increases the verbosity level from 0 (error) over 1 (warning) and 2 (info) to 3 (debug). Verbosity level 2 (``-vv``) is recommended.
* ``number-of-elements``: Reduce the number of elements per class (digit) to the given value.

Four different kinds of experimentations can be performed:

1. Uni-variate classification using the stump trainer:

  $ ./bin/boosting_example.py -vv --trainer-type stump --digits 5 6 --classifier-file stump.hdf5

2. Uni-variate classification using the LUT trainer:

  $ ./bin/boosting_example.py -vv --trainer-type lut --digits 5 6 --classifier-file lut_uni.hdf5

3. Multi-variate classification using LUT training and shared features.

  $ ./bin/boosting_example.py -vv --trainer-type lut --all-digits ----classifier-file lut_shared.hdf5

4. Multi-variate classification using LUT training and independent features.

  $ ./bin/boosting_example.py -vv --trainer-type lut --all-digits --classifier-file lut_shared.hdf5


User Guide
----------

This section explains how to use the package in order to: a) test the MNIST dataset for binary classification
b) test the dataset for multi class classification.

a) The following command will run a single binary test for the digits specified and display the classification
accuracy on the console:

  $ ./bin/mnist_binary_one.py

if you want to see all the option associated with the command type:

  $ ./bin/mnist_binary_one.py -h

To run the tests for all the combination of of ten digits use the following command:

  $ ./bin/mnist_binary_all.py

This command tests all the possible calumniation of digits which results in 45 different binary tests. The
accuracy of individual tests and the final average accuracy of all the tests is displayed on the console.

b) The following command can be used for the multivariate digits test:

  $ ./bin/mnist_multi.py

Because of large number of samples and multivariate problem it requires times in days on a normal system. Use -h
option to see different option available with this command.


