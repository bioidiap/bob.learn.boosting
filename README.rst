========================================================================================
Generalized Boosting Framework using Stump and Look Up Table (LUT) based Weak Classifier
========================================================================================

The package implements a generalized boosting framework, which incorporates different boosting approaches.
The Boosting algorithms implemented in this package are:

1) Gradient Boost [Fri00]_ (generalized version of Adaboost [FS99]_) for univariate cases using stump decision classifiers, as in [VJ04]_.
2) TaylorBoost [SMV11]_ for univariate and multivariate cases using Look-Up-Table based classifiers [Ata12]_

.. [Fri00]      *Jerome H. Friedman*. **Greedy function approximation: a gradient boosting machine**. Annals of Statistics, 29:1189--1232, 2000.
.. [FS99]       *Yoav Freund and Robert E. Schapire*. **A short introduction to boosting**. Journal of Japanese Society for Artificial Intelligence, 14(5):771-780, September, 1999.

.. [VJ04]       *Paul Viola and Michael J. Jones*. **Robust real-time face detection**. International Journal of Computer Vision (IJCV), 57(2): 137--154, 2004.
.. [SMV11]      *Mohammad J. Saberian, Hamed Masnadi-Shirazi, Nuno Vasconcelos*. **TaylorBoost: First and second-order boosting algorithms with explicit margin control**. IEEE Conference on Conference on Computer Vision and Pattern Recognition (CVPR), 2929--2934, 2011.
.. [Ata12]      *Cosmin Atanasoaei*. **Multivariate boosting with look-up tables for face processing**. PhD Thesis, École Polytechnique Fédérale de Lausanne (EPFL), Switzerland, 2012.

Installation:
-------------

Bob
...

The boosting framework is dependent on the open source signal-processing and machine learning toolbox Bob_, which you need to download from its web page.
For more information, please read Bob's `installation instructions <https://github.com/idiap/bob/wiki/Packages>`_.

This package
............
The most simple way to download the latest stable version of the package is to use the Download button above and extract the archive into a directory of your choice.
If y want, you can also check out the latest development branch of this package using::

  $ git clone https://github.com/bioidiap/bob.learn.boosting.git

Afterwards, please open a terminal in this directory and call::

  $ python bootstrap.py
  $ ./bin/buildout

These 2 commands should download and install all dependencies and get you a fully operational test and development environment.


Example
-------

To show an exemplary usage of the boosting algorithm, binary and multi-variate classification of hand-written digits from the MNIST database is performed.
For simplicity, we just use the pixel gray values as (discrete) features to classify the digits.
In each boosting round, a single pixel location is selected.
In case of the stump classifier, this pixel value is compared to a threshold (which is determined during training), and one of the two classes is assigned.
The LUT weak classifier selects a feature (i.e., a pixel location in the images) and determines the most probable digit for each pixel value.
Finally, the strong classifier combines several weak classifiers by a weighted sum of their predictions.

The script ``./bin/boosting_example.py`` is provided to perform all different examples.
This script has several command line parameters, which vary the behavior of the training and/or testing procedure.
All parameters have a long value (starting with ``--``) and a shortcut (starting with a single ``-``).
These parameters are (see also ``./bin/boosting_example.py --help``):

To control the type of training, you can select:

* ``--trainer-type``: Select the type of weak classifier. Possible values are ``stump`` and ``lut``
* ``--loss-type``: Select the loss function. Possible values are ``tan``, ``log`` and ``exp``. By default, a loss function suitable to the trainer type is selected.
* ``--number-of-boosting-rounds``: The number of weak classifiers to select.
* ``--multi-variate`` (only valid for LUT trainer): Perform multi-variate classification, or binary (one-to-one) classification.
* ``--feature-selection-style`` (only valid for multi-variate training): Select the feature for each output ``independent``ly or ``shared``?

To control the experimentation, you can choose:

* ``--digits``: The digits to classify. For multi-variate training, one classifier is trained for all given digits, while for uni-variate training all possible one-to-one classifiers are trained.
* ``--all``: Select all 10 digits.
* ``--classifier-file``: Save the trained classifier(s) into the given file and/or read the classifier(s) from this file.
* ``--force``: Overwrite the given classifier file if it already exists.

For information and debugging purposes, it might be interesting to use:

* ``--verbose`` (can be used several times): Increases the verbosity level from 0 (error) over 1 (warning) and 2 (info) to 3 (debug). Verbosity level 2 (``-vv``) is recommended.
* ``--number-of-elements``: Reduce the number of elements per class (digit) to the given value.

Four different kinds of experiments can be performed:

1. Uni-variate classification using the stump classifier, classifying digits 5 and 6::

    $ ./bin/boosting_example.py -vv --trainer-type stump --digits 5 6

2. Uni-variate classification using the LUT classifier, classifying digits 5 and 6::

    $ ./bin/boosting_example.py -vv --trainer-type lut --digits 5 6

3. Multi-variate classification using LUT classifier and shared features, classifying all 10 digits::

    $ ./bin/boosting_example.py -vv --trainer-type lut --all-digits --multi-variate --feature-selection-style shared

4. Multi-variate classification using LUT classifier and independent features, classifying all 10 digits::

    $ ./bin/boosting_example.py -vv --trainer-type lut --all-digits --multi-variate --feature-selection-style independent


.. note:
  During the execution of the experiments, the warning message "L-BFGS returned warning '2': ABNORMAL_TERMINATION_IN_LNSRCH" might appear.
  This warning message is normal and does not influence the results much.

.. note:
  For experiment 1, the training terminates after 75 of 100 rounds since the computed weight for the weak classifier of that round is vanishing.
  Hence, performing more boosting rounds will not change the strong classifier any more.

All experiments should be able to run using several minutes of execution time.
The results of the above experiments should be the following (split in the remaining classification error on the training set, and the error on the test set)

+------------+----------+----------+
| Experiment | Training |   Test   |
+------------+----------+----------+
|   1        |  91.04 % |  92.05 % |
+------------+----------+----------+
|   2        |  100.0 % |  95.35 % |
+------------+----------+----------+
|   3        |  97.59 % |  83.47 % |
+------------+----------+----------+
|   4        |  99.04 % |  86.25 % |
+------------+----------+----------+

Of course, you can try out different combinations of digits for experiments 1 and 2.


Getting Help
------------

In case you experience problems with the code, or with downloading the required databases and/or software, please contact manuel.guenther@idiap.ch or file a bug report under https://github.com/bioidiap/bob.learn.boosting.

.. _bob: http://www.idiap.ch/software/bob
