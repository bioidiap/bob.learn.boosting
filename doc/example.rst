.. vim: set fileencoding=utf-8 :
.. Manuel Guenther <Manuel.Guenther@idiap.ch>
.. Thu May  1 19:08:03 CEST 2014
..
.. Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

.. testsetup:: *

   import os
   import numpy
   import bob.learn.boosting
   import bob.learn.boosting.utils

   numpy.set_printoptions(precision=3, suppress=True)


===========================================
 Example: Handwritten Digit Classification
===========================================

As an example for the classification task, we perform a classification of hand-written digits using the `MNIST <http://yann.lecun.com/exdb/mnist>`_ database.
There, images of single hand-written digits are stored, and a training and test set is provided, which we can access with our `bob.db.mnist <http://pypi.python.org/pypi/bob.db.mnist>`_ database interface.

.. note::
  In fact, to minimize the dependencies to other packages, the ``bob.db.mnist`` database interface is replaced by a local interface.

In our experiments, we simply use the pixel gray values as features.
Since the gray values are discrete in range :math:`[0, 255]`, we can employ both the stump decision classifiers and the look-up-table's.
Nevertheless, other discrete features, like Local Binary Patterns (LBP) could be used as well.


Running the example script
--------------------------

The script ``./bin/boosting_example.py`` is provided to execute digit classification tasks.
This script has several command line parameters, which vary the behavior of the training and/or testing procedure.
All parameters have a long value (starting with ``--``) and a shortcut (starting with a single ``-``).
These parameters are (see also ``./bin/boosting_example.py --help``):

To control the type of training, you can select:

* ``--trainer-type``: Select the type of weak classifier. Possible values are ``stump`` and ``lut``
* ``--loss-type``: Select the loss function. Possible values are ``tan``, ``log`` and ``exp``. By default, a loss function suitable to the trainer type is selected.
* ``--number-of-boosting-rounds``: The number of weak classifiers to select.
* ``--multi-variate`` (only valid for LUT trainer): Perform multi-variate classification, or binary (one-to-one) classification.
* ``--feature-selection-style`` (only valid for multi-variate training): Select the feature for each output ``independent`` or ``shared``?

To control the experimentation, you can choose:

* ``--digits``: The digits to classify. For multi-variate training, one classifier is trained for all given digits, while for uni-variate training all possible one-to-one classifiers are trained.
* ``--all``: Select all 10 digits.
* ``--classifier-file``: Save the trained classifier(s) into the given file and/or read the classifier(s) from this file.
* ``--force``: Overwrite the given classifier file if it already exists.

For information and debugging purposes, it might be interesting to use:

* ``--verbose`` (can be used several times): Increases the verbosity level from 0 (error) over 1 (warning) and 2 (info) to 3 (debug). Verbosity level 2 (``-vv``) is recommended.
* ``--number-of-elements``: Reduce the number of elements per class (digit) to the given value.

Four different kinds of experiments can be performed:

1. Uni-variate classification using the stump classifier :py:class:`bob.learn.boosting.StumpMachine`, classifying digits 5 and 6::

    $ ./bin/boosting_example.py -vv --trainer-type stump --digits 5 6

2. Uni-variate classification using the LUT classifier :py:class:`bob.learn.boosting.LUTMachine`, classifying digits 5 and 6::

    $ ./bin/boosting_example.py -vv --trainer-type lut --digits 5 6

3. Multi-variate classification using LUT classifier :py:class:`bob.learn.boosting.LUTMachine` and shared features, classifying all 10 digits::

    $ ./bin/boosting_example.py -vv --trainer-type lut --all-digits --multi-variate --feature-selection-style shared

4. Multi-variate classification using LUT classifier :py:class:`bob.learn.boosting.LUTMachine` and independent features, classifying all 10 digits::

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


One exemplary test case in details
----------------------------------

Having a closer look into the example script, there are several steps that are performed.
The first step is generating the training examples from the MNIST database interface.
Here, we describe the more complex way, i.e., the multi-variate case.

.. doctest::

   >>> # open the database interface (will download the digits from the webpage)
   >>> db = bob.learn.boosting.utils.MNIST()
   >>> # get the training data for digits 0, 1
   >>> training_samples, training_labels = db.data("train", labels = [0, 1])
   >>> # limit the training samples (for test purposes only)
   >>> training_samples = training_samples[:100]
   >>> training_labels = training_labels[:100]

   >>> # create the correct entries for the training targets from the classes; pre-fill with negative class
   >>> training_targets = -numpy.ones((training_labels.shape[0], 2))
   >>> # set positive class
   >>> for i in [0,1]:
   ...   training_targets[training_labels == i, i] = 1
   >>> training_labels[:10]
   array([0, 1, 1, 1, 1, 0, 1, 1, 0, 0], dtype=uint8)
   >>> training_targets[:10]
   array([[ 1., -1.],
          [-1.,  1.],
          [-1.,  1.],
          [-1.,  1.],
          [-1.,  1.],
          [ 1., -1.],
          [-1.,  1.],
          [-1.,  1.],
          [ 1., -1.],
          [ 1., -1.]])

Now, we can train the classifier using the :py:class:`bob.learn.boosting.Boosting` boosting trainer.
Here, we use the multi-variate LUT trainer :py:class:`bob.learn.boosting.LUTTrainer` with logit loss :py:class:`bob.learn.boosting.LogitLoss`:

.. doctest::

  >>> weak_trainer = bob.learn.boosting.LUTTrainer(
  ...       maximum_feature_value = 256,
  ...       number_of_outputs = 2,
  ...       selection_style = 'independent'
  ... )
  >>> loss_function = bob.learn.boosting.LogitLoss()
  >>> strong_trainer = bob.learn.boosting.Boosting(weak_trainer, loss_function)

  >>> # perform training for 100 rounds (i.e., select 100 weak machines)
  >>> strong_classifier = strong_trainer.train(training_samples.astype(numpy.uint16), training_targets, 10)

Having the strong classifier (which is of type :py:class:`bob.learn.boosting.BoostedMachine`), we can classify the test samples:

.. doctest::

   >>> # get the test data for digits 0, 1
   >>> test_samples, test_labels = db.data("test", labels = [0, 1])

   >>> # create the correct entries for the test targets from the classes; pre-fill with negative class
   >>> test_targets = -numpy.ones((test_labels.shape[0], 2))
   >>> # set positive class
   >>> for i in [0,1]:
   ...   test_targets[test_labels == i, i] = 1

  >>> # classify the test samples
  >>> scores = numpy.zeros(test_targets.shape)
  >>> classification = numpy.zeros(test_targets.shape)
  >>> strong_classifier(test_samples.astype(numpy.uint16), scores, classification)

  >>> # evaluate the results
  >>> row_sum = numpy.sum(test_targets == classification, 1)
  >>> # the example is correctly classified, when all test labels correspond to all target labels
  >>> correctly_classified = numpy.sum(row_sum == 2)
  >>> correctly_classified
  2004
  >>> classification.shape[0]
  2115

