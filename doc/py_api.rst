============
 Python API
============

This section includes information for using the Python API of ``bob.learn.boosting``.

Machines
........

The :py:mod:`bob.learn.boosting` module contains classifiers that can predict classes for given input value:

* :py:class:`bob.learn.boosting.BoostedMachine` : the strong classifier, which is a weighted combination of several machines of type :py:class:`bob.learn.boosting.WeakMachine`.

Weak machines might be:

* :py:class:`bob.learn.boosting.LUTMachine` : A weak machine that performs a classification by a look-up-table thresholding.
* :py:class:`bob.learn.boosting.StumpMachine` : A weak machine that performs classification by simple threshlding.

Theoretically, the strong classifier can consist of different types of weak classifiers, but usually all weak classifiers have the same type.


Trainers
........

Available trainers in :py:mod:`bob.learn.boosting` are:

* :py:class:`bob.learn.boosting.Boosting` : Trains a strong machine of type :py:class:`bob.learn.boosting.BoostedMachine`.
* :py:class:`bob.learn.boosting.LUTTrainer` : Trains a weak machine of type :py:class:`bob.learn.boosting.LUTMachine`.
* :py:class:`bob.learn.boosting.StumpTrainer` : Trains a weak machine of type :py:class:`bob.learn.boosting.StumpMachine`.


Loss functions
..............

Loss functions are used to define new weights for the weak machines using the ``scipy.optimize.fmin_l_bfgs_b`` function.
A base class loss function :py:class:`bob.learn.boosting.LossFunction` is called by that function, and derived classes implement the actual loss for a single sample.

.. note::
  Loss functions are designed to be used in combination with a specific weak trainer in specific cases.
  Not all combinations of loss functions and weak trainers make sense.
  Here is a list of useful combinations:

  1. :py:class:`bob.learn.boosting.ExponentialLoss` with :py:class:`bob.learn.boosting.StumpTrainer` (uni-variate classification only).
  2. :py:class:`bob.learn.boosting.LogitLoss` with :py:class:`bob.learn.boosting.StumpTrainer` or :py:class:`bob.learn.boosting.LUTTrainer` (uni-variate or multi-variate classification).
  3. :py:class:`bob.learn.boosting.TangentialLoss` with :py:class:`bob.learn.boosting.StumpTrainer` or :py:class:`bob.learn.boosting.LUTTrainer` (uni-variate or multi-variate classification).
  4. :py:class:`bob.learn.boosting.JesorskyLoss` with :py:class:`bob.learn.boosting.LUTTrainer` (multi-variate regression only).

Details
.......

.. automodule:: bob.learn.boosting
