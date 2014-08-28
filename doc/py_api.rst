============
 Python API
============

This section includes information for using the Python API of ``bob.learn.boosting``.

Machines
........

The :py:mod:`bob.learn.boosting.machine` sub-module contains classifiers that can predict classes for given input values.
The strong classifier is the :py:class:`bob.learn.boosting.BoostedMachine`, which is a weighted combination of :py:class:`bob.learn.boosting.WeakMachine`.
Weak machines might be a :py:class:`bob.learn.boosting.LUTMachine` or a :py:class:`bob.learn.boosting.StumpMachine`.
Theoretically, the strong classifier can consist of different types of weak classifiers, but usually all weak classifiers have the same type.

.. automodule:: bob.learn.boosting.machine


Trainers
........

The :py:mod:`bob.learn.boosting.trainer` sub-module contains trainers that trains:

* :py:class:`bob.learn.boosting.Boosting` : a strong machine of type :py:class:`bob.learn.boosting.BoostedMachine`
* :py:class:`bob.learn.boosting.LUTTrainer` : a weak machine of type :py:class:`bob.learn.boosting.LUTMachine`
* :py:class:`bob.learn.boosting.StrumTrainer` : a weak machine of type :py:class:`bob.learn.boosting.StumpMachine`


.. automodule:: bob.learn.boosting.trainer


Loss functions
..............

Loss functions are used to define new weights for the weak machines using the ``scipy.optimize.fmin_l_bfgs_b`` function.
A base class loss function :py:class:`bob.learn.boosting.LossFunction` is called by that function, and derived classes implement the actual loss for a single sample.

.. note::
  Loss functions are designed to be used in combination with a specific weak trainer in specific cases.
  Not all combinations of loss functions and weak trainers make sense.
  Here is a list of useful combinations:

  1. :py:class:`bob.learn.boosting.ExponentialLoss` with :py:class:`bob.learn.boosting.StrumTrainer` (uni-variate classification only)
  2. :py:class:`bob.learn.boosting.LogitLoss` with :py:class:`bob.learn.boosting.StrumTrainer` or :py:class:`bob.learn.boosting.LUTTrainer` (uni-variate or multi-variate classification)
  3. :py:class:`bob.learn.boosting.TangentialLoss` with :py:class:`bob.learn.boosting.StrumTrainer` or :py:class:`bob.learn.boosting.LUTTrainer` (uni-variate or multi-variate classification)
  4. :py:class:`bob.learn.boosting.JesorskyLoss` with :py:class:`bob.learn.boosting.LUTTrainer` (multi-variate regression only)

.. automodule:: bob.learn.boosting.loss

