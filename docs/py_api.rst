============
 Python API
============

This section includes information for using the Python API of ``xbob.boosting``.

Machines
........

The :py:mod:`xbob.boosting.machine` sub-module contains classifiers that can predict classes for given input values.
The strong classifier is the :py:class:`xbob.boosting.machine.BoostedMachine`, which is a weighted combination of :py:class:`xbob.boosting.machine.WeakMachine`.
Weak machines might be a :py:class:`xbob.boosting.machine.LUTMachine` or a :py:class:`xbob.boosting.machine.StumpMachine`.
Theoretically, the strong classifier can consist of different types of weak classifiers, but usually all weak classifiers have the same type.

.. automodule:: xbob.boosting.machine


Trainers
........

The :py:mod:`xbob.boosting.trainer` sub-module contains trainers that trains:

* :py:class:`xbob.boosting.trainer.Boosting` : a strong machine of type :py:class:`xbob.boosting.machine.BoostedMachine`
* :py:class:`xbob.boosting.trainer.LUTTrainer` : a weak machine of type :py:class:`xbob.boosting.machine.LUTMachine`
* :py:class:`xbob.boosting.trainer.StrumTrainer` : a weak machine of type :py:class:`xbob.boosting.machine.StumpMachine`


.. automodule:: xbob.boosting.trainer


Loss functions
..............

Loss functions are used to define new weights for the weak machines using the ``scipy.optimize.fmin_l_bfgs_b`` function.
A base class loss function :py:class:`xbob.boosting.loss.LossFunction` is called by that function, and derived classes implement the actual loss for a single sample.

.. note::
  Loss functions are designed to be used in combination with a specific weak trainer in specific cases.
  Not all combinations of loss functions and weak trainers make sense.
  Here is a list of useful combinations:

  1. :py:class:`xbob.boosting.loss.ExponentialLoss` with :py:class:`xbob.boosting.trainer.StrumTrainer` (uni-variate classification only)
  2. :py:class:`xbob.boosting.loss.LogitLoss` with :py:class:`xbob.boosting.trainer.StrumTrainer` or :py:class:`xbob.boosting.trainer.LUTTrainer` (uni-variate or multi-variate classification)
  3. :py:class:`xbob.boosting.loss.TangentialLoss` with :py:class:`xbob.boosting.trainer.StrumTrainer` or :py:class:`xbob.boosting.trainer.LUTTrainer` (uni-variate or multi-variate classification)
  4. :py:class:`xbob.boosting.loss.JesorskyLoss` with :py:class:`xbob.boosting.trainer.LUTTrainer` (multi-variate regression only)

.. automodule:: xbob.boosting.loss

