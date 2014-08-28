.. vim: set fileencoding=utf-8 :
.. Manuel Guenther <Manuel.Guenther@idiap.ch>
.. Thu May  1 14:44:48 CEST 2014
..
.. Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland


=============================
 Boosting Strong Classifiers
=============================

Several tasks can be achieved by a boosted classifier:

1. A univariate classification task assigns each sample :math:`\vec x` one of two possible classes: :math:`{+1, -1}`.
   In this implementation, class :math:`+1` is assigned when the (real-valued) outcome of the classifier is positive, or :math:`-1` otherwise.

2. A multivariate classification task assigns each sample :math:`\vec x` one of :math:`N` possible classes: :math:`{C_1, C_2, \dots, C_N}.
   In this implementation, an :math:`N`-dimensional output vector :math:`\vec y = [y_1, y_2, ... y_n]` is assigned for each class, and the class with the highest outcome is assigned: :math:`C_n` with :math:`n = \arg \max_n y_n`.
   To train the multi-variate classifier, target values for each training sample are assigned a :math:`+1` for the correct class, and a :math:`-1` for all other classes.

3. A (multivariate) regression task tries to learn a function :math:`f(\vec x) = \vec y` based on several training examples.

To achieve this goal, a strong classifier :math:`S` is build out of a weighted list of :math:`I` weak classifiers :math:`W_i`:

.. math::
   S(\vec x) = \sum_{i=1}^I w_i \cdot W_i(\vec x)

.. note::
   For the univariate case, both :math:`w_i` and the weak classifier result :math:`W_i` are floating point values.
   In the multivariate case, :math:`w_i` is a vector of weights -- one for each output dimension -- and the weak classifier :math:`W_i` returns a vector of floating point values as well.

Weak Classifiers
----------------

Currently, two types of weak classifiers are implemented in this boosting framework.

Stump classifier
................

The first classifier, which can only handle univariate classification tasks, is the :py:class:`bob.learn.boosting.StumpMachine`.
For a given input vector :math:`\vec x`, the classifier bases its decision on **a single element** :math:`x_m` of the input vector:

.. math::
   W(\vec x) = \left\{ \begin{array}{r@{\text{ if }}l} +1 & (x_m - \theta) * \phi >= 0 \\ -1 & (x_m - \theta) * \phi < 0 \end{array}\right.

Threshold :math:`\theta`, polarity :math:`phi` and index :math:`m` are parameters of the classifier, which are trained using the :py:class:`bob.learn.boosting.StumpTrainer`.
For a given training set :math:`\{\vec x_p \mid p=1,\dots,P\}` and according target values :math:`\{t_p \mid p=1,\dots,P\}`, the threshold :math:`\theta_m` is computed for each input index :math:`m`, such that the lowest classification error is obtained, and the :math:`m` with the lowest training classification error is taken.
The polarity :math:`\phi` is set to :math:`-1`, if values lower than the threshold should be considered as positive examples, or to :math:`+1` otherwise.

To compute the classification error for a given :math:`\theta_m`, the gradient of a loss function is taken into consideration.
For the stump trainer, usually the :py:class:`bob.learn.boosting.ExponentialLoss` is considered as the loss function.


Look-Up-Table classifier
........................

The second classifier, which can handle univariate and multivariate classification and regression tasks, is the :py:class:`bob.learn.boosting.LUTMachine`.
This classifier is designed to handle input vectors with **discrete** values only.
Again, the decision of the weak classifier is based on a single element of the input vector :math:`\vec x`.

In the univariate case, for each of the possible discrete values of :math:`x_m`, a decision :math:`{+1, -1}` is selected:

.. math::
   W(\vec x) = LUT[x_m]

This look-up-table LUT and the feature index :math:`m` is trained by the :py:class:`bob.learn.boosting.LUTTrainer`.

In the multivariate case, each output :math:`W^o` is handled independently, i.e., a separate look-up-table :math:`LUT^o` and a separate feature index :math:`m^o` is assigned for each output dimension :math:`o`:

.. math::
   W^o(\vec x) = LUT^o[x_{m^o}]

.. note::
   As a variant, the feature index :math:`m^o` can be selected to be ``shared`` for all outputs, see :py:class:`bob.learn.boosting.LUTTrainer` for details.

A weak look-up-table classifier is learned using the :py:class:`bob.learn.boosting.LUTTrainer`.


Strong classifier
-----------------

The strong classifier, which is of type :py:class:`bob.learn.boosting.BoostedMachine`, is a weighted combination of weak classifiers, which are usually of the same type.
It can be trained with the :py:class:`bob.learn.boosting.Boosting` trainer, which takes a list of training samples, and a list of univariate or multivariate target vectors.
In several rounds, the trainer computes (here, only the univariate case is considered, but the multivariate case is similar -- simply replace scores by score vectors.):

1. The classification results (the so-called *scores*) for the current strong classifier:

   .. math::
      s_p = S(\vec x_p)

2. The derivative :math:`L'` of the loss function, based on the current scores and the target values:

   .. math::
      \nabla_p = L'(t_p, s_p)

3. This loss gradient is used to select a new weak machine :math:`W_i` using a weak trainer (see above).

   .. code-block:: py

      W_i = trainer.train([\vec x_p], [\nabla_p])

4. The scores of the *weak machine* are computed:

   .. math::
      r_p = W_i(\vec x_p)

5. The weight for the new machine is optimized using ``scipy.optimize.fmin_l_bfgs_b``.
   This call will use both the loss :math:`L` and its derivative :math:`L'` to compute the optimal weight for the new classifier:

   .. code-block:: py

      w_i = scipy.optimize.fmin_l_bfgs_b(...)

6. The new weak machine is added to the strong classifier.


Loss functions
--------------

As shown above, the loss functions define, how well the currently predicted scores :math:`s_p` fit to the target values :math:`t_p`.
Depending on the desired task, and on the type of classifier, different loss functions might be used:

1. The :py:class:`bob.learn.boosting.ExponentialLoss` can be used for the binary classification task, i.e., when target values are in :math:`{+1, -1}`

2. The :py:class:`bob.learn.boosting.LogitLoss` can be used for the multi-variate classification task, i.e., when target vectors have entries from :math:`{+1, 0}`

3. The :py:class:`bob.learn.boosting.JesorskyLoss` can be used for the particular multi-variate regression task of learning the locations of facial features.

Other loss functions, e.g., using the Euclidean distance for regression, should be easily implementable.


