.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Mon 25 Nov 09:43:43 2013 CET
..
.. Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

.. _bob.learn.boosting:

===========================================================================================
 Generalized Boosting Framework using Stump and Look Up Table (LUT) based Weak Classifiers
===========================================================================================

.. todolist::


The package implements a generalized boosting framework, which incorporates different boosting approaches.
The implementation is a mix of pure Python code and C++ implementations of identified bottle-necks, including their python bindings.

The Boosting algorithms implemented in this package are:

1) Gradient Boost [Fri00]_ (generalized version of Adaboost [FS99]_) for univariate cases using stump decision classifiers, as in [VJ04]_.
2) TaylorBoost [SMV11]_ for univariate and multivariate cases using Look-Up-Table based classifiers [Ata12]_

.. [Fri00]      *Jerome H. Friedman*. **Greedy function approximation: a gradient boosting machine**. Annals of Statistics, 29:1189--1232, 2000.
.. [FS99]       *Yoav Freund and Robert E. Schapire*. **A short introduction to boosting**. Journal of Japanese Society for Artificial Intelligence, 14(5):771-780, September, 1999.

.. [VJ04]       *Paul Viola and Michael J. Jones*. **Robust real-time face detection**. International Journal of Computer Vision (IJCV), 57(2): 137--154, 2004.
.. [SMV11]      *Mohammad J. Saberian, Hamed Masnadi-Shirazi, Nuno Vasconcelos*. **TaylorBoost: First and second-order boosting algorithms with explicit margin control**. IEEE Conference on Conference on Computer Vision and Pattern Recognition (CVPR), 2929--2934, 2011.
.. [Ata12]      *Cosmin Atanasoaei*. **Multivariate boosting with look-up tables for face processing**. PhD Thesis, École Polytechnique Fédérale de Lausanne (EPFL), Switzerland, 2012.


Documentation
-------------

.. toctree::
   :maxdepth: 2

   guide
   example
   py_api



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


