import bob.learn.boosting
import numpy
import bob
import nose
import os
import tempfile


def test_example_mnist():
  # test that the MNIST examples work
  from ..examples import mnist

  # test 1: stump trainer
  options = ['-t', 'stump', '-r', '5', '-n', '100']

  mnist.main(options)

  # test 2: lut trainer -- uni-variate
  options = ['-t', 'lut', '-r', '5', '-n', '100']

  mnist.main(options)

  # test 3: lut trainer -- multi-variate, shared
  options = ['-t', 'lut', '-r', '5', '-n', '20', '-ams', 'shared']

  mnist.main(options)

  # test 4: lut trainer -- multi-variate, independent
  options = ['-t', 'lut', '-r', '5', '-n', '20', '-ams', 'independent']

  mnist.main(options)


if __name__ == '__main__':
  test_machine()
