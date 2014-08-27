# utilities to read the MNIST data

import tempfile
import tarfile
import os, shutil
import numpy
import bob.io.base
import bob.io.base.test_utils

class MNIST:
  def __init__(self):
    # loads the MNIST data from the packed data file
    tar = bob.io.base.test_utils.datafile("mnist.tar.bz2", __name__)
    temp_dir = tempfile.mkdtemp(prefix="bob_boosting_", suffix = "hdf5")[1]

    f = tarfile.open(tar, 'r')
    f.extractall(temp_dir)
    del f

    datafile = os.path.join(temp_dir, "mnist_data.hdf5")
    assert os.path.exists(datafile)

    hdf5 = bob.io.base.HDF5File(datafile)
    self._data = {}
    for group in ('train', 'test'):
      self._data[group] = []
      hdf5.cd(group)
      for i in range(10):
        self._data[group].append(hdf5.read(str(i)))
      hdf5.cd('..')

    shutil.rmtree(temp_dir)

  def data(self, groups = ('train', 'test'), labels=range(10)):
    """Returns the digits and the labels for the given labels"""
    if isinstance(groups, str):
      groups = (groups,)

    if isinstance(labels, int):
      labels = (labels,)

    _data = numpy.ndarray((0,784), dtype = numpy.uint8)
    _labels = numpy.ndarray((0), dtype = numpy.uint8)
    for group in groups:
      for label in labels:
        _data = numpy.vstack((_data, self._data[group][int(label)]))
        _labels = numpy.hstack((_labels, numpy.ones(self._data[group][int(label)].shape[:1], numpy.uint8) * int(label)))
    return _data, _labels


