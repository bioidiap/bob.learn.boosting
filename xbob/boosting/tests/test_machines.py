import random
import xbob.boosting
import numpy
import bob
import nose
import os

global temp_file
temp_file = None

def get_temp_file():
  global temp_file
  if temp_file is None:
    import tempfile
    temp_file = tempfile.mkstemp(prefix = "xbobtest_", suffix=".hdf5")[1]
  return temp_file

def test_stump_machine():
  # test the stump machine
  machine = xbob.boosting.core.trainers.StumpMachine(0, 1, 0)

  scores = machine.get_weak_scores(numpy.ones((1,1), dtype=numpy.float64))
  assert scores.shape == (1,1)
  nose.tools.eq_(scores[0,0], 1)

  score = machine.get_weak_score(numpy.ones((1,), dtype=numpy.float64))
  nose.tools.eq_(scores, 1)


def test_lut_machine():
  # test the LUT machine
  machine = xbob.boosting.core.trainers.LutMachine(1, 1)

  scores = machine.get_weak_scores(numpy.zeros((1,1), dtype=numpy.uint16))
  assert scores.shape == (1,1)
  nose.tools.eq_(scores[0,0], 1)

  score = machine.get_weak_score(numpy.zeros((1,), dtype=numpy.uint16))
  nose.tools.eq_(scores, 1)


def test_boosted_machine():
  # test the boosted machine, by adding two different machine types (doesn't usually make sense, though...)

  stump_machine = xbob.boosting.core.trainers.StumpMachine(0., 1., 0)
  lut_machine = xbob.boosting.core.trainers.LutMachine(1, 1)

  boost_machine = xbob.boosting.core.boosting.BoostMachine()
  boost_machine.add_weak_trainer(stump_machine, numpy.array([1.]))
  boost_machine.add_weak_trainer(lut_machine, numpy.array([1.]))

  # forward some features
  scores, labels = boost_machine.classify(numpy.zeros((1,1), dtype=numpy.uint16))
  assert scores.shape == (1,1)
  nose.tools.eq_(scores[0,0], 2)

  score = boost_machine(numpy.zeros((1,), dtype=numpy.uint16))
  nose.tools.eq_(score, 2)

  # write the machine to file
  f = get_temp_file()
  boost_machine.save(bob.io.HDF5File(f, 'w'))

  new_machine = xbob.boosting.core.boosting.BoostMachine()
  new_machine.load(bob.io.HDF5File(f))
  assert (new_machine.alpha == 1).all()

  # forward some features with the new machine
  scores, labels = new_machine.classify(numpy.zeros((1,1), dtype=numpy.uint16))
  assert scores.shape == (1,1)
  nose.tools.eq_(scores[0,0], 2)

  score = new_machine(numpy.zeros((1,), dtype=numpy.uint16))
  nose.tools.eq_(score, 2)


def test_cpp_machine():
  # test the stump machine
  stump = xbob.boosting.StumpMachine(0., 1., 0)

  scores = numpy.ndarray((1,), numpy.float64)
  stump(numpy.ones((1,1), dtype=numpy.uint16), scores)
  nose.tools.eq_(scores[0], 1)

  score = stump(numpy.ones((1,), dtype=numpy.float64))
  nose.tools.eq_(scores, 1)

  # test the LUT machine
  LUT = numpy.ones((1,1), numpy.float)
  indices = numpy.zeros((1,), numpy.int32)
  machine = xbob.boosting.LUTMachine(LUT, indices)

  boosted_machine = xbob.boosting.BoostedMachine()
  boosted_machine.add_weak_machine(machine, 1.)

  score = boosted_machine(numpy.zeros((1,), dtype=numpy.uint16))
  nose.tools.eq_(score, 1)

  scores = numpy.ndarray((1,), numpy.float64)
  labels = numpy.ndarray((1,), numpy.float64)
  boosted_machine(numpy.zeros((1,1), dtype=numpy.uint16), scores, labels)
  nose.tools.eq_(scores[0], 1)
  nose.tools.eq_(labels[0], 1)

  # try to read the machine from the temp file, which was written with the python version
  f = get_temp_file()
  new_machine = xbob.boosting.BoostedMachine(bob.io.HDF5File(f))
  assert (new_machine.alpha() == 1).all()

  os.remove(get_temp_file())

  # forward some features with the new strong machine
  score = new_machine(numpy.zeros((1,), dtype=numpy.uint16))
  nose.tools.eq_(score, 2)

  scores = numpy.ndarray((1,), numpy.float64)
  labels = numpy.ndarray((1,), numpy.float64)
  new_machine(numpy.zeros((1,1), dtype=numpy.uint16), scores, labels)
  nose.tools.eq_(scores[0], 2)

