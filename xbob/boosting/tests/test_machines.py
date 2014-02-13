import xbob.boosting
import numpy
import bob
import nose
import os
import tempfile


def test_machine():
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
  boosted_machine.add_weak_machine(stump, 1.)
  boosted_machine.add_weak_machine(machine, 1.)

  score = boosted_machine(numpy.zeros((1,), dtype=numpy.uint16))
  nose.tools.eq_(score, 2)

  scores = numpy.ndarray((1,), numpy.float64)
  labels = numpy.ndarray((1,), numpy.float64)
  boosted_machine(numpy.zeros((1,1), dtype=numpy.uint16), scores, labels)
  nose.tools.eq_(scores[0], 2)
  nose.tools.eq_(labels[0], 1)

  # check IO functionality
  file = tempfile.mkstemp(prefix='xbob_test_')[1]
  boosted_machine.save(bob.io.HDF5File(file, 'w'))
  new_machine = xbob.boosting.BoostedMachine(bob.io.HDF5File(file))
  os.remove(file)
  assert (new_machine.alpha == 1).all()

  # forward some features with the new strong machine
  score = new_machine(numpy.zeros((1,), dtype=numpy.uint16))
  nose.tools.eq_(score, 2)

  scores = numpy.ndarray((1,), numpy.float64)
  new_machine(numpy.zeros((1,1), dtype=numpy.uint16), scores)
  nose.tools.eq_(scores[0], 2)

  labels = numpy.ndarray((1,), numpy.float64)
  new_machine(numpy.zeros((1,1), dtype=numpy.uint16), scores, labels)
  nose.tools.eq_(scores[0], 2)
  nose.tools.eq_(labels[0], 1)

