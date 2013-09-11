import unittest
import random
import xbob.boosting
import numpy
import bob

class TestMachines(unittest.TestCase):
  """Perform test on stump weak trainer"""

  def test_stump_machine(self):
    # test the stump machine
    machine = xbob.boosting.core.trainers.StumpMachine(0, 1, 0)

    scores = machine.get_weak_scores(numpy.ones((1,1), dtype=numpy.float64))
    self.assertTrue(scores.shape == (1,1))
    self.assertEqual(scores[0,0], 1)

    score = machine.get_weak_score(numpy.ones((1,), dtype=numpy.float64))
    self.assertEqual(scores, 1)


  def test_lut_machine(self):
    # test the LUT machine
    machine = xbob.boosting.core.trainers.LutMachine(1, 1)

    print machine.luts
    print machine.selected_indices

    scores = machine.get_weak_scores(numpy.zeros((1,1), dtype=numpy.uint8))
    self.assertTrue(scores.shape == (1,1))
    self.assertEqual(scores[0,0], 1)

    score = machine.get_weak_score(numpy.zeros((1,), dtype=numpy.uint8))
    self.assertEqual(scores, 1)


  def test_boosted_machine(self):
    # test the boosted machine, by adding two different machine types (doesn't usually make sense, though...)

    stump_machine = xbob.boosting.core.trainers.StumpMachine(0, 1, 0)
    lut_machine = xbob.boosting.core.trainers.LutMachine(1, 1)

    boost_machine = xbob.boosting.core.boosting.BoostMachine()
    boost_machine.add_weak_trainer(stump_machine, 1.)
    boost_machine.add_weak_trainer(lut_machine, 1.)

    # forward some features
    scores, labels = boost_machine.classify(numpy.zeros((1,1), dtype=numpy.uint8))
    self.assertTrue(scores.shape == (1,1))
    self.assertEqual(scores[0,0], 2)

    score = boost_machine(numpy.zeros((1,), dtype=numpy.uint8))
    self.assertEqual(scores, 2)

    # write the machine to file
    import tempfile
    f = tempfile.mkstemp(prefix = "xbobtest_")[1]
    boost_machine.save(bob.io.HDF5File(f, 'w'))

    new_machine = xbob.boosting.core.boosting.BoostMachine()
    new_machine.load(bob.io.HDF5File(f))
    self.assertTrue((new_machine.alpha == 1).all())

    # forward some features with the new machine
    scores, labels = new_machine.classify(numpy.zeros((1,1), dtype=numpy.uint8))
    self.assertTrue(scores.shape == (1,1))
    self.assertEqual(scores[0,0], 2)

    score = new_machine(numpy.zeros((1,), dtype=numpy.uint8))
    self.assertEqual(scores, 2)

