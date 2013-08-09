import unittest
import random
import xbob.boosting
import numpy

class TestStumpTrainer(unittest.TestCase):
    """Perform test on loss function """

    def test_stump_trainer(self):
        # test the stump trainer for basic linearly seperable case and check the conditions on stump parameters
        trainer = xbob.boosting.core.trainers.StumpTrainer()
        n_samples = 100
        dim = 5
        x_train1 = numpy.random.randn(n_samples, dim) + 4
        x_train2 = numpy.random.randn(n_samples, dim) - 4
        x_train = numpy.vstack((x_train1, x_train2))
        y_train = numpy.hstack((numpy.ones(n_samples),-numpy.ones(n_samples)))

        scores = numpy.zeros(2*n_samples)
        t = y_train*scores
        loss = -y_train*(numpy.exp(y_train*scores))

        stump = trainer.compute_weak_trainer(x_train,loss)

        self.assertTrue(stump.threshold <= numpy.max(x_train))
        self.assertTrue(stump.threshold >= numpy.min(x_train))
        self.assertTrue(stump.selected_indices >= 0)
        self.assertTrue(stump.selected_indices < dim)

        x_test1 = numpy.random.randn(n_samples, dim) + 4
        x_test2 = numpy.random.randn(n_samples, dim) - 4
        x_test = numpy.vstack((x_test1, x_test2))
        y_test = numpy.hstack((numpy.ones(n_samples),-numpy.ones(n_samples)))

        prediction = trainer.get_weak_scores(x_test)   # return negative labels

        self.assertTrue(numpy.all(prediction.T * y_test < 0) )

