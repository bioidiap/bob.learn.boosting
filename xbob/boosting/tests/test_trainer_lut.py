import unittest
import random
import xbob.boosting
import numpy
import bob

def get_single_feature():
    num_feature = 100


class TestLutTrainer(unittest.TestCase):
    """Class to test the LUT trainer """

    def test_hist_grad(self):

        num_feature = 100
        range_feature = 10
        trainer = xbob.boosting.core.trainers.LutTrainer(range_feature,'indep', 1)

        features = numpy.array([2, 8, 4, 7, 1, 0, 6, 3, 6, 1, 7, 0, 6, 8, 3, 6, 8, 2, 6, 9, 4, 6,
                                2, 0, 4, 9, 7, 4, 1, 3, 9, 9, 3, 3, 5, 2, 4, 0, 1, 3, 8, 8, 6, 7,
                                3, 0, 6, 7, 4, 0, 6, 4, 1, 2, 4, 2, 1, 9, 3, 5, 5, 8, 8, 4, 7, 4,
                                1, 5, 1, 8, 5, 4, 2, 4, 5, 3, 0, 0, 6, 2, 4, 7, 1, 4, 1, 4, 4, 4,
                                1, 4, 7, 5, 6, 9, 7, 5, 3, 3, 6, 6])

        loss_grad = numpy.ones(100)

        hist_value, bins = numpy.histogram(features,range(range_feature +1))
        sum_grad = trainer.compute_grad_hist(loss_grad,features)
        self.assertEqual(sum_grad.shape[0],range_feature)
        self.assertTrue((sum_grad == hist_value).all())









    
