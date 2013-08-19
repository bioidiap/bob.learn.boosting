import unittest
import random
import xbob.boosting
import numpy

class TestExpLossMulti(unittest.TestCase):

    """ Test the loss function using multivariate data  """

    def test_log_multivariate_dimensions(self):

        """ Check the loss function values for multivariate targets """

        loss_function = xbob.boosting.core.losses.ExpLossFunction()
        num_samples = 2
        num_dimension = 2
        targets = numpy.array([[1, -1], [-1, 1]])
        score = numpy.array([[0.5, 0.5], [0.5, 0.5]], 'float64')
        alpha = 0.5
        weak_scores = numpy.array([[0.2, 0.4], [0.5, 0.6]], 'float64')
        prev_scores = numpy.array([[0.1, 0.2],[0.3, 0.4]], 'float64')
        
        # check the loss dimensions
        loss_value = loss_function.update_loss(targets, score) 
        self.assertTrue(loss_value.shape[0] == num_samples)
        self.assertTrue(loss_value.shape[1] == num_dimension)

        # Check loss gradient
        grad_value = loss_function.update_loss_grad( targets, score)
        self.assertTrue(grad_value.shape[0] == num_samples)
        self.assertTrue(grad_value.shape[1] == num_dimension)

        # Check loss sum
        loss_sum = loss_function.loss_sum(alpha, targets, prev_scores, weak_scores)
        self.assertTrue(loss_sum.shape[0] == num_samples)



        # Check the gradient sum
        grad_sum = loss_function.loss_grad_sum(alpha, targets, prev_scores, weak_scores)
        self.assertTrue(grad_sum.shape[0] == num_samples)



    def test_exp_negative_target(self):

        loss_function = xbob.boosting.core.losses.ExpLossFunction()
        num_samples = 2
        num_dimension = 2
        targets = numpy.array([[1, -1], [-1, 1]])
        score = numpy.array([[0.5, 0.5], [0.5, 0.5]], 'float64')
        alpha = 0.5
        weak_scores = numpy.array([[0.2, 0.4], [0.5, 0.6]], 'float64')
        prev_scores = numpy.array([[0.1, 0.2],[0.3, 0.4]], 'float64')
        
        # check the loss values
        loss_value = loss_function.update_loss(targets, score) 
        val1 = numpy.exp(- targets * score)
        self.assertTrue((loss_value == val1).all())

        # Check loss gradient
        loss_grad = loss_function.update_loss_grad( targets, score)

        temp = numpy.exp(-targets * score)
        val2 = -targets * temp
        self.assertTrue((loss_grad == val2).all())

        # Check loss sum
        loss_sum_val = loss_function.loss_sum(alpha, targets, prev_scores, weak_scores)

        curr_scores = prev_scores + alpha*weak_scores
        val3 = sum(numpy.exp(-targets * curr_scores))
        self.assertTrue((val3 == loss_sum_val).all())

        # Check the gradient sum
        grad_sum_val = loss_function.loss_grad_sum(alpha, targets, prev_scores, weak_scores)

        curr_scores = prev_scores + alpha*weak_scores        
        temp = numpy.exp(-targets * curr_scores)
        grad = -targets * temp
        val4 = numpy.sum(grad * weak_scores,0)

        self.assertTrue((val4 == grad_sum_val).all())
