import unittest
import random
import xbob.boosting
import numpy

class TestExpLossFunctions(unittest.TestCase):
    """Perform test on exponential loss function """

    def test_exp_positive_target(self):
        """ Loss values computation test for postitive targets. """

        loss_function = xbob.boosting.core.losses.ExpLossFunction()
        target = 1
        score = 0.34
        alpha = 0.5
        targets = numpy.array([1, 1, 1,1,1, 1,1,1,1,1])
        weak_scores = numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'float64')
        prev_scores = numpy.array([0.53, 0.23, 0.63, 0.12, 1.34, 5.76, 3.21, 2.11, 1.21, 5.36], 'float64')

        # check the loss values
        loss_value = loss_function.loss(target, score)
        val = numpy.exp(- target * score)
        self.assertEqual(loss_value,val)
        self.assertTrue(loss_value >= 0)

        # Check loss gradient
        loss_grad = loss_function.loss_gradient( target, score)

        temp = numpy.exp(-target * score)
        val2 = -target * temp
        self.assertEqual(loss_grad,val2)

        # Check loss sum
        loss_sum_val = loss_function.loss_sum(alpha, targets, prev_scores, weak_scores)

        curr_scores = prev_scores + alpha*weak_scores
        val3 = sum(numpy.exp(-targets * curr_scores))
        self.assertEqual(val3, loss_sum_val)

        # Check the gradient sum
        grad_sum_val = loss_function.loss_grad_sum(alpha, targets, prev_scores, weak_scores)

        curr_scores = prev_scores + alpha*weak_scores
        temp = numpy.exp(-targets * curr_scores)
        grad = -target * temp
        val4 = numpy.sum(grad * weak_scores,0)

        self.assertEqual(val4, grad_sum_val)

    def test_exp_negative_target(self):
        """ Exponential Loss values computation test for negative targets. """

        loss_function = xbob.boosting.core.losses.ExpLossFunction()
        target = -1
        score = 0.34
        alpha = 0.5
        targets = numpy.array([-1, -1, -1,-1,-1, -1,-1,-1,-1,-1])
        weak_scores = numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'float64')
        prev_scores = numpy.array([0.53, 0.23, 0.63, 0.12, 1.34, 5.76, 3.21, 2.11, 1.21, 5.36], 'float64')

        # check the loss values
        loss_value = loss_function.loss(target, score)
        val = numpy.exp(- target * score)
        self.assertEqual(loss_value,val)
        self.assertTrue(loss_value >= 0)

        # Check loss gradient
        loss_grad = loss_function.loss_gradient( target, score)

        temp = numpy.exp(-target * score)
        val2 = -target * temp
        self.assertEqual(loss_grad,val2)

        # Check loss sum
        loss_sum_val = loss_function.loss_sum(alpha, targets, prev_scores, weak_scores)

        curr_scores = prev_scores + alpha*weak_scores
        val3 = sum(numpy.exp(-targets * curr_scores))
        self.assertEqual(val3, loss_sum_val)

        # Check the gradient sum
        grad_sum_val = loss_function.loss_grad_sum(alpha, targets, prev_scores, weak_scores)

        curr_scores = prev_scores + alpha*weak_scores
        temp = numpy.exp(-targets * curr_scores)
        grad = -target * temp
        val4 = numpy.sum(grad * weak_scores,0)

        self.assertEqual(val4, grad_sum_val)


