import unittest
import random
import xbob.boosting
import numpy

class TestLossFunctions(unittest.TestCase):
    """Perform test on loss function """

    def test_exp_loss(self):

        exp_ = xbob.boosting.core.losses.ExpLossFunction()
        target = 1
        score = numpy.random.rand()
        
        # check the loss values
        l1 = exp_.update_loss(target, score) 
        val1 = numpy.exp(- target * score)
        self.assertEqual(l1,val1)

        # Check loss gradient
        l2 = exp_.update_loss_grad( target, score)
        temp = numpy.exp(-target * score)
        val2 = -target * temp
        self.assertEqual(l2,val2)

        # Check loss sum
        weak_scores = numpy.random.rand(10)
        prev_scores = numpy.random.rand(10)
        x = numpy.random.rand(1)
        curr_scores = prev_scores + x*weak_scores
        l3 = exp_.loss_sum(x, target, prev_scores, weak_scores)
        val3 = sum(numpy.exp(-target * curr_scores))
        self.assertEqual(val3, l3)

        # Check the gradient sum
        weak_scores = numpy.random.rand(10)
        prev_scores = numpy.random.rand(10)
        x = numpy.random.rand(1)
        curr_scores = prev_scores + x*weak_scores
        l4 = exp_.loss_grad_sum(x, target, prev_scores, weak_scores)
        temp = numpy.exp(-target * curr_scores)
        grad = -target * temp
        val4 = numpy.sum(grad * weak_scores,0)
        self.assertEqual(val4, l4)
            
    def test_log_loss(self):

        exp_ = xbob.boosting.core.losses.LogLossFunction()
        target = 1
        score = numpy.random.rand()
        
        # check the loss values
        l1 = exp_.update_loss(target, score) 
        val1 = numpy.log(1 + numpy.exp(- target * score))
        self.assertEqual(l1,val1)

        # Check loss gradient
        l2 = exp_.update_loss_grad( target, score)
        temp = numpy.exp(-target * score)
        val2 = -(target * temp* (1/(1 + temp)) )
        self.assertEqual(l2,val2)

        # Check loss sum
        weak_scores = numpy.random.rand(10)
        prev_scores = numpy.random.rand(10)
        x = numpy.random.rand(1)
        curr_scores = prev_scores + x*weak_scores
        l3 = exp_.loss_sum(x, target, prev_scores, weak_scores)
        val3 = sum(numpy.log(1 + numpy.exp(-target * curr_scores)))
        self.assertEqual(val3, l3)

        # Check the gradient sum
        weak_scores = numpy.random.rand(10)
        prev_scores = numpy.random.rand(10)
        x = numpy.random.rand(1)
        curr_scores = prev_scores + x*weak_scores
        l3 = exp_.loss_grad_sum(x, target, prev_scores, weak_scores)
        temp = numpy.exp(-target * curr_scores)
        grad = -target * temp *(1/ (1 + temp))
        val3 = numpy.sum(grad * weak_scores)
        self.assertEqual(val3, l3)

             

             
