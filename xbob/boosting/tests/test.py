#import unittest
#import DummyBoost
#import random
#import numpy as np

class TestLossFunctions(unittest.TestCases):
    """Perform test on loss function """

    def test_exp_loss(self):

        exp__ = DummyBoost.ExpLossFunction()

        # Test the loss computation with 20 randomly generated values
        for i in range(20):
            target = 1
            score = np.random.rand()
            
            # check the loss values
            l1 = exp_.update_loss(target, score) 
            val1 = exp(- target * score)
            self.assertEqual(l1,val1)

            # Check loss gradient
            l2 = exp_.update_loss_grad( target, score)
            temp = exp(-target * score)
            val2 = -target * temp
            self.assertEqual(l2,val2)

            # Check loss sum
            weak_scores = np.random.rand([1,10])
            prev_scores = np.random.rand([1,10])
            x = np.random.rand(1)
            curr_scores = prev_score + x*weak_scores
            l3 = exp_.loss_sum(x, weak_scores, prev__scores)
            val3 = sum(exp(-target * curr_scores))
            self.assertEqual(val3, l3)

            # Check all the above function for negative target
            target = -1
            score = np.random.rand()
            
            l1 = exp_.update_loss(target, score) 
            val1 = exp(- target * score)
            self.assertEqual(l1,val1)


            l2 = exp_.update_loss_grad( target, score)
            temp = exp(-target * score)
            val2 = -target * temp
            self.assertEqual(l2,val2)

            weak_scores = np.random.rand([1,10])
            prev_scores = np.random.rand([1,10])
            x = np.random.rand(1)
            curr_scores = prev_score + x*weak_scores
            l3 = exp_.loss_sum(x, weak_scores, prev__scores)
            val3 = sum(exp(-target * curr_scores))
            self.assertEqual(val3, l3)

            

             

class TestStumpTrainer(unittest_testCases)

    def test_stumps(self):
        
        trainer = DummyBoost.StumpTrainer()
        

             
