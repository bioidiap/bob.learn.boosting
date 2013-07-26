"""
This module contains methods for Boosting the features using for classification.
Bossting algorithms have three main dimensions: weak trainers that are boosting, optimization strategy 
for boosting and loss function that guide the optimization. For each one of these the following 
choices are implemented.

Weak Trainers: StumpTrainer- classifies the features based on a specified threshold
               LutTrainer- Look-Up-Table are used for classification

Optimization Strategy: For StumpTrainer the gradient descent (GradBoost) is used and for LutTrainer the 
                optimization is based on Taylor's Boosting framework. 
                See following references:
                Saberian et. al.  "Taylorboost: First and second-order boosting algorithms with explicit
                                    margin control."
                Cosmin Atanasoea "Multivariate Boosting with Look-Up Tables for face processing" Phd Thesis

Loss Function: Exponential Loss (Preferred with the StumpTrainer)
               Log Loss (Preferred with LutTrainer)
               Tangent Loss 

The module structure is the following:

- The "BoostTrainer" base class implments the 


              
"""



import numpy as np
import math
from pylab import *
from abc import ABCMeta


""" Abstract base class for the loss functions""" 
class LossFunction():
    __metaclass__ = ABCMeta
    def __init__(self):

    @abstractmethod
    def update_loss(self, targets, scores):

    @abstractmethod
    def update_loss_grad(self, targets, scores):
 
    #@abstractmethod
    def loss_sum(self, targets, scores):
        loss = self.update_loss(self,targets, scores)
        return np.sum(loss, 0)
        
    #@abstractmethod
    def loss_grad_sum(self, targets, scores):
        loss_grad = self.update_loss_grad(self, targets, scores)
        return np.sum(loss_grad, 0)


"""Exponential loss function """
class ExpLossFunction(LossFunction):

    def update_loss(self, targets, scores):
	    loss = exp(-(targets * scores))
        return loss 

    def update_loss_grad(self, targets, scores):
	    loss = exp(-(targets * scores))
	    loss_grad = -targets * loss
        return loss_grad

    """def loss_sum(self, targets, scores):
        loss = self.update_loss(self,targets, scores)
        return np.sum(loss, 0)

    def loss_grad_sum(self, targets, scores):
        loss_grad = self.update_loss_grad(self, targets, scores)
        return np.sum(loss_grad, 0)"""


"""Log loss function """
class LogLossFunction(LossFunction):
    def update_loss(self, targets, scores):
	    e = exp(-(targets * scores))
	    loss = log(1 + e)
        return loss 

    def update_loss_grad(self, targets, scores):
	    e = exp(-(targets * scores))
	    denom = 1/(1 + e)
	    loss_grad = - targets* e* denom
        return loss_grad

    """def loss_sum(self, targets, scores):
        loss = self.update_loss(self,targets, scores)
        return np.sum(loss, 0)

    def loss_grad_sum(self, targets, scores)
        loss_grad = self.update_loss_grad(self, targets, scores)"""




"""Sym Log loss function """
class SymLogLossFunction(LossFunction):
    def update_loss(self, targets, scores):
	    e = exp(scores - targets)
	    loss = log(2 + eval + (1/e)) - log(4)
        return loss 

    def update_loss_grad(self, targets, scores):
	    e = exp(scores - targets)
	    denom = 1/(1+e)
	    loss_grad = (e -1)*denom
        return loss_grad



"""Sym exponential function"""
class SymExpLossFunction(LossFunction):
    def update_loss(self, targets, scores):
	    e = exp(scores - targets)
	    ie = 1/e
	    delta = 2
	    loss = e + ie - delta
        return loss 

    def update_loss_grad(self, targets, scores):
	    e = exp(scores - targets)
	    ie = 1/e
	    loss_grad = e - ie
        return loss_grad



"""Tangent loss function"""
class TangLossFunction(LossFunction):
    def update_loss(self, targets, scores):
        loss = (2* np.arctan(targets * scores) -1)**2
        return loss

    def update_loss_grad(self, targets, scores):
        m = targets*scores
        numer = 4(2*np.arctan(m) -1)
        denom = 1 + m**2
        loss_grad = numer/demon
        return loss_grad


LOSS_FUNCTIONS = {'log':LogLossFunction,
                  'exp':ExpLossFunction,
                  'symlog':SymLogLossFunction,
                  'symexp':SynExpLossFunction,
                  'tang': TangLossFunction }   

 
