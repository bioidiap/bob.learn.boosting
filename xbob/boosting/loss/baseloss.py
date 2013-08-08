import numpy as np
import math
from pylab import *


class base_loss():
    def __init__(self,bl_type,loss_type):
	self.bl_type = bl_type
	self.loss_type =  loss_type


    def update_loss_grad(self,targets,scores):
	if self.bl_type == 'log':
	    e = exp(-(targets * scores))
	    denom = 1/(1 + e)
	    loss = log(1 + e)
	    loss_grad = - targets* e* denom
	elif self.bl_type == 'exp':
	    e = exp(-(targets * scores))
	    loss = e
	    loss_grad = -targets * e
	elif self.bl_type == 'symexp':
	    e = exp(scores - targets)
	    ie = 1/e
	    delta = 2
	    loss = e + ie - delta
	    loss_grad = e - ie
	elif self.bl_type == 'symlog':
	    e = exp(scores - targets)
	    denom = 1/(1+e)
	    loss = log(2 + eval + (1/e)) - log(4)
	    loss_grad = (e -1)*denom
	loss = sum(loss,1)
	if self.loss_type == 'var':
	    loss_grad = 2*loss_grad*(lamda*num_samp*loss + (1-lamda)*sum(loss))
	return loss, loss_grad


    def ls_compute_grad(self,*args):
        x = args[0]
        targets = args[1]
        pred_scores = args[2]
        weak_scores = args[3]
        curr_scores_x = pred_scores + x*weak_scores
        [loss, loss_grad] = self.update_loss_grad(targets, curr_scores_x )
        sum_grad = np.sum(loss_grad * weak_scores,0)
        return sum_grad

    # linesearch function to compute the loss function value at point x
    def ls_compute_fx(self, *args):
        x = args[0]
        targets = args[1]
        pred_scores = args[2]
        weak_scores = args[3]
        curr_scores_x = pred_scores + x*weak_scores
        [loss, loss_grad] = self.update_loss_grad(targets, curr_scores_x)
        sum_loss = np.sum(loss,0)
        return sum_loss

