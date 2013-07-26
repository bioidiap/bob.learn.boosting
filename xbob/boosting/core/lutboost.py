import numpy as np
import math
import baseloss
import random
import DummyBoost
#import StumpTrainer
import boostMachine
import weaklearner
from pylab import *
from scipy import optimize





class LUTBoost:

    def __init__(self, weakTrainer):
        self.num_rnds = 100
        self. num_entries = 256
        self.bl_type = 'log' 
        self.loss_type = 'ept'
        self.lamda = 0.5
        self.selection_type = 'indep'
        self.weak_trainer_type = weakTrainer
							
	

	# function LUTBoost_train boosts the discrete features (fset) to provide a strong classifier 
	# as a combintaion of weak classifier. Each weak classifier is represented by a feature index
	# and a LUT for univariate case and a set of feature indices and LUTs for multivariate case.

	# fset- (num_sam x num_features) features extracted from the samples
	#        features should be discrete
	# labels- class labels of the samples of dimension (#samples x #outputs)
	# param- parameters for the boosting
	
    def train(self, fset, targets):
		# Initializations
        # num_samp = len(fset)
        machine = boostMachine.boost_machine()
        num_op = targets.shape[1]
        num_samp = fset.shape[0]
        pred_scores = np.zeros([num_samp,num_op])
		#pred_scores = np.zeros([num_samp])
        strong_learner = {}
        fset = fset.astype(int)
        #loss_class = baseloss.base_loss(self.bl_type,self.loss_type)
	loss_class = DummyBoost.LOSS_FUNCTIONS[self.bl_type]
        loss_ = loss_class()
        #print loss_
	
        # Start boosting iterations for num_rnds rounds
        for r in range(self.num_rnds):
            if(self.weak_trainer_type == 'LutTrainer'):
                wl = weaklearner.LutTrainer(256, self.selection_type, num_op )
            elif (self.weak_trainer_type == 'StumpTrainer'):
                wl = DummyBoost.StumpTrainer()

            # Compute the baseloss, l(y,f(x)), and its gradient, l'(y,f(x)) using loss type ltype
            loss_grad = loss_.update_loss_grad(targets,pred_scores)


            # Select the best feature indices and corresponding LUTs for current round of boosting
            curr_weak_trainer = wl.compute_weak_trainer(fset, loss_grad)

            # Compute the classification scores of the samples based only on the current round weak classifier
            curr_pred_scores = wl.get_weak_scores(fset)
				#lg = wl.ls_compute_grad(targets, pred_scores, weak_pred_scores, bl)
				#lv = wl.ls_compute_fx(targets, pred_scores, weak_pred_scores, bl)
			# Use the line search method to find the alpha values for current classifier
            ls_res = optimize.fmin_l_bfgs_b(loss_.loss_sum, 0, fprime = loss_.loss_grad_sum, args = (targets, pred_scores, curr_pred_scores)) 
            #ls_res = optimize.fmin_l_bfgs_b(bl.compute, 0, fprime = loss_.loss_grad_sum, args = (targets, pred_scores, curr_pred_scores)) 
            #print ls_res
			# The scale for the current classifier returned from the lbpfs optimization
            alpha = ls_res[0]
            #print curr_weak_trainer.selected_indices
            #print np.transpose(curr_weak_trainer.luts) 
			# Update the current prediction score f(x) using the strong learner f
            pred_scores = pred_scores + alpha* curr_pred_scores 
		   	# pred_scores =  self.update_scores(pred_scores, luts, fset, findices)
            #print np.transpose(pred_scores[0:100])
            machine.add_weak_trainer(curr_weak_trainer, alpha)
            print "Boosting done for round number : %d" % (r+1)
			
			
        return machine
	# linesearch function to compute the gradient of loss function at point x
def ls_compute_grad(x,*args):
    targets = args[0]
    pred_scores = args[1]
    weak_scores = args[2]
    bl_type = args[3]
    curr_scores_x = pred_scores + x*weak_scores
    [loss, loss_grad] = baseloss.update_loss_grad(targets, curr_scores_x, bl_type )
    cum_grad = np.sum(loss_grad * weak_scores,0)
    return cum_grad
                                        
# linesearch function to compute the loss function value at point x
def ls_compute_fx(x, *args):
    targets = args[0]
    pred_scores = args[1]
    weak_scores = args[2]
    bl_type = args[3]
    curr_scores_x = pred_scores + x*weak_scores
    [loss, loss_grad] = baseloss.update_loss_grad(targets, curr_scores_x, bl_type)
    cum_loss = np.sum(loss,0)
    return cum_loss

