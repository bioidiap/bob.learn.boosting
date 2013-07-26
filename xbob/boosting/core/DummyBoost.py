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
from scipy import optimize
from abc import ABCMeta



"""Exponential loss function """
class ExpLossFunction():

    def update_loss(self, targets, scores):
	    return exp(-(targets * scores))
        #return loss 

    def update_loss_grad(self, targets, scores):
	    loss = exp(-(targets * scores))
	    return -targets * loss
        #return loss_grad

    def loss_sum(self, *args):
        x = args[0]
        targets = args[1]
        pred_scores = args[2]
        weak_scores = args[3]
        curr_scores_x = pred_scores + x*weak_scores
        loss = self.update_loss(targets, curr_scores_x)
        sum_l = np.sum(loss,0)
        return sum_l
        
    #@abstractmethod
    def loss_grad_sum(self, *args):
        x = args[0]
        targets = args[1]
        pred_scores = args[2]
        weak_scores = args[3]
        curr_scores_x = pred_scores + x*weak_scores
        loss_grad = self.update_loss_grad(targets, curr_scores_x)
        sum_g = np.sum(loss_grad*weak_scores, 0)
        return sum_g




"""Log loss function """
class LogLossFunction():
    def update_loss(self, targets, scores):
	    e = exp(-(targets * scores))
	    return log(1 + e)
        #return loss 

    def update_loss_grad(self, targets, scores):
	    e = exp(-(targets * scores))
	    denom = 1/(1 + e)
	    return - targets* e* denom

    def loss_sum(self, *args):
        x = args[0]
        targets = args[1]
        pred_scores = args[2]
        weak_scores = args[3]
        curr_scores_x = pred_scores + x*weak_scores
        loss = self.update_loss(targets, curr_scores_x)
        sum_l = np.sum(loss,0)
        return sum_l
        
    #@abstractmethod
    def loss_grad_sum(self, *args):
        x = args[0]
        targets = args[1]
        pred_scores = args[2]
        weak_scores = args[3]
        curr_scores_x = pred_scores + x*weak_scores
        loss_grad = self.update_loss_grad( targets, curr_scores_x)
        sum_g = np.sum(loss_grad*weak_scores, 0)
        return sum_g


    """def loss_sum(self, targets, scores):
        loss = self.update_loss(self,targets, scores)
        return np.sum(loss, 0)

    def loss_grad_sum(self, targets, scores)
        loss_grad = self.update_loss_grad(self, targets, scores)"""


"""Tangent loss function """

class TangLossFunction():
    def update_loss(self, targets, scores):
        loss = (2* np.arctan(targets * scores) -1)**2
        return loss

    def update_loss_grad(self, targets, scores):
        m = targets*scores
        numer = 4*(2*np.arctan(m) -1)
        denom = 1 + m**2
        loss_grad = numer/denom
        return loss_grad

    def loss_sum(self, *args):
        x = args[0]
        targets = args[1]
        pred_scores = args[2]
        weak_scores = args[3]
        curr_scores_x = pred_scores + x*weak_scores
        loss = self.update_loss(targets, curr_scores_x)
        return np.sum(loss, 0)
        
    #@abstractmethod
    def loss_grad_sum(self, *args):
        x = args[0]
        targets = args[1]
        pred_scores = args[2]
        weak_scores = args[3]
        curr_scores_x = pred_scores + x*weak_scores
        loss_grad = self.update_loss_grad( targets, curr_scores_x)
        return np.sum(loss_grad*weak_scores, 0)



LOSS_FUNCTIONS = {'log':LogLossFunction,
                  'exp':ExpLossFunction,
                  'tang':TangLossFunction}





""" The weak trainer class for training stumps as classifiers. The trainer is parameterized 
    the threshold th and the polarity p. """
class StumpTrainer():


    """ Initilize the stump classifier"""
    def  __init__(self):
        self.th = 0
        self.p = 0
        self.selected_indices = 0
 


    """The function computes the weak stump trainer. It is called at each boosting round.
       The best weak stump trainer is chosen to maximize the dot product of the outputs 
       and the weights (gain). The weights in the Adaboost are the negative of the loss gradient
       for exponential loss.
       Inputs: fea: the training feature set
               loss_grad: the gradient of the loss function for the training samples
                          Chose preferable exponential loss function to simulate Adaboost 
       Return: self: a StumpTrainer Object, i.e. the optimal trainer that minimizes the loss
                     """

    def compute_weak_trainer(self, fea, loss_grad):

        # Initialization
        numSamp, numFea = fea.shape
        th = np.zeros([numFea])
        p = np.zeros([numFea])
        g = np.zeros([numFea])

        # For each feature find the optimum threshold, polarity and the gain
        for i in range(numFea):
            p[i],th[i], g[i] = self.compute_thresh(fea[:,i],loss_grad)

        #  Find the optimum id and tis corresponding trainer
        opt_id = g.argmax()
        self.th = th[opt_id]
        self.p = p[opt_id]
        self.selected_indices = opt_id
        return self



    """  Function to compute the threshold for a single feature. The threshold is computed for 
    the given feature values using the weak learner algorithm given in the Voila Jones Robust Face classification

    Inputs:
    f: The feature values for a single index, array of dimension N = No. of training Samples
    loss_grad: The loss gradient values for the training samples. Array of dimension N.
    labels: The class of the training samples. Array of dimension N.

    Return:(weak stump classifier for given feature)
    th: threshold which minimizes the error
    p: the polarity or the direction used for stump classification
    gain: gain of the classifier"""


    def compute_thresh(self, f,loss_grad):
        # The weights for Adaboost are negative of exponential loss gradient
        loss_grad = -loss_grad
        num_samp = f.shape[0]

        # Sort the feature and rearrange the corresponding weights and feature values
        sorted_id = np.argsort(f)
        f = f[sorted_id] 
        loss_grad = loss_grad[sorted_id]

        # For all the threshold compute the dot product
        grad_cs =  np.cumsum(loss_grad)
        grad_sum = grad_cs[-1]
        g = (grad_sum - grad_cs)

        # Find the index that maximizes the dot product
        opt_id = np.argmax(np.absolute(g))        
        g_opt = np.absolute(g[opt_id])

        # Find the corresponding threshold value
        if(opt_id == num_samp-1):
            th = f[opt_id]
        else:
            th = float(f[opt_id] + f[opt_id+1])/2

        # Find the polarity or the directionality of the current trainer
        if(g_opt == g[opt_id]):
            p = -1
        else:
            p = 1

        return p, th, g_opt


    """ The function computes the classification scores for the test features using 
        a weak stump trainer. Since we use the stump classifier the classification 
        scores are either +1 or -1.
        Input: self: a weak stump trainer
               test_features: A matrix of the test features of dimension. 
                              Num. of Test images x Num of features
        Return: weak_scores: classification scores of the test features use the weak classifier self
                             Array of dimension =  Num. of samples """

    def get_weak_scores(self,test_features):
        # Initialize the values
        numSamp = test_features.shape[0]
        weak_scores = np.ones([numSamp,1])
  
        # Select feature corresponding to the specific index
        weak_features = test_features[:,self.selected_indices]

        # classify the features and compute the score
        weak_scores[weak_features < self.th] = -1
        weak_scores = self.p *weak_scores
        return weak_scores


""" The LutTrainer class contain methods to learn weak trainer using LookUp Tables. 
    It can be used for multi-variate binary classfication  """


class LutTrainer():
 
    """ Function to initilize the weak LutTrainer. Each weak Luttrainer is specified with a 
        LookUp Table and the feature index which corresponds to the feature on which the 
        current classifier has to applied.  """
    
    def __init__(self, num_entries, s_type, num_op):
        self.num_entries = num_entries
        self.luts = np.ones((num_entries, num_op), dtype = np.int)
        self.s_type = s_type
        self.selected_indices = np.zeros([num_op,1], 'int16')
    

    """ The function to learn the weak LutTrainer.  """

    def compute_weak_trainer(self, fea, loss_grad):

        # Initializations
        num_op = loss_grad.shape[1]
        fea_grad = np.zeros([self.num_entries,num_op])

        # Compute the sum of the gradient based on the feature values or the loss associated with each 
        # feature index
        sum_loss = self.compute_fgrad(loss_grad, fea)


        # Select the most discriminative index (or indices) for classification which minimizes the loss
        #  and compute the sum of gradient for that index
       
        if self.s_type == 'indep':

            # indep (independent) feature selection is used if all the dimension of output use different feature
            # each of the selected feature minimize a dimension of the loss function

            selected_indices = [np.argmin(col) for col in np.transpose(sum_loss)]

            for oi in range(num_op):
                curr_id = sum_loss[:,oi].argmin()
                fea_grad[:,oi] = self.compute_hgrad(loss_grad[:,oi],fea[:,curr_id])
                self.selected_indices[oi] = curr_id


        elif self.s_type == 'shared':

            # for 'shared' feature selection the loss function is summed over multiple dimensions and 
            # the feature that minimized this acumulative loss is used for all the outputs

            accum_loss = np.sum(sum_loss,1)
            selected_findex = accum_loss.argmin()
            self.selected_indices = selected_findex*np.ones([num_op,1],'int16')

            for oi in range(num_op):
                fea_grad[:,oi] = self.compute_hgrad(loss_grad[:,oi],fea[:,selected_findex])
     
        # Assign the values to LookUp Table
        self.luts[fea_grad <= 0.0] = -1
        return self
    


    """ The function computes the loss for whole set of features. The loss refers to the sum of the loss gradient
        of the features which have the same values.
  
        Inputs: loss_grad: The loss gradient for the features. No. of samples x No. of outputs 
                fea: set of features. No. of samples x No. of features

        Output: sum_loss: the loss values for all features. No. of samples x No. of outputs"""
     
    def compute_fgrad(self, loss_grad, fea):
        # initialize values
        num_fea = len(fea[0])
        num_samp = len(fea)
        num_op = len(loss_grad[0])
        sum_loss = np.zeros([num_fea,num_op])
       
        # Compute the loss for each feature
        for fi in range(num_fea):
            for oi in range(num_op):
                hist_grad = self.compute_hgrad(loss_grad[:,oi],fea[:,fi])
                sum_loss[fi,oi] = - sum(abs(hist_grad))
        #for u in range(num_entries):
        #    sum_loss[fi] = sum_loss[fi] - abs(hist_grad[u])

        return sum_loss



    """ The function computes the loss for a single feature 
        Input: loss_grado: loss gradient for a single output values. No of Samples x 1
               fval: single feature selected for all samples. No. of samples x 1

        return: hist_grad: The sum of the loss gradient"""

    def compute_hgrad(self, loss_grado,fval):
        # initialize the values
        num_samp = len(fval)
        hist_grad = np.zeros([self.num_entries])

        # compute the sum of the gradient
        for hi in range(self.num_entries):
            hist_grad[hi] = sum(loss_grado[fval == hi])
        return hist_grad



	""" Function computes classification results according to current weak classifier
        Input: fset: The set test features. No. of test samples x No. of total features

        return: The classification scores of the features based on current weak classifier"""
    def get_weak_scores(self, fset):
		num_samp = len(fset)
		num_op = len(self.luts[0])
		weak_scores = np.zeros([num_samp,num_op])
		for oi in range(num_op):
			a = self.luts[fset[:,self.selected_indices[oi]],oi]
			weak_scores[:,oi] = np.transpose(self.luts[fset[:,self.selected_indices[oi]],oi])
		return weak_scores


""" The class to perform the classification using the set of weak trainer """


class BoostMachine():

    """ Initialize the set of weak trainers and the alpha values (scale)"""
    def __init__(self, num_op):
        self.alpha = []
        self.weak_trainer = []
        self.num_op = num_op

    """ Function adds a weak trainer and the scale into the list
        Input: curr_trainer: the weak trainer learner during a single round of boosting
               curr_alpha: the scale for the curr_trainer"""

    def add_weak_trainer(self, curr_trainer, curr_alpha):
        self.alpha.append(curr_alpha)
        self.weak_trainer.append(curr_trainer)


    """ Function to classify the test features using a set of the trained weak trainers """
    def classify(self, test_features):
        # Initilization
        num_trainer = len(self.weak_trainer)
        num_samp = test_features.shape[0]
        pred_labels = np.ones([num_samp, self.num_op])
        pred_scores = np.zeros([num_samp, self.num_op]) 


        # For each round of boosting calculate the weak scores for that round and add to the total
        for i in range(num_trainer):
            curr_trainer = self.weak_trainer[i]
            weak_scores = curr_trainer.get_weak_scores(test_features)
            pred_scores = pred_scores + self.alpha[i] * weak_scores

        pred_labels[pred_scores <=0] = -1
        return pred_labels



""" The main class to perform the boosting. It iteratively adds new trainer models
    to assemble a strong classifier. In each round of iteration a weak trainer is learned 
    by optimization of a differentiable function. The following parameters are involved


    Parameters:
    num_rnds:      Type int, Default = 100 
                   The number of rounds of boosting. The boosting strategies implemented here
                   (GradBoost and TaylorBoost) are fairly robust to overfitting, so the large
                   number of rounds generally results in a small error rate.

    loss_type:    Type string, Default = 'log'
                  It is the type of loss function to be optimized. Currently we support the
                  following classes of loss function:
                  'log', 'exp', 'symlog', 'symexp' and 'tang'. 
                  'exp' loss function is preferred with StumpTrainer and 'log' with LutTrainer.

    trainer_type:  Type string, Default = 'stump'
                   The type of weak trainer to be learned. Two types of weak trainers are
                   supported currently.

                   'LutTrainer':  It is used for descrete feature types.LUT are used as weak
                                  trainers and Taylor Boost is used as optimization strategy.
                                  Eg: LBP features, MCT features.

                   'StumpTrainer': Decsion Stumps are used as weak trainer and GradBoost is 
                                used as optimization strategy.It can be used with both descrete
                                and continuous type of features 

     num_entries:  Type int, Default = 256
                   This is the parameter for the LutTrainer. It is the 
                   number of entries in the LookUp table. It can be determined from the range of
                   feature values. For examples, for LBP features the number of entries in the 
                   LookUp table is 256.



"""


class Boost:

    def __init__(self, weakTrainer):
        self.num_rnds = 100
        self. num_entries = 256
        self.bl_type = 'log' 
        self.loss_type = 'ept'
        self.lamda = 0.5
        self.selection_type = 'indep'
        self.weak_trainer_type = weakTrainer
							
	

	""" function LUTBoost_train boosts the discrete features (fset) to provide a strong classifier 
	 as a combintaion of weak classifier. Each weak classifier is represented by a feature index
	  and a LUT for univariate case and a set of feature indices and LUTs for multivariate case.

	  fset- (num_sam x num_features) features extracted from the samples
	         features should be discrete
	  labels- class labels of the samples of dimension (#samples x #outputs)
	  param- parameters for the boosting"""
	
    def train(self, fset, targets):
	# Initializations
        # num_samp = len(fset)
        num_op = targets.shape[1]
        machine = BoostMachine(num_op)
        num_samp = fset.shape[0]
        pred_scores = np.zeros([num_samp,num_op])
        fset = fset.astype(int)
        #loss_class = baseloss.base_loss(self.bl_type,self.loss_type)
        loss_class = LOSS_FUNCTIONS[self.bl_type]
        loss_ = loss_class()
        #print loss_
	
        # Start boosting iterations for num_rnds rounds
        for r in range(self.num_rnds):
            if(self.weak_trainer_type == 'LutTrainer'):
                wl = LutTrainer(self.num_entries, self.selection_type, num_op )
            elif (self.weak_trainer_type == 'StumpTrainer'):
                wl = StumpTrainer()

            # Compute the gradient of the loss function, l'(y,f(x)) using loss_ class
            loss_grad = loss_.update_loss_grad(targets,pred_scores)

            # Select the best weak trainer for current round of boosting
            curr_weak_trainer = wl.compute_weak_trainer(fset, loss_grad)

            # Compute the classification scores of the samples based only on the current round weak classifier
            curr_pred_scores = wl.get_weak_scores(fset)
            
            # Initlize the start point for lbfgs minimization
            f0 = np.zeros(num_op)

            # Perform lbfgs minimization and compute the scale for current weak trainer
            ls_res = optimize.fmin_l_bfgs_b(loss_.loss_sum, f0, fprime = loss_.loss_grad_sum, args = (targets, pred_scores, curr_pred_scores)) 
            alpha = ls_res[0]

            # Update the prediction score after adding the score from the current weak classifier
            pred_scores = pred_scores + alpha* curr_pred_scores 

            # Add the current trainer into the boosting machine
            machine.add_weak_trainer(curr_weak_trainer, alpha)
            print "Boosting done for round number : %d" % (r+1)
			
			
        return machine





