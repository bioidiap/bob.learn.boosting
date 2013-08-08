import numpy
import math
from scipy import optimize



class ExpLossFunction():
    """ The class to implement the exponential loss function for the boosting framework. 
    """


    def update_loss(self, targets, scores):
        """The function computes the exponential loss values using prediction scores and targets.

        Inputs: 
        targets: The targets for the samples
                 type: numpy array (# number of samples x #number of outputs)
        
        scores: The current prediction scores for the samples.
                type: numpy array (# number of samples) 

        Return:
        loss: The loss values for the samples     """

        return numpy.exp(-(targets * scores))
        #return loss 

    def update_loss_grad(self, targets, scores):
        """The function computes the gradient of the exponential loss function using prediction scores and targets.

        Inputs: 
        targets: The targets for the samples
                 type: numpy array (# number of samples x #number of outputs)
        
        scores: The current prediction scores for the samples.
                type: numpy array (# number of samples) 

        Return:
        gradient: The loss gradient values for the samples     """
        loss = numpy.exp(-(targets * scores))
        return -targets * loss
        #return loss_grad

    def loss_sum(self, *args):
        """The function computes the sum of the exponential loss which is used to find the optmized values of alpha (x).
         
        The functions computes sum of loss values which is required during the linesearch step for the optimization of the alpha.
        This function is given as the input for the lbfgs optimization function. 

        Inputs: 
        x: The current value of the alpha.
           type: float

        targets: The targets for the samples
                 type: numpy array (# number of samples x #number of outputs)
        
        pred_scores: The cummulative prediction scores of the samples until the previous round of the boosting.
                 type: numpy array (# number of samples) 

        curr_scores: The prediction scores of the samples for the current round of the boosting.
                 type: numpy array (# number of samples) 


        Return:
        sum_loss: The sum of the loss values for the current value of the alpha    
                 type: float"""

        # initialize the values
        x = args[0]
        targets = args[1]
        pred_scores = args[2]
        weak_scores = args[3]

        # compute the scores and loss for the current alpha
        curr_scores_x = pred_scores + x*weak_scores
        loss = self.update_loss(targets, curr_scores_x)

        # compute the sum of the loss
        sum_loss = numpy.sum(loss,0)
        return sum_loss
        

    def loss_grad_sum(self, *args):
        """The function computes the sum of the exponential loss which is used to find the optmized values of alpha (x).
         
        The functions computes sum of loss values which is required during the linesearch step for the optimization of the alpha.
        This function is given as the input for the lbfgs optimization function. 

        Inputs: 
        x: The current value of the alpha.
           type: float

        targets: The targets for the samples
                 type: numpy array (# number of samples x #number of outputs)
        
        pred_scores: The cummulative prediction scores of the samples until the previous round of the boosting.
                 type: numpy array (# number of samples) 

        curr_scores: The prediction scores of the samples for the current round of the boosting.
                 type: numpy array (# number of samples) 


        Return:
        sum_loss: The sum of the loss gradient values for the current value of the alpha    
                 type: float"""
        # initilize the values
        x = args[0]
        targets = args[1]
        pred_scores = args[2]
        weak_scores = args[3]

        # compute the loss gradient for the updated score
        curr_scores_x = pred_scores + x*weak_scores
        loss_grad = self.update_loss_grad(targets, curr_scores_x)

        # take the sum of the loss gradient values
        sum_grad = numpy.sum(loss_grad*weak_scores, 0)
        return sum_grad




"""Log loss function """
class LogLossFunction():
    """ The class to implement the logit loss function for the boosting framework. 
    """
    def update_loss(self, targets, scores):
        """The function computes the exponential loss values using prediction scores and targets.

        Inputs: 
        targets: The targets for the samples
                 type: numpy array (# number of samples x #number of outputs)
        
        scores: The current prediction scores for the samples.
                type: numpy array (# number of samples) 

        Return:
        loss: The loss values for the samples     """
        e = numpy.exp(-(targets * scores))
        return numpy.log(1 + e)


    def update_loss_grad(self, targets, scores):
        """The function computes the gradient of the exponential loss function using prediction scores and targets.

        Inputs: 
        targets: The targets for the samples
                 type: numpy array (# number of samples x #number of outputs)
        
        scores: The current prediction scores for the samples.
                type: numpy array (# number of samples) 

        Return:
        gradient: The loss gradient values for the samples     """
        e = numpy.exp(-(targets * scores))
        denom = 1/(1 + e)
        return - targets* e* denom

    def loss_sum(self, *args):
        """The function computes the sum of the logit loss which is used to find the optmized values of alpha (x).
         
        The functions computes sum of loss values which is required during the linesearch step for the optimization of the alpha.
        This function is given as the input for the lbfgs optimization function. 

        Inputs: 
        x: The current value of the alpha.
           type: float

        targets: The targets for the samples
                 type: numpy array (# number of samples x #number of outputs)
        
        pred_scores: The cummulative prediction scores of the samples until the previous round of the boosting.
                 type: numpy array (# number of samples) 

        curr_scores: The prediction scores of the samples for the current round of the boosting.
                 type: numpy array (# number of samples) 


        Return:
        sum_loss: The sum of the loss values for the current value of the alpha    
                 type: float"""

        x = args[0]
        targets = args[1]
        pred_scores = args[2]
        weak_scores = args[3]
        curr_scores_x = pred_scores + x*weak_scores
        loss = self.update_loss(targets, curr_scores_x)
        sum_l = numpy.sum(loss,0)
        return sum_l
        
    #@abstractmethod
    def loss_grad_sum(self, *args):
        """The function computes the sum of the logit loss gradient which is used to find the optmized values of alpha (x).
         
        The functions computes sum of loss values which is required during the linesearch step for the optimization of the alpha.
        This function is given as the input for the lbfgs optimization function. 

        Inputs: 
        x: The current value of the alpha.
           type: float

        targets: The targets for the samples
                 type: numpy array (# number of samples x #number of outputs)
        
        pred_scores: The cummulative prediction scores of the samples until the previous round of the boosting.
                 type: numpy array (# number of samples) 

        curr_scores: The prediction scores of the samples for the current round of the boosting.
                 type: numpy array (# number of samples) 


        Return:
        sum_loss: The sum of the loss gradient values for the current value of the alpha    
                 type: float"""
        x = args[0]
        targets = args[1]
        pred_scores = args[2]
        weak_scores = args[3]
        curr_scores_x = pred_scores + x*weak_scores
        loss_grad = self.update_loss_grad( targets, curr_scores_x)
        sum_g = numpy.sum(loss_grad*weak_scores, 0)
        return sum_g


    """def loss_sum(self, targets, scores):
        loss = self.update_loss(self,targets, scores)
        return np.sum(loss, 0)

    def loss_grad_sum(self, targets, scores)
        loss_grad = self.update_loss_grad(self, targets, scores)"""


"""Tangent loss function """

class TangLossFunction():
    def update_loss(self, targets, scores):
        loss = (2* numpy.arctan(targets * scores) -1)**2
        return loss

    def update_loss_grad(self, targets, scores):
        m = targets*scores
        numer = 4*(2*numpy.arctan(m) -1)
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
        return numpy.sum(loss, 0)
        
    #@abstractmethod
    def loss_grad_sum(self, *args):
        x = args[0]
        targets = args[1]
        pred_scores = args[2]
        weak_scores = args[3]
        curr_scores_x = pred_scores + x*weak_scores
        loss_grad = self.update_loss_grad( targets, curr_scores_x)
        return numpy.sum(loss_grad*weak_scores, 0)



LOSS_FUNCTIONS = {'log':LogLossFunction,
                  'exp':ExpLossFunction,
                  'tang':TangLossFunction}


