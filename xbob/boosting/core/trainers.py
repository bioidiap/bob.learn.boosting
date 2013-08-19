""" The module consists of the weak trainers which are used in the boosting framework.
    currently two trainer types are implemented: Stump trainer and Lut trainer. 
    The modules structure is as follows: 

    StumpTrainer class provides the methods to compute the weak stump trainer
     and test the features using these trainers. 

    LutTrainer class provides the methods to compute the weak LUT trainer
     and test the features using these trainers. 

"""

import numpy
import math


class StumpTrainer():
    """ The weak trainer class for training stumps as classifiers. The trainer is parametrized 
    the threshold and the polarity. 
    """

    def  __init__(self):
        """ Initialize the stump classifier"""
        self.threshold = 0
        self.polarity = 0
        self.selected_indices = 0
 


    

    def compute_weak_trainer(self, fea, loss_grad):

        """ The function to compute weak Stump trainer.  
        
        The function computes the weak stump trainer. It is called at each boosting round.
        The best weak stump trainer is chosen to maximize the dot product of the outputs 
        and the weights (gain). The weights in the Adaboost are the negative of the loss gradient
        for exponential loss.


        Inputs: 
        fea: the training feature set
        loss_grad: the gradient of the loss function for the training samples
                          Chose preferable exponential loss function to simulate Adaboost 

        Return: 
        self: a StumpTrainer Object, i.e. the optimal trainer that minimizes the loss
        """

        # Initialization
        numSamp, numFea = fea.shape
        threshold = numpy.zeros([numFea])
        polarity = numpy.zeros([numFea])
        gain = numpy.zeros([numFea])

        # For each feature find the optimum threshold, polarity and the gain
        for i in range(numFea):
            polarity[i],threshold[i], gain[i] = self.compute_thresh(fea[:,i],loss_grad)

        #  Find the optimum id and its corresponding trainer
        opt_id = gain.argmax()
        self.threshold = threshold[opt_id]
        self.polarity = polarity[opt_id]
        self.selected_indices = opt_id
        return self




    def compute_thresh(self, fea ,loss_grad):
        """ Function computes the stump classifier (threshold) for a single feature 
       
        Function to compute the threshold for a single feature. The threshold is computed for 
        the given feature values using the weak learner algorithm of Viola Jones.

        Inputs:

        fea: The feature values for a single index, array of dimension N = No. of training Samples
        loss_grad: The loss gradient values for the training samples. Array of dimension N.
        labels: The class of the training samples. Array of dimension N.

        Return: weak stump classifier for given feature

        threshold: threshold that minimizes the error
        polarity: the polarity or the direction used for stump classification
        gain: gain of the classifier"""


        # The weights for Adaboost are negative of exponential loss gradient
        loss_grad = -loss_grad
        num_samp = fea.shape[0]

        # Sort the feature and rearrange the corresponding weights and feature values
        sorted_id = numpy.argsort(fea)
        fea = fea[sorted_id] 
        loss_grad = loss_grad[sorted_id]

        # For all the threshold compute the dot product
        grad_cs =  numpy.cumsum(loss_grad)
        grad_sum = grad_cs[-1]
        gain = (grad_sum - grad_cs)

        # Find the index that maximizes the dot product
        opt_id = numpy.argmax(numpy.absolute(gain))        
        gain_max = numpy.absolute(gain[opt_id])

        # Find the corresponding threshold value
        threshold = 0.0
        if (opt_id == num_samp-1):
            threshold = fea[opt_id]
        else:
            threshold = (float(fea[opt_id]) + float(fea[opt_id+1]))*0.5

        # Find the polarity or the directionality of the current trainer
        if(gain_max == gain[opt_id]):
            polarity = -1
        else:
            polarity =  1

        return polarity, threshold, gain_max




    def get_weak_scores(self,test_features):

        """ The function to perform classification using a weak stump classifier.
 
         The function computes the classification scores for the test features using 
        a weak stump trainer. Since we use the stump classifier the classification 
        scores are either +1 or -1.
        Input: self: a weak stump trainer
               test_features: A matrix of the test features of dimension. 
                              Num. of Test images x Num. of features
        Return: weak_scores: classification scores of the test features use the weak classifier self
                             Array of dimension =  Num. of samples 
        """
        # Initialize the values
        numSamp = test_features.shape[0]
        weak_scores = numpy.ones([numSamp,1])
  
        # Select feature corresponding to the specific index
        weak_features = test_features[:,self.selected_indices]

        # classify the features and compute the score
        weak_scores[weak_features < self.threshold] = -1
        weak_scores = self.polarity *weak_scores
        return weak_scores





class LutTrainer():
    """ The LutTrainer class contain methods to learn weak trainer using LookUp Tables. 
    It can be used for multi-variate binary classification  """
 

    
    def __init__(self, num_entries, selection_type, num_outputs):
        """ Function to initialize the parameters.

        Function to initialize the weak LutTrainer. Each weak Luttrainer is specified with a 
        LookUp Table and the feature index which corresponds to the feature on which the 
        current classifier has to applied. 

        Inputs: 
        self:
        num_entries: The number of entries for the LUT
                     type: int
        
        selection_type: The feature selection can be either independent or shared. For independent 
                        case the loss function is separately considered for each of the output. For
                        shared selection type the sum of the loss function is taken over the outputs
                        and a single feature is used for all the outputs. See Cosmin's thesis for more details.
                       Type: string {'indep', 'shared'}

        num_outputs: The number of outputs for the classification task. 
                    type: Integer

        """
        self.num_entries = num_entries
        self.luts = numpy.ones((num_entries, num_outputs), dtype = numpy.int)
        self.selection_type = selection_type
        self.selected_indices = numpy.zeros([num_outputs,1], 'int16')
    



    def compute_weak_trainer(self, fea, loss_grad):

        """ The function to learn the weak LutTrainer.  
     
        The function searches for a features index that minimizes the the sum of the loss gradient and computes 
        the LUT corresponding to that feature index. 

        Inputs:
        self: empty trainer object to be trained
        
        fea: The training features samples
             type: integer numpy array (#number of samples x number of features)

        loss_grad: The loss gradient values for the training samples
              type: numpy array (#number of samples)

        Return:
        self: a trained LUT trainer
        """

        # Initializations
        num_outputs = loss_grad.shape[1]
        print num_outputs
        fea_grad = numpy.zeros([self.num_entries,num_outputs])

        # Compute the sum of the gradient based on the feature values or the loss associated with each 
        # feature index
        sum_loss = self.compute_grad_sum(loss_grad, fea)


        # Select the most discriminative index (or indices) for classification which minimizes the loss
        #  and compute the sum of gradient for that index
       
        if self.selection_type == 'indep':

            # indep (independent) feature selection is used if all the dimension of output use different feature
            # each of the selected feature minimize a dimension of the loss function

            selected_indices = [numpy.argmin(col) for col in numpy.transpose(sum_loss)]

            for oi in range(num_outputs):
                curr_id = sum_loss[:,oi].argmin()
                fea_grad[:,oi] = self.compute_grad_hist(loss_grad[:,oi],fea[:,curr_id])
                print oi
                self.selected_indices[oi] = curr_id


        elif self.selection_type == 'shared':

            # for 'shared' feature selection the loss function is summed over multiple dimensions and 
            # the feature that minimized this cumulative loss is used for all the outputs

            accum_loss = numpy.sum(sum_loss,1)
            selected_findex = accum_loss.argmin()
            self.selected_indices = selected_findex*numpy.ones([num_outputs,1],'int16')

            for oi in range(num_outputs):
                fea_grad[:,oi] = self.compute_grad_hist(loss_grad[:,oi],fea[:,selected_findex])
     
        # Assign the values to LookUp Table
        self.luts[fea_grad <= 0.0] = -1
        return self
    



     
    def compute_grad_sum(self, loss_grad, fea):
        """ The function to compute the loss gradient for all the features.

        The function computes the loss for whole set of features. The loss refers to the sum of the loss gradient
        of the features which have the same values.
  
        Inputs: 
        loss_grad: The loss gradient for the features. No. of samples x No. of outputs 
                   Type: float numpy array
        fea: set of features. No. of samples x No. of features
             
        Output: 
        sum_loss: the loss values for all features. No. of samples x No. of outputs"""

        # initialize values
        num_fea = len(fea[0])
        num_samp = len(fea)
        num_outputs = len(loss_grad[0])
        sum_loss = numpy.zeros([num_fea,num_outputs])
       
        # Compute the loss for each feature
        for fi in range(num_fea):
            for oi in range(num_outputs):
                hist_grad = self.compute_grad_hist(loss_grad[:,oi],fea[:,fi])
                sum_loss[fi,oi] = - sum(abs(hist_grad))


        return sum_loss





    def compute_grad_hist(self, loss_grado,fval):
        """ The function computes the loss for a single feature.

        Function computes sum of the loss gradient that have same feature values. 
        
        
        Input: loss_grado: loss gradient for a single output values. No of Samples x 1
               fval: single feature selected for all samples. No. of samples x 1

        return: hist_grad: The sum of the loss gradient"""
        # initialize the values
        num_samp = len(fval)
        hist_grad = numpy.zeros([self.num_entries])

        # compute the sum of the gradient
        for hi in range(self.num_entries):
            hist_grad[hi] = sum(loss_grado[fval == hi])
        return hist_grad




    def get_weak_scores(self, fset):
	""" Function computes classification results according to current weak classifier

        Function classifies the features based on a single weak classifier. 

        Input: 
        fset: The set test features. No. of test samples x No. of total features

        return: 
        weak_scores: The classification scores of the features based on current weak classifier"""
        num_samp = len(fset)
        num_outputs = len(self.luts[0])
        weak_scores = numpy.zeros([num_samp,num_outputs])
        for oi in range(num_outputs):
            a = self.luts[fset[:,self.selected_indices[oi]],oi]
            weak_scores[:,oi] = numpy.transpose(self.luts[fset[:,self.selected_indices[oi]],oi])
        return weak_scores

