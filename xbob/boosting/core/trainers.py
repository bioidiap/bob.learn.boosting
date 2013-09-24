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

class StumpMachine():
    """ The StumpMachine class consist of the core elements of the Stump weak classifier i.e. the threshold,
        the polarity and the feature index corresponding to the weak classifier.  """


    def  __init__(self, threshold = 0, polarity = 0, selected_index = 0):
        """ Initialize the stump classifier"""
        self.threshold = threshold
        self.polarity = polarity
        self.selected_index = numpy.int32(selected_index)

    def feature_indices(self):
      return [self.selected_index]

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
        weak_features = test_features[:,self.selected_index]

        # classify the features and compute the score
        weak_scores[weak_features < self.threshold] = -1
        weak_scores = self.polarity *weak_scores
        return weak_scores


    def get_weak_score(self, feature):
      """Returns the weak score for the given single feature, assuming only a single output.

      Input: a single feature vector of size No. of total features.

      Output: a single number (+1/-1)
      """
      # classify the features and compute the score
      return self.polarity * (-1. if feature[self.selected_index] < self.threshold else 1.)


    def save(self, hdf5File):
      """Saves the current state of this machine to the given HDF5File."""
      hdf5File.set("Index", self.selected_index)
      hdf5File.set("Threshold", self.threshold)
      hdf5File.set("Polarity", self.polarity)

    def load(self, hdf5File):
      """Reads the state of this machine from the given HDF5File."""
      self.selected_index = hdf5File.read("Index")
      self.threshold = hdf5File.read("Threshold")
      self.polarity = hdf5File.read("Polarity")


from .. import StumpMachine as CppStumpMachine

class StumpTrainer():
    """ The weak trainer class for training stumps as classifiers. The trainer is parametrized
    the threshold and the polarity.
    """

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
            selected_feature = numpy.copy(fea[:,i])
            gradient = -numpy.copy(loss_grad)
            polarity[i],threshold[i], gain[i] = self.compute_thresh(selected_feature, gradient)

        #  Find the optimum id and its corresponding trainer
        opt_id = gain.argmax()
        return CppStumpMachine(threshold[opt_id], polarity[opt_id], numpy.int32(opt_id))




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





class LutMachine():
    """ The LUT machine consist of the core elements of the LUT weak classfier i.e. the LUT and
         the feature index corresponding to the weak classifier.  """

    def __init__(self, num_outputs = 0, num_entries = 0):
        """ The function initializes the weak LUT machine.

        The function initializes the look-up-table and the feature indices of the LUT machine.
        Inputs:
        self:
        num_entries: The number of entries for the LUT
                     type: int


        num_outputs: The number of outputs for the classification task.
                    type: Integer

        """
        self.luts = numpy.ones((num_entries, num_outputs), numpy.float64)
        self.selected_indices = numpy.zeros((num_outputs,), numpy.int32)


    def feature_indices(self):
      return self.selected_indices


    def get_weak_scores(self, features):
        """ Function computes classification results according to the LUT machine

        Function classifies the features based on a single LUT machine.

        Input:
        fset: The set test features. No. of test samples x No. of total features

        return:
        weak_scores: The classification scores of the features based on current weak classifier"""

        # Initialize
        num_samp = len(features)
        num_outputs = len(self.luts[0])
        weak_scores = numpy.zeros([num_samp,num_outputs])

        # Compute weak scores
        for output_index in range(num_outputs):
            weak_scores[:,output_index] = numpy.transpose(self.luts[features[:,self.selected_indices[output_index]],output_index])
        return weak_scores



    def get_weak_score(self, feature):
      """Returns the weak score for the given single feature, assuming only a single output.

      Input: a single feature vector of size No. of total features.

      Output: a single number (+1/-1)
      """
      return self.luts[feature[self.selected_indices[0]],0]


    def save(self, hdf5File):
      """Saves the current state of this machine to the given HDF5File."""
      hdf5File.set("LUT", self.luts)
      hdf5File.set("Indices", self.selected_indices)

    def load(self, hdf5File):
      """Reads the state of this machine from the given HDF5File."""
      self.luts = hdf5File.read("LUT")
      self.selected_indices = hdf5File.read("Indices")
      if isinstance(self.selected_indices, int):
        self.selected_indices = numpy.array([self.selected_indices], dtype=numpy.int)



from .. import LUTMachine

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
        self.num_outputs = num_outputs
        self.selection_type = selection_type



    def compute_weak_trainer(self, fea, loss_grad):

        """ The function to learn the weak LutTrainer.

        The function searches for a features index that minimizes the the sum of the loss gradient and computes
        the LUT corresponding to that feature index.

        Inputs:
        self: empty trainer object to be trained

        fea: The training features samples
             type: integer numpy array (#number of samples x number of features)

        loss_grad: The loss gradient values for the training samples
              type: numpy array (#number of samples x #number of outputs)

        Return:
        self: a trained LUT trainer
        """

        # Initializations
        # num_outputs = loss_grad.shape[1]
        fea_grad = numpy.zeros([self.num_entries, self.num_outputs])
        luts = numpy.ones((self.num_entries, self.num_outputs), numpy.float64)
        selected_indices = numpy.ndarray((self.num_outputs,), numpy.int32)

        # Compute the sum of the gradient based on the feature values or the loss associated with each
        # feature index
        sum_loss = self.compute_grad_sum(loss_grad, fea)



        # Select the most discriminative index (or indices) for classification which minimizes the loss
        #  and compute the sum of gradient for that index

        if self.selection_type == 'indep':

            # indep (independent) feature selection is used if all the dimension of output use different feature
            # each of the selected feature minimize a dimension of the loss function

#            selected_indices = [numpy.argmin(col) for col in numpy.transpose(sum_loss)]

            for output_index in range(self.num_outputs):
                curr_id = sum_loss[:,output_index].argmin()
                fea_grad[:,output_index] = self.compute_grad_hist(loss_grad[:,output_index],fea[:,curr_id])
                selected_indices[output_index] = curr_id


        elif self.selection_type == 'shared':

            # for 'shared' feature selection the loss function is summed over multiple dimensions and
            # the feature that minimized this cumulative loss is used for all the outputs

            accum_loss = numpy.sum(sum_loss,1)
            selected_findex = accum_loss.argmin()
            selected_indices = selected_findex*numpy.ones([self.num_outputs,1],'int16')

            for output_index in range(self.num_outputs):
                fea_grad[:,output_index] = self.compute_grad_hist(loss_grad[:,output_index],fea[:,selected_findex])


        # Assign the values to LookUp Table
        luts[fea_grad <= 0.0] = -1
        return LUTMachine(luts, selected_indices)








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
        sum_loss = numpy.zeros([num_fea,self.num_outputs])

        # Compute the loss for each feature
        for feature_index in range(num_fea):
            for output_index in range(self.num_outputs):
                hist_grad = self.compute_grad_hist(loss_grad[:,output_index],fea[:,feature_index])
                sum_loss[feature_index,output_index] = - sum(abs(hist_grad))


        return sum_loss





    def compute_grad_hist(self, loss_grado,features):
        """ The function computes the loss for a single feature.

        Function computes sum of the loss gradient that have same feature values.


        Input: loss_grado: loss gradient for a single output values. No of Samples x 1
               fval: single feature selected for all samples. No. of samples x 1

        return: hist_grad: The sum of the loss gradient"""
        # initialize the values
        # hist_grad = numpy.zeros([self.num_entries])



        # compute the sum of the gradient
        hist_grad, bin_val = numpy.histogram(features, bins = self.num_entries, range = (0,self.num_entries-1), weights = loss_grado)
        # hist_grad = [sum(loss_grado[features == feature_value]) for feature_value in xrange(self.num_entries)]
        #for feature_value in range(self.num_entries):
        #    hist_grad[feature_value] = sum(loss_grado[features == feature_value])
        return hist_grad


"""
class GaussianMachine():

    def __init__(self, num_classes):
        self.means = numpy.zeros(num_classes)
        self.variance = numpy.zeros(num_classes)
        self.selected_index = 0


    def get_weak_scores(self, features):
        num_classes = self.means.shape[0]
        num_features = features.shape[0]
        scores = numpy.zeros([num_features,num_classes])


        for i in range(num_classes):
            mean_i = self.means[i]
            variance_i = self.variance[i]
            feature_i = features[:,self.selected_index]
            denom = numpy.sqrt(2*numpy.pi*variance_i)
            temp = ((feature_i - mean_i)**2)/2*variance_i
            numerator =  numpy.exp(-temp)

            scores[:,i] = numerator/denom


        return scores

class GaussianTrainer():

    def __init__(self, num_classes):
        self.num_classes = num_classes


    def compute_weak_trainer(self, features, loss_grad):

        num_features = features.shape[1]
        means = numpy.zeros([num_features,self.num_classes])
        variances = numpy.zeros([num_features,self.num_classes])
        summed_loss = numpy.zeros(num_features)
        gauss_machine = GaussianMachine(self.num_classes)

        for feature_index in range(num_features):
            single_feature = features[:,feature_index]
            means[feature_index,:], variances[feature_index,:], summed_loss[feature_index] = self.compute_current_loss(single_feature,  loss_grad)
        selected_index = numpy.argmin(summed_loss)
        gauss_machine.selected_index = selected_index
        gauss_machine.means = means[selected_index,:]
        gauss_machine.variance = variances[selected_index,:]
        return gauss_machine



    def compute_current_loss(self, feature, loss_grad):
        num_samples = feature.shape[0]
        mean = numpy.zeros([self.num_classes])
        variance = numpy.zeros(self.num_classes)
        scores = numpy.zeros([num_samples, self.num_classes])

        for class_index in range(self.num_classes):
            samples_i = feature[loss_grad[:,class_index] < 0]
            mean[class_index] = numpy.mean(samples_i)
            variance[class_index] = numpy.std(samples_i)**2
            denom = numpy.sqrt(2*numpy.pi*variance[class_index])
            scores[:,class_index] = numpy.exp(-(((feature - mean[class_index])**2)/2*variance[class_index]))/denom


        # print mean
        scores_sum = numpy.sum(scores)
        return mean, variance, scores_sum

"""
"""
class BayesMachine():

    def __init__(self, num_outputs, num_entries):

        self.luts = numpy.ones((num_entries, num_outputs), dtype = numpy.int)
        self.selected_indices = numpy.zeros([num_outputs,1], 'int16')



    def get_weak_scores(self, features):


        # Initialize
        num_samp = len(features)
        num_outputs = len(self.luts[0])
        weak_scores = numpy.zeros([num_samp,num_outputs])

        # Compute weak scores
        for output_index in range(num_outputs):
            weak_scores[:,output_index] = numpy.transpose(self.luts[features[:,self.selected_indices[output_index]],output_index])
        return weak_scores


class BayesTrainer():




    def __init__(self, num_entries, num_outputs):

        self.num_entries = num_entries
        self.num_outputs = num_outputs
        self.selection_type = selection_type




    def compute_weak_trainer(self, fea, loss_grad):

        # Initializations
        # num_outputs = loss_grad.shape[1]
        fea_grad = numpy.zeros([self.num_entries, self.num_outputs])
        lut_machine = LutMachine(self.num_outputs, self.num_entries)

        # Compute the sum of the gradient based on the feature values or the loss associated with each
        # feature index
        sum_loss = self.compute_grad_sum(loss_grad, fea)



        # Select the most discriminative index (or indices) for classification which minimizes the loss
        #  and compute the sum of gradient for that index

        if self.selection_type == 'indep':

            # indep (independent) feature selection is used if all the dimension of output use different feature
            # each of the selected feature minimize a dimension of the loss function

            selected_indices = [numpy.argmin(col) for col in numpy.transpose(sum_loss)]

            for output_index in range(self.num_outputs):
                curr_id = sum_loss[:,output_index].argmin()
                fea_grad[:,output_index] = self.compute_grad_hist(loss_grad[:,output_index],fea[:,curr_id])
                lut_machine.selected_indices[output_index] = curr_id


        elif self.selection_type == 'shared':

            # for 'shared' feature selection the loss function is summed over multiple dimensions and
            # the feature that minimized this cumulative loss is used for all the outputs

            accum_loss = numpy.sum(sum_loss,1)
            selected_findex = accum_loss.argmin()
            lut_machine.selected_indices = selected_findex*numpy.ones([self.num_outputs,1],'int16')

            for output_index in range(self.num_outputs):
                fea_grad[:,output_index] = self.compute_grad_hist(loss_grad[:,output_index],fea[:,selected_findex])


        # Assign the values to LookUp Table
        lut_machine.luts[fea_grad <= 0.0] = -1
        return lut_machine





    def compute_grad_sum(self, loss_grad, fea):


        # initialize values
        num_fea = len(fea[0])
        num_samp = len(fea)
        sum_loss = numpy.zeros([num_fea,self.num_outputs])

        # Compute the loss for each feature
        for feature_index in range(num_fea):
            for output_index in range(self.num_outputs):
                for feature_value in range(self.num_entries):
                    luts[]



        return sum_loss





    def compute_grad_hist(self, loss_grado,features):

        # initialize the values
        num_samp = len(features)
        hist_grad = numpy.zeros([self.num_entries])

        # compute the sum of the gradient
        for output_index in range(self.num_outputs):
            for feature_value in range(self.num_entries):
                num_feature_i = sum(features == feature_value)
                luts[feature_value,output_index] = sum(loss_grado[features == feature_value])
        return hist_grad
"""

