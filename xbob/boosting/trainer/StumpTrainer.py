from .._boosting import StumpMachine

import numpy

class StumpTrainer():
  """ The class for training weak stump classifiers.
  The weak stump is parameterized the threshold and the polarity.
  """

  def train(self, training_features, loss_gradient):

    """ The function to compute a weak stump machine.

    The function computes the weak stump machine.
    It is called at each boosting round.
    The best weak machine is chosen to maximize the dot product of the outputs and the weights (gain).
    The weights are the negative of the loss gradient for exponential loss.

    Keyword parameters
      training_features (int<#number of samples, #number of features>): The training features samples

      loss_gradient (float<#number of samples>): The loss gradient values for the training samples

    Returns
      A (weak) StumpMachine
    """

    # Initialization
    number_of_features = training_features.shape[1]
    threshold = numpy.zeros(number_of_features)
    polarity = numpy.zeros(number_of_features)
    gain = numpy.zeros(number_of_features)

    # For each feature find the optimum threshold, polarity and the gain
    for i in range(number_of_features):
      polarity[i], threshold[i], gain[i] = self.compute_threshold(training_features[:,i], -loss_gradient)

    #  Find the optimum id and its corresponding trainer
    best_index = gain.argmax()
    return StumpMachine(threshold[best_index], polarity[best_index], numpy.int32(best_index))




  def compute_threshold(self, features, gradient):
    """Computes the stump classifier threshold for a single feature

    Function to compute the threshold for a single feature. The threshold is computed for
    the given feature values using the weak learner algorithm of Viola Jones.

    Keyword parameters
      features (float<#number of samples>): The feature values for a single index

      gradient (float<#number of samples>): The negative loss gradient values for the training samples


    Returns a triplet containing:
      threshold (float): threshold that minimizes the error
      polarity (float): the polarity or the direction used for stump classification
      gain (float): gain of the classifier
    """
    # The weights for the weak machine are negative of exponential loss gradient

    # Sort the feature and rearrange the corresponding weights and feature values
    sort_indices = numpy.argsort(features)
    features = features.copy()[sort_indices]
    gradient = gradient.copy()[sort_indices]

    # For all the threshold compute the dot product
    grad_cs =  numpy.cumsum(gradient)
    grad_sum = grad_cs[-1]
    gain = (grad_sum - grad_cs)

    # Find the index that maximizes the dot product
    best_index = numpy.argmax(numpy.absolute(gain))
    gain_max = numpy.absolute(gain[best_index])

    # Find the corresponding threshold value
    threshold = 0.0
    if (best_index == features.shape[0]-1):
        threshold = features[best_index]
    else:
        threshold = (float(features[best_index]) + float(features[best_index+1]))*0.5

    # Find the polarity or the directionality of the current trainer
    if(gain_max == gain[best_index]):
        polarity = -1
    else:
        polarity =  1

    # return polarity, threshold and the gain
    return polarity, threshold, gain_max

