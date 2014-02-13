from .._boosting import BoostedMachine

import numpy
import logging
logger = logging.getLogger('bob')


class Boosting:
  """ The class to boost the features from  a set of training samples.

  It iteratively adds new weak models to assemble a strong classifier.
  In each round of iteration a weak machine is learned by optimizing a differentiable function.
  The following parameters are involved:


  Parameters:

  trainer_type:  Type string, Default = 'stump'
           The type of weak trainer to be learned. Two types of weak trainers are
           supported currently.

           'LutTrainer':  It is used for discrete feature types.LUT are used as weak
                  trainers and Taylor Boost is used as optimization strategy.
                  Ex.: LBP features, MCT features.

           'StumpTrainer': Decision Stumps are used as weak trainer and GradBoost is
                used as optimization strategy.It can be used with both discrete
                and continuous type of features

  num_rnds:    Type int, Default = 100
           The number of rounds for boosting. The boosting strategies implemented here
           (GradBoost and TaylorBoost) are fairly robust to overfitting, so the large
           number of rounds generally results in a small error rate.

  loss_type:  Type string, Default = 'log'
          It is the type of loss function to be optimized. Currently we support the
          following classes of loss function:
          'log' and 'exp'
          'exp' loss function is preferred with StumpTrainer and 'log' with LutTrainer.



   num_entries:  Type int, Default = 256
           This is the parameter for the LutTrainer. It is the
           number of entries in the LookUp table. It can be determined from the range of
           feature values. For examples, for LBP features the number of entries in the
           LookUp table is 256.


  Example Usage:

  # Initialize the boosting parameter
  num_rounds = 50
  feature_range = 256
  loss_type = 'log'
  selection_type = 'indep'
  boost_trainer = boosting.Boost('LutTrainer', num_rounds, feature_range, loss_type, selection_type )

  # Train machine using training samples
  machine = boost_trainer.train(train_fea, train_targets)

  # Classify the samples using boosted classifier
  prediction_labels = machine.classify(test_fea)


  """


  def __init__(self, weak_trainer, loss_function, number_of_rounds = 20):
    """ The function to initialize the boosting parameters.

    Keyword parameters:

      weak_trainer (trainer.LUTTrainer or trainer.StumpTrainer): The class to train weak machines.

      loss_function (a class derived from loss.LossFunction): The function to define the weights for the weak machines.

      number_of_rounds (int): The number of rounds of boosting, i.e., the number of weak classifiers to select.
    """
    self.m_trainer = weak_trainer
    self.m_loss_function = loss_function
    self.m_number_of_rounds = number_of_rounds


  def get_loss_function(self):
    """Returns the loss function this trainer will use."""
    return self.m_loss_function


  def train(self, training_features, training_targets, boosted_machine = None):
    """The function to train a boosting machine.

    The function boosts the training features and returns a strong classifier as a weighted combination of weak classifiers.

    Keyword parameters:

      training_features (uint16 <#samples, #features> or float <#samples, #features>): Features extracted from the training samples.

      training_targets (float <#samples, #outputs>): The values that the boosted classifier should reach for the given samples.

      boosted_machine (BoostedMachine or None): the machine to add the weak machines to. If not given, a new machine is created.

    Returns

      (BoostedMachine) The boosted machine that is combination of the weak classifiers.
    """

    # Initializations
    if(len(training_targets.shape) == 1):
      training_targets = training_targets[:,numpy.newaxis]

    number_of_samples = training_features.shape[0]
    number_of_outputs = training_targets.shape[1]

    strong_predicted_scores = numpy.zeros((number_of_samples, number_of_outputs))
    weak_predicted_scores = numpy.ndarray((number_of_samples, number_of_outputs))
    if boosted_machine is not None:
      boosted_machine(training_features, strong_predicted_scores)
    else:
      boosted_machine = BoostedMachine()

    # Start boosting iterations for num_rnds rounds
    logger.info("Starting %d rounds of boosting" % self.m_number_of_rounds)
    for round in range(self.m_number_of_rounds):

      logger.debug("Starting round %d" % (round+1))

      # Compute the gradient of the loss function, l'(y,f(x)) using loss_class
      loss_gradient = self.m_loss_function.loss_gradient(training_targets, predicted_scores)

      # Select the best weak machine for current round of boosting
      weak_machine = self.m_trainer.train(training_features, loss_gradient)

      # Compute the classification scores of the samples based only on the current round weak classifier (g_r)
      weak_machine(training_features, weak_predicted_scores)

      # Perform L-BFGS minimization and compute the scale (alpha_r) for current weak machine
      alpha = scipy.optimize.fmin_l_bfgs_b(
          func   = self.m_loss_function.loss_sum,
          x0     = numpy.zeros(number_of_outputs),
          fprime = loss_func.loss_grad_sum,
          args   = (targets, pred_scores, curr_pred_scores)
      )[0]

      # Update the prediction score after adding the score from the current weak classifier f(x) = f(x) + alpha_r*g_r
      strong_predicted_scores += alpha * weak_predicted_scores

      # Add the current weak machine into the boosting machine
      boosted_machine.add_weak_machine(weak_machine, alpha)

      logger.info("Finished round %d / %d" % (round+1, self.m_number_of_rounds))

    return boosted_machine

