import numpy
import math


class LossFunction:

  def loss(self, targets, scores):
    raise NotImplementedError("This is a pure abstract function. Please implement that in your derived class.")

  def loss_gradient(self, targets, scores):
    raise NotImplementedError("This is a pure abstract function. Please implement that in your derived class.")


  def loss_sum(self, alpha, targets, prediction_scores, weak_scores):
    """The function computes the sum of the loss which is used to find the optimized values of alpha (x).

    The functions computes sum of loss values which is required during the linesearch step for the optimization of the alpha.
    This function is given as the input for the lbfgs optimization function.

    Inputs:
    alpha: The current value of the alpha.
       type: float

    targets: The targets for the samples
             type: numpy array (# number of samples x # number of outputs)

    pred_scores: The cumulative prediction scores of the samples until the previous round of the boosting.
             type: numpy array (# number of samples x # number of outputs)

    curr_scores: The prediction scores of the samples for the current round of the boosting.
             type: numpy array (# number of samples x # number of outputs)


    Return:
    sum_loss: The sum of the loss values for the current value of the alpha
             type: float
    """

    # compute the scores and loss for the current alpha
    curr_scores = prediction_scores + alpha * weak_scores
    loss = self.loss(targets, curr_scores)

    # compute the sum of the loss
    return numpy.sum(loss, 0)


  def loss_grad_sum(self, alpha, targets, prediction_scores, weak_scores):
    """The function computes the gradient as the sum of the derivatives per sample which is used to find the optimized values of alpha (x).

    The functions computes sum of loss values which is required during the linesearch step for the optimization of the alpha.
    This function is given as the input for the lbfgs optimization function.

    Inputs:
    alpha: The current value of the alpha.
       type: float

    targets: The targets for the samples
             type: numpy array (# number of samples x # number of outputs)

    pred_scores: The cumulative prediction scores of the samples until the previous round of the boosting.
             type: numpy array (# number of samples x # number of outputs)

    curr_scores: The prediction scores of the samples for the current round of the boosting.
             type: numpy array (# number of samples x # number of outputs)


    Returns
      The sum of the loss gradient values for the current value of the alpha
      type: numpy array (# number of outputs)
    """

    # compute the loss gradient for the updated score
    curr_scores = prediction_scores + alpha * weak_scores
    loss_grad = self.loss_gradient(targets, curr_scores)

    # take the sum of the loss gradient values
    return numpy.sum(loss_grad * weak_scores, 0)


class ExpLossFunction(LossFunction):
  """ The class to implement the exponential loss function for the boosting framework.
  """
  def loss(self, targets, scores):
    """The function computes the exponential loss values using prediction scores and targets.

    Inputs:
    targets: The targets for the samples
             type: numpy array (# number of samples x # number of outputs)

    scores: The current prediction scores for the samples.
            type: numpy array (# number of samples x # number of outputs)

    Returns
      The loss values for the samples
    """
    return numpy.exp(-(targets * scores))

  def loss_gradient(self, targets, scores):
    """The function computes the gradient of the exponential loss function using prediction scores and targets.

    Inputs:
    targets: The targets for the samples
             type: numpy array (# number of samples x # number of outputs)

    scores: The current prediction scores for the samples.
            type: numpy array (# number of samples x # number of outputs)

    Returns
      The loss gradient values for the samples
    """
    loss = numpy.exp(-(targets * scores))
    return -targets * loss


class LogLossFunction(LossFunction):
  """ The class to implement the logit loss function for the boosting framework.
  """
  def loss(self, targets, scores):
    """The function computes the exponential loss values using prediction scores and targets.

    Inputs:
    targets: The targets for the samples
             type: numpy array (# number of samples x # number of outputs)

    scores: The current prediction scores for the samples.
            type: numpy array (# number of samples x # number of outputs)

    Returns
      The loss values for the samples
    """
    e = numpy.exp(-(targets * scores))
    return numpy.log(1 + e)


  def loss_gradient(self, targets, scores):
    """The function computes the gradient of the exponential loss function using prediction scores and targets.

    Inputs:
    targets: The targets for the samples
             type: numpy array (# number of samples x # number of outputs)

    scores: The current prediction scores for the samples.
            type: numpy array (# number of samples x # number of outputs)

    Returns
      The loss gradient values for the samples
    """
    e = numpy.exp(-(targets * scores))
    denom = 1./(1. + e)
    return - targets * e * denom



class TangLossFunction():
  """Tangent loss function """
  def loss(self, targets, scores):
      return (2. * numpy.arctan(targets * scores) -1)**2

  def loss_gradient(self, targets, scores):
    m = targets*scores
    numer = 4.*(2. * numpy.arctan(m) - 1.)
    denom = 1. + m**2
    return numer/denom



class JesorskyLossFunction (LossFunction):

  def _inter_eye_distance(self, targets):
    """Computes the inter eye distance from the given target vector.
    It assumes that the eyes are stored as the first two elements in the vector,
    as: [0]: re_y [1]: re_x, [2]: le_y, [3]: re_x
    """
    return math.sqrt((targets[0] - targets[2])**2 + (targets[1] - targets[3])**2)

  def loss(self, targets, scores):
    """Computes the jesorsky loss for the given target and score vectors."""

    """
    errors = numpy.ndarray(targets.shape[0], numpy.float)
    for i in range(targets.shape[0]):
      scale = 0.5/self._inter_eye_distance(targets[i])
      errors[i] = math.sqrt(numpy.sum((targets[i] - scores[i])**2)) * scale
    """
    errors = numpy.zeros((targets.shape[0],1), numpy.float)
    for i in range(targets.shape[0]):
      scale = 0.5/self._inter_eye_distance(targets[i])
      for j in range(0, targets.shape[1], 2):
        dx = scores[i,j] - targets[i,j]
        dy = scores[i,j+1] - targets[i,j+1]
        errors[i,0] += math.sqrt(dx**2 + dy**2) * scale

    return errors

  def loss_gradient(self, targets, scores):
    """Computes the gradient of the jesorsky loss."""
    gradient = numpy.ndarray(targets.shape, numpy.float)
    for i in range(targets.shape[0]):
      scale = 0.5/self._inter_eye_distance(targets[i])
      for j in range(0, targets.shape[1], 2):
        dx = scores[i,j] - targets[i,j]
        dy = scores[i,j+1] - targets[i,j+1]
        error = math.sqrt(dx**2 + dy**2)
        gradient[i,j] = dx * scale / error
        gradient[i,j+1] = dy * scale / error

    return gradient


LOSS_FUNCTIONS = {'log':LogLossFunction,
                  'exp':ExpLossFunction,
                  'tang':TangLossFunction,
                  'jesorsky':JesorskyLossFunction}


