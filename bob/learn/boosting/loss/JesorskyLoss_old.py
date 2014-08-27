from .LossFunction import LossFunction

import math
import numpy

class JesorskyLoss (LossFunction):
  """This class computes the Jesorsky loss that is used in regression tasks like feature localization."""

  def _inter_eye_distance(self, targets):
    """Computes the inter-eye distance from the given target vector.
    It assumes that the eyes are stored as the first two elements in the vector,
    as: [0]: re_y [1]: re_x, [2]: le_y, [3]: re_x
    """
    return math.sqrt((targets[0] - targets[2])**2 + (targets[1] - targets[3])**2)

  def loss(self, targets, scores):
    """Computes the Jesorsky loss for the given target and score vectors.
    Both vectors are assumed to have contained feature positions in y and x,
    and the first four values correspond to the eye locations:
    [0]: re_y [1]: re_x, [2]: le_y, [3]: re_x

    Keyword parameters:

      targets (float <#samples, #outputs>): The target values that should be reached.

      scores (float <#samples, #outputs>): The scores provided by the classifier.

    Returns
      (float <#samples, 1>): One error for each target/score pair.
    """
    # compute one error for each sample
    errors = numpy.zeros((targets.shape[0],1))
    for i in range(targets.shape[0]):
      # compute inter-eye-distance
      scale = 0.5/self._inter_eye_distance(targets[i])
      # compute error for all positions
      # which are assumed to be 2D points
      for j in range(0, targets.shape[1], 2):
        dx = scores[i,j] - targets[i,j]
        dy = scores[i,j+1] - targets[i,j+1]
        # sum errors
        errors[i,0] += math.sqrt(dx**2 + dy**2) * scale

    return errors


  def loss_gradient(self, targets, scores):
    """Computes the gradient of the Jesorsky loss for the given target and score vectors
    Both vectors are assumed to have contained feature positions in y and x,
    and the first four values correspond to the eye locations:
    [0]: re_y [1]: re_x, [2]: le_y, [3]: re_x

    Keyword parameters:

      targets (float <#samples, #outputs>): The target values that should be reached.

      scores (float <#samples, #outputs>): The scores provided by the classifier.

    Returns
      (float <#samples, #outputs>): One gradient vector for each target/score pair.
    """
    # allocate memory for the gradients
    gradient = numpy.ndarray(targets.shape, numpy.float)
    # iterate over all samples
    for i in range(targets.shape[0]):
      # compute inter-eye-distance
      scale = 0.5/self._inter_eye_distance(targets[i])
      # compute gradient for all elements in the vector
      # which are assumed to be 2D points
      for j in range(0, targets.shape[1], 2):
        dx = scores[i,j] - targets[i,j]
        dy = scores[i,j+1] - targets[i,j+1]
        error = math.sqrt(dx**2 + dy**2)
        # set gradient
        gradient[i,j] = dx * scale / error
        gradient[i,j+1] = dy * scale / error

    return gradient
