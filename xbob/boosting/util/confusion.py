import numpy
import matplotlib.pyplot 

def confusion_matrix(expected, predicted):
  """Transforms the prediction list into a confusion matrix
  
  This method takes lists of expected and predicted classes and returns a
  confusion matrix, which represents the percentage of classified examples in
  each combination of "expected class" of samples and "predicated class" of the
  same samples.

  Keyword parameters:

  expected (numpy.ndarray, 1D)
    The ground-thruth

  predicted (numpy.ndarray, 1D)
    The predicted classes with your neural network

  You must combine these scores column wise and determine what are the
  annotated rates (below) for each of the column entries, returning the
  following 2D numpy.ndarray::

    [ TP0    / N0    FP1(0) / N1    FP2(0) / N2 ... ]
    [ FP0(1) / N0    TP1    / N1    FP2(1) / N2 ... ]
    [ FP0(2) / N0    FP1(2) / N1    TP2    / N2 ... ]
    [     ...              ...            ...       ]
    [ FP0(9) / N0    FP1(9) / N1    TP9    / N9 ... ]

  Where:

  TPx / Nx
    True Positive Rate for class ``x``

  FPx(y) / Nz
    Rate of False Positives for class ``y``, from class ``x``. That is,
    elements from class ``x`` that have been **incorrectly** classified as
    ``y``.
  """

  retval = numpy.zeros((10,10), dtype=float)

  for k in range(10):
    pred_k = predicted[expected==k] # predictions that are supposed to be 'k'
    retval[:,k] = numpy.array([len(pred_k[pred_k==p]) for p in range(10)])
    retval[:,k] /= len(pred_k)

  return retval




def display_cm(cm, title_str = "Confusion Matrix"):
    """ The function to display the confusion matrix given the confusion matrix numerically.
    
    Inputs:
    cm: The confusion matrix which to be displayed.
    title_str: The title for the confusion matrix.

    """

    res = matplotlib.pyplot.imshow(cm, cmap=matplotlib.pyplot.cm.summer, interpolation='nearest')

    for x in numpy.arange(cm.shape[0]):
        for y in numpy.arange(cm.shape[1]):
            col = 'white'
            if cm[x,y] > 0.5: col = 'black'
            matplotlib.pyplot.annotate('%.2f' % (cm[x,y],), xy=(y,x), color=col,
              fontsize=8, horizontalalignment='center', verticalalignment='center')

    classes = [str(k) for k in range(10)]

    matplotlib.pyplot.xticks(numpy.arange(10), classes)
    matplotlib.pyplot.yticks(numpy.arange(10), classes, rotation=90)
    matplotlib.pyplot.ylabel("(Prediction)")
    matplotlib.pyplot.xlabel("(Real class)")
    matplotlib.pyplot.title(title_str)
    matplotlib.pyplot.show()
