import numpy
import matplotlib.pyplot 



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
