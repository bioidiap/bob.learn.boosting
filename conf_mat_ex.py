import numpy
import matplotlib.pyplot as mpl


def display_cm():
    score = numpy.loadtxt('conf_mat.out',delimiter = ',')
    cm = score/numpy.sum(score,1)
    res = mpl.imshow(cm, cmap=mpl.cm.summer, interpolation='nearest')

    for x in numpy.arange(cm.shape[0]):
      for y in numpy.arange(cm.shape[1]):
          col = 'white'
          if cm[x,y] > 0.5: col = 'black'
          mpl.annotate('%.2f' % (100*cm[x,y],), xy=(y,x), color=col,
              fontsize=8, horizontalalignment='center', verticalalignment='center')

    classes = [str(k) for k in range(10)]

    mpl.xticks(numpy.arange(10), classes)
    mpl.yticks(numpy.arange(10), classes, rotation=90)
    mpl.ylabel("(Your prediction)")
    mpl.xlabel("(Real class)")
    mpl.title("Confusion Matrix" )
    mpl.show()
