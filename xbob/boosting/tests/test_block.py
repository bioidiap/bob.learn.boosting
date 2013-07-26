import numpy
import lbp_features
import math

img = numpy.cumsum(numpy.cumsum(numpy.ones([10,10]),0),1)
int_img = lbp_features.compute_mlbp(img,3,3)
print int_img
