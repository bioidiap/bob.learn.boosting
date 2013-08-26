import unittest
import random
import xbob.boosting
import numpy

def get_image_3x3(val):
    img = numpy.zeros([3,3])
    img[0,0] = val[0]
    img[0,1] = val[1]
    img[0,2] = val[2]
    img[1,2] = val[3]
    img[2,2] = val[4]
    img[2,1] = val[5]
    img[2,0] = val[6]
    img[1,0] = val[7]
    img[1,1] = val[8]
    return img




class TestmlbpFeatures(unittest.TestCase):
    """Perform test for mlbp features"""

    """ The neighbourhood is defined as 
        p0 | p1 | p2
        p7 | pc | p3
        p6 | p5 | p4 """

    def test_mlbp_image(self):
        feature_extractor = xbob.boosting.features.local_feature.lbp_feature('mlbp')
        img_values = numpy.array([1,1,1,1,1,1,1,1,1])  # p0,p1,p2,p3,p4,p5,p6,p7,pc, mean = 1
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.mlbp(img)
        self.assertTrue(returned_lbp == 255)

        img_values = numpy.array([1,1,1,1,1,1,1,1,0])  # mean = 1
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.mlbp(img)
        self.assertTrue(returned_lbp == 255)

        img_values = numpy.array([0,0,0,0,0,0,0,0,1])  # mean = 0
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.mlbp(img)
        self.assertTrue(returned_lbp == 255)

        img_values = numpy.array([1,0,0,0,0,0,0,0,100]) # mean = 0.125, first bit pass 
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.mlbp(img)
        self.assertTrue(returned_lbp == 1)

        img_values = numpy.array([1,1,0,0,0,0,0,0,100]) # mean = 0.25, first two bits pass 
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.mlbp(img)
        self.assertTrue(returned_lbp == 3)

        img_values = numpy.array([1,1,1,0,0,0,0,0,100]) # mean = 3/8, first three bits pass 
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.mlbp(img)
        self.assertTrue(returned_lbp == 7)

        img_values = numpy.array([1,1,1,1,0,0,0,0,100]) # mean = 4/8, first four bits pass 
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.mlbp(img)
        self.assertTrue(returned_lbp == 15)

        img_values = numpy.array([1,1,1,1,1,0,0,0,100]) # mean = 5/8, first five bits pass 
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.mlbp(img)
        self.assertTrue(returned_lbp == 31)

        img_values = numpy.array([1,1,1,1,1,1,0,0,100]) # mean = 6/8, first six bits pass  
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.mlbp(img)
        self.assertTrue(returned_lbp == 63)

        img_values = numpy.array([1,1,1,1,1,1,1,0,100]) # mean = 7/8, first seven bits pass  
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.mlbp(img)
        self.assertTrue(returned_lbp == 127)

