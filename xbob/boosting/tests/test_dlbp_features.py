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




class TestdlbpFeatures(unittest.TestCase):
    """Perform test for dlbp features"""

    """ The neighbourhood is defined as 
        p0 | p1 | p2
        p7 | pc | p3
        p6 | p5 | p4 """

    def test_dlbp_image(self):
        feature_extractor = xbob.boosting.features.local_feature.lbp_feature('dlbp')
        img_values = numpy.array([1,1,1,1,1,1,1,1,1])  # p0,p1,p2,p3,p4,p5,p6,p7,pc
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.dlbp(img)
        self.assertTrue(returned_lbp == 255)


        img_values = numpy.array([20,1,1,1,10,10,10,10,5]) 
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.dlbp(img)
        print returned_lbp
        self.assertTrue(returned_lbp == 3)

        img_values = numpy.array([1,20,1,1,10,10,10,10,5]) 
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.dlbp(img)
        self.assertTrue(returned_lbp == 12)

        img_values = numpy.array([1,1,20,1,10,10,10,10,5]) 
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.dlbp(img)
        self.assertTrue(returned_lbp == 48)

        img_values = numpy.array([1,1,1,20,10,10,10,10,5]) 
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.dlbp(img)
        self.assertTrue(returned_lbp == 192)

        
