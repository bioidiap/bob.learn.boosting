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




class TesttlbpFeatures(unittest.TestCase):
    """Perform test for mlbp features"""

    """ The neighbourhood is defined as 
        p0 | p1 | p2
        p7 | pc | p3
        p6 | p5 | p4 """

    def test_tlbp_image(self):
        feature_extractor = xbob.boosting.features.local_feature.lbp_feature('mlbp')
        img_values = numpy.array([1,1,1,1,1,1,1,1,1])  # p0,p1,p2,p3,p4,p5,p6,p7,pc
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.tlbp(img)
        self.assertTrue(returned_lbp == 255)

        img_values = numpy.array([1,1,1,1,1,1,1,1,0])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.tlbp(img)
        self.assertTrue(returned_lbp == 255)

        img_values = numpy.array([0,0,0,0,0,0,0,0,1])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.tlbp(img)
        self.assertTrue(returned_lbp == 255)

        img_values = numpy.array([7,0,1,2,3,4,5,6,100])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.tlbp(img)
        self.assertTrue(returned_lbp == 1)

        img_values = numpy.array([6,7,0,1,2,3,4,5,100])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.tlbp(img)
        self.assertTrue(returned_lbp == 2)

        img_values = numpy.array([5,6,7,0,1,2,3,4,100])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.tlbp(img)
        self.assertTrue(returned_lbp == 4)

        img_values = numpy.array([4,5,6,7,0,1,2,3,100])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.tlbp(img)
        self.assertTrue(returned_lbp == 8)

        img_values = numpy.array([3,4,5,6,7,0,1,2,100])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.tlbp(img)
        self.assertTrue(returned_lbp == 16)

        img_values = numpy.array([2,3,4,5,6,7,0,1,100])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.tlbp(img)
        self.assertTrue(returned_lbp == 32)

        img_values = numpy.array([1,2,3,4,5,6,7,0,100])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.tlbp(img)
        self.assertTrue(returned_lbp == 64)

        img_values = numpy.array([0,1,2,3,4,5,6,7,100])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.tlbp(img)
        self.assertTrue(returned_lbp == 128)
