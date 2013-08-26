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



class TestIntegralImage(unittest.TestCase):
    """Perform test on integral images"""

    def test_integral_image(self):
        feature_extractor = xbob.boosting.features.local_feature.lbp_feature('lbp')
        img = numpy.array([[1,1,1],
                           [1,1,1],
                           [1,1,1]])

        int_img = numpy.array([[1,2,3],
                               [2,4,6],
                               [3,6,9]])

        returned_integral = feature_extractor.compute_integral_image(img)
        self.assertEqual(returned_integral.shape[0],int_img.shape[0])
        self.assertEqual(returned_integral.shape[1],int_img.shape[1])
        self.assertTrue((returned_integral == int_img).all())



class TestLbpFeatures(unittest.TestCase):
    """Perform test on integral images"""

    """ The neighbourhood is defined as 
        p0 | p1 | p2
        p7 | pc | p3
        p6 | p5 | p4 """

    def test_integral_image(self):
        feature_extractor = xbob.boosting.features.local_feature.lbp_feature('lbp')
        img_values = numpy.array([1,1,1,1,1,1,1,1,1])  # p0,p1,p2,p3,p4,p5,p6,p7,pc
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.lbp(img)
        self.assertTrue(returned_lbp == 255)

        img_values = numpy.array([1,1,1,1,1,1,1,1,0])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.lbp(img)
        self.assertTrue(returned_lbp == 255)

        img_values = numpy.array([0,0,0,0,0,0,0,0,1])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.lbp(img)
        self.assertTrue(returned_lbp == 0)

        img_values = numpy.array([1,0,0,0,0,0,0,0,1])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.lbp(img)
        self.assertTrue(returned_lbp == 1)

        img_values = numpy.array([0,1,0,0,0,0,0,0,1])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.lbp(img)
        self.assertTrue(returned_lbp == 2)

        img_values = numpy.array([0,0,1,0,0,0,0,0,1])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.lbp(img)
        self.assertTrue(returned_lbp == 4)

        img_values = numpy.array([0,0,0,1,0,0,0,0,1])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.lbp(img)
        self.assertTrue(returned_lbp == 8)

        img_values = numpy.array([0,0,0,0,2,0,0,0,1])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.lbp(img)
        self.assertTrue(returned_lbp == 16)

        img_values = numpy.array([0,0,0,0,0,4,0,0,1])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.lbp(img)
        self.assertTrue(returned_lbp == 32)

        img_values = numpy.array([0,0,0,0,0,0,5,0,1])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.lbp(img)
        self.assertTrue(returned_lbp == 64)

        img_values = numpy.array([0,0,0,0,0,0,0,5,1])
        img = get_image_3x3(img_values)
        returned_lbp = feature_extractor.lbp(img)
        self.assertTrue(returned_lbp == 128)
