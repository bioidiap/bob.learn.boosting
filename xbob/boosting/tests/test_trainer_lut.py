import unittest
import random
import xbob.boosting
import numpy
import bob


class TestLutTrainer(unittest.TestCase):
    """Class to test the LUT trainer """

    def test_hist_grad(self):

        num_feature = 100
        range_feature = 10
        trainer = xbob.boosting.core.trainers.LutTrainer(range_feature,'indep', 1)

        features = numpy.array([2, 8, 4, 7, 1, 0, 6, 3, 6, 1, 7, 0, 6, 8, 3, 6, 8, 2, 6, 9, 4, 6,
                                2, 0, 4, 9, 7, 4, 1, 3, 9, 9, 3, 3, 5, 2, 4, 0, 1, 3, 8, 8, 6, 7,
                                3, 0, 6, 7, 4, 0, 6, 4, 1, 2, 4, 2, 1, 9, 3, 5, 5, 8, 8, 4, 7, 4,
                                1, 5, 1, 8, 5, 4, 2, 4, 5, 3, 0, 0, 6, 2, 4, 7, 1, 4, 1, 4, 4, 4,
                                1, 4, 7, 5, 6, 9, 7, 5, 3, 3, 6, 6])

        loss_grad = numpy.ones(100)

        hist_value, bins = numpy.histogram(features,range(range_feature +1))
        sum_grad = trainer.compute_grad_hist(loss_grad,features)
        self.assertEqual(sum_grad.shape[0],range_feature)
        self.assertTrue((sum_grad == hist_value).all())



    def test_lut_selected_index(self):

        num_samples = 100
        max_feature = 20
        dimension_feature = 10

        selected_index = 5
        range_feature = max_feature
        trainer = xbob.boosting.core.trainers.LutTrainer(range_feature,'indep', 1)

        data_file = bob.io.File('xbob/boosting/tests/datafile.hdf5', 'r')
        #features = bob.io.load('test_data.hdf5')
        features = data_file.read()

        x_train1 = numpy.copy(features)
        x_train1[x_train1[:,selected_index] >=10, selected_index] = 9
        x_train2 = numpy.copy(features)
        x_train2[x_train2[:,selected_index] < 10, selected_index] = 10
        x_train = numpy.vstack((x_train1, x_train2))

        y_train = numpy.vstack((numpy.ones([num_samples,1]),-numpy.ones([num_samples,1])))

        scores = numpy.zeros([2*num_samples,1])
        loss_grad = -y_train*(numpy.exp(y_train*scores))

        machine = trainer.compute_weak_trainer(x_train, loss_grad)

        self.assertTrue((machine.luts[0:9] == -1).all())   # The values of the LUT are negative of the classes sign
        self.assertTrue((machine.luts[10:] ==  1).all())



    def test_lut_selected_index(self):

        num_samples = 100
        max_feature = 20
        dimension_feature = 10
        delta = 5
        selected_index = 5
        range_feature = max_feature + delta
        trainer = xbob.boosting.core.trainers.LutTrainer(range_feature,'indep', 1)
        data_file = bob.io.File('xbob/boosting/tests/datafile.hdf5', 'r')

        features = data_file.read()

        x_train1 = numpy.copy(features)
        x_train2 = numpy.copy(features)
        x_train = numpy.vstack((x_train1, x_train2))
        x_train[0:num_samples,selected_index] = x_train[0:num_samples,selected_index] + delta
        y_train = numpy.vstack((numpy.ones([num_samples,1]),-numpy.ones([num_samples,1])))

        scores = numpy.zeros([2*num_samples,1])
        loss_grad = -y_train*(numpy.exp(y_train*scores))

        machine = trainer.compute_weak_trainer(x_train, loss_grad)

        self.assertEqual(machine.feature_indices()[0], selected_index)



