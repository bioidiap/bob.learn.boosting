#!/usr/bin/env python

"""The test script to perform the binary classification on the digits from the MNIST dataset.
   The MNIST data is exported using the xbob.db.mnist module which provide the train and test 
   partitions for the digits. Pixel values of grey scale images are used as features and the
   available algorithms for classification are Lut based Boosting and Stump based Boosting.
   The script test digits provided by the command line. Thus it conducts only one binary classifcation test. 


"""


import xbob.db.mnist
import numpy
import sys, getopt
import argparse
import string
from ..core import boosting
from ..util import confusion
import matplotlib.pyplot as mpl
 

def main():

    parser = argparse.ArgumentParser(description = " The arguments for the boosting. ")
    parser.add_argument('-r', default = 20, dest = "num_rnds", type = int, help = "The number of round for the boosting.")
    parser.add_argument('-d', default = 10, dest = "num_digits", type = int, help = "The number of digits to be be tested.")
    parser.add_argument('-l', default = 'exp', dest = "loss_type", type= str, choices = {'log','exp'}, help = "The type of the loss function. Logit and Exponential functions are the avaliable options")
    parser.add_argument('-s', default = 'indep', dest = "selection_type", choices = {'indep', 'shared'}, type = str, help = "The feature selection type for the LUT based trainer. For multivarite case the features can be selected by sharing or independently ")
    parser.add_argument('-n', default = 256, dest = "num_entries", type = int, help = "The number of entries in the LookUp table. It is the range of the feature values, e.g. if LBP features are used this values is 256.")

    args = parser.parse_args()

    # download the dataset
    db_object = xbob.db.mnist.Database()
    # Hardcode the number of digits
    num_digits = args.num_digits


    # get the data (features and labels) for the selected digits from the xbob_db_mnist class functions
    fea_train, label_train = db_object.data('train',labels = range(num_digits))
    fea_test, label_test = db_object.data('test', labels = range(num_digits))


    # Format the label data into int and change the class labels to -1 and +1
    label_train = label_train.astype(int)
    label_test = label_test.astype(int)

    # initialize the label data for multivariate case
    train_targets = -numpy.ones([fea_train.shape[0],num_digits])
    test_targets = -numpy.ones([fea_test.shape[0],num_digits])

    for i in range(num_digits):
        train_targets[label_train == i,i] = 1
        test_targets[label_test == i,i] = 1


    # Initilize the trainer with 'LutTrainer' or 'StumpTrainer'
    boost_trainer = boosting.Boost('LutTrainer')

    # Set the parameters for the boosting
    boost_trainer.num_rnds = args.num_rnds     
    boost_trainer.loss_type = args.loss_type        
    boost_trainer.selection_type = args.selection_type
    boost_trainer.num_entries = args.num_entries


    # Perform boosting of the feature set samp
    print fea_train.shape
    print train_targets.shape 
    machine = boost_trainer.train(fea_train, train_targets)

    # Classify the test samples (testsamp) using the boosited classifier generated above
    prediction_labels = machine.classify(fea_test)

    # Calulate the values for confusion matrix
    confusion_matrix = numpy.zeros([num_digits,num_digits])
    for i in range(num_digits):
        prediction_i = prediction_labels[test_targets[:,i] == 1,:]
        num_samples_i = prediction_i.shape[0]
        for j in range(num_digits):
            confusion_matrix[j,i] = 100*(float(sum(prediction_i[:,j] == 1)/float(num_samples_i)))

    # Plot the confusion matrix
    cm_title = 'MultiLUT_pixel_round' + str(args.num_rnds)
    confusion.display_cm(confusion_matrix, cm_title)



if __name__ == "__main__":
   main()
