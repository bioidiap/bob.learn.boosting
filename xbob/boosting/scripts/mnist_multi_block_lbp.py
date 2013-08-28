#!/usr/bin/env python

"""The test script to perform the binary classification on the digits from the MNIST dataset.
   The MNIST data is exported using the xbob.db.mnist module which provide the train and test 
   partitions for the digits. Block based LBP type (LBP, tLBP, mLBP) features are captured  and the
   available algorithms for classification is Lut based Boosting.

"""


import xbob.db.mnist
import numpy 
import sys, getopt
import argparse
import string
from ..util import confusion
from ..features import local_feature
from ..core import boosting
import matplotlib.pyplot
 

def main():

    parser = argparse.ArgumentParser(description = " The arguments for the boosting. ")
    parser.add_argument('-r', default = 20, dest = "num_rnds", type = int, help = "The number of round for the boosting")
    parser.add_argument('-l', default = 'exp', dest = "loss_type", type= str, choices = {'log','exp'}, help = "The type of the loss function. Logit and Exponential functions are the avaliable options")
    parser.add_argument('-s', default = 'indep', dest = "selection_type", choices = {'indep', 'shared'}, type = str, help = "The feature selection type for the LUT based trainer. For multivarite case the features can be selected by sharing or independently ")
    parser.add_argument('-n', default = 256, dest = "num_entries", type = int, help = "The number of entries in the LookUp table. It is the range of the feature values, e.g. if LBP features are used this values is 256.")
    parser.add_argument('-f', default = 'lbp', dest = "feature_type", type = str, choices = {'lbp', 'mlbp', 'tlbp', 'dlbp'}, help = "The type of LBP features to be extracted from the image to perform the classification. The features are extracted from the block of varying scales")
    parser.add_argument('-sy', default = 4, dest = "scale_y", type = int, help = "The maximum scale for the block feature extraction along the y direction.")
    parser.add_argument('-sx', default = 4, dest = "scale_x", type = int, help = "The maximum scale for the block feature extraction along the x direction.")

    args = parser.parse_args()

    # download the dataset
    db_object = xbob.db.mnist.Database()
    # Hardcode the number of digits and the image size
    num_digits = 10
    img_size = 28


    # get the data (features and labels) for the selected digits from the xbob_db_mnist class functions
    img_train, label_train = db_object.data('train',labels = range(num_digits))
    img_test, label_test = db_object.data('test', labels = range(num_digits))


    # Format the label data into int and change the class labels to -1 and +1
    label_train = label_train.astype(int)
    label_test = label_test.astype(int)

    
    # initialize the label data for multivariate case
    train_targets = -numpy.ones([img_train.shape[0],num_digits])
    test_targets = -numpy.ones([img_test.shape[0],num_digits])

    for i in range(num_digits):
        train_targets[label_train == i,i] = 1
        test_targets[label_test == i,i] = 1
    

    # Extract the local features from the images
    feature_extractor = local_feature.lbp_feature(args.feature_type)
    scale_y = args.scale_y
    scale_x = args.scale_x
    
    num_fea = feature_extractor.get_feature_number(img_size,img_size,scale_y, scale_x)
    
    train_fea = numpy.zeros([img_train.shape[0], num_fea],dtype = 'uint8')
    test_fea = numpy.zeros([img_test.shape[0], num_fea], dtype = 'uint8')
    for img_num in range(img_train.shape[0]):
	img = img_train[img_num,:].reshape([img_size,img_size])
	train_fea[img_num,:] = feature_extractor.get_features(img, scale_y, scale_x)

    for img_num in range(img_test.shape[0]):
	img = img_test[img_num,:].reshape([img_size,img_size])
	test_fea[img_num,:] = feature_extractor.get_features(img, scale_y, scale_x)
    

    # Initilize the trainer with LutTrainer
    boost_trainer = boosting.Boost('LutTrainer')

    # Set the parameters for the boosting
    boost_trainer.num_rnds = args.num_rnds     
    boost_trainer.loss_type = args.loss_type        
    boost_trainer.selection_type = args.selection_type
    boost_trainer.num_entries = args.num_entries

    print "Start boosting the features"
    # Perform boosting of the feature set samp
    machine = boost_trainer.train(train_fea, train_targets)

    # Classify the test samples (testsamp) using the boosited classifier generated above
    prediction_labels = machine.classify(test_fea)

    # Calulate the values for confusion matrix
    confusion_matrix = numpy.zeros([num_digits,num_digits])
    for i in range(num_digits):
        prediction_i = prediction_labels[test_targets[:,i] == 1,:]
        num_samples_i = prediction_i.shape[0]
        for j in range(num_digits):
            confusion_matrix[j,i] = 100*(float(sum(prediction_i[:,j] == 1)/float(num_samples_i)))

    # Plot the confusion matrix
    cm_title = 'MultiLUT_Block_' + args.feature_type + str(scale_y) + '_round' + str(args.num_rnds)
    confusion.display_cm(confusion_matrix, cm_title)



if __name__ == "__main__":
   main()
