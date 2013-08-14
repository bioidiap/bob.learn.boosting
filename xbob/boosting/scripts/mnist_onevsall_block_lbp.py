#!/usr/bin/env python

"""The test script to perform the binary classification on the digits from the MNIST dataset.
The MNIST data is exported using the xbob.db.mnist module which provide the train and test 
partitions for the digits. Pixel values of grey scale images are used as features and the
available algorithms for classification are Lut based Boosting and Stump based Boosting.
Thus it conducts only one binary classifcation test. 


"""


import xbob.db.mnist
import bob
import numpy
import sys, getopt
import string
import argparse
from ..core import boosting
from ..features import local_feature
from ..util import confusion

def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', default = 'StumpTrainer',dest = "trainer_type", type = str, choices = {'StumpTrainer', 'LutTrainer'}, help = "This is the type of trainer used for the boosting." )
    parser.add_argument('-r', default = 20, dest = "num_rnds", type = int , help = "The number of round for the boosting")
    parser.add_argument('-l', default = 'exp', dest = "loss_type", type= str, choices = {'log','exp'}, help = "The type of the loss function. Logit and Exponential functions are the avaliable options")
    parser.add_argument('-s', default = 'indep', dest = "selection_type", choices = {'indep', 'shared'}, type = str, help = "The feature selection type for the LUT based trainer. For multivarite case the features can be selected by sharing or independently ")
    parser.add_argument('-n', default = 256, dest = "num_entries", type = int, help = "The number of entries in the LookUp table. It is the range of the feature values, e.g. if LBP features are used this values is 256.")
    parser.add_argument('-c', default = 'Confusion Matrix', dest = "title_str", type = str, help = "The title for the confusion matrix.")
   
    args = parser.parse_args()

    # download the dataset
    db_object = xbob.db.mnist.Database()
    train_img, train_label = db_object.data('train',labels = range(10))
    test_img, test_label = db_object.data('test', labels = range(10))
    #test_label = test_label[:,numpy.newaxis]

    # Extract the lbp features from the images
    feature_extractor = local_feature.lbp_feature('lbp')
    scale_y = 1
    scale_x = 1
    img_size = 28 
    num_fea = feature_extractor.get_feature_number(img_size,img_size,scale_y, scale_x)
    
    train_fea = numpy.zeros([train_img.shape[0], num_fea],dtype = 'uint8')
    test_fea = numpy.zeros([test_img.shape[0], num_fea], dtype = 'uint8')
    for img_num in range(train_img.shape[0]):
        img = train_img[img_num,:].reshape([img_size,img_size])
        train_fea[img_num,:] = feature_extractor.get_features(img, scale_y, scale_x)

    for img_num in range(test_img.shape[0]):
        img = test_img[img_num,:].reshape([img_size,img_size])
        test_fea[img_num,:] = feature_extractor.get_features(img, scale_y, scale_x)


    print "LBP features computed for the training and testing images."
    num_train_samples = 5000
    confusion_matrix = numpy.zeros([10,10])


    # Start the tests for digits classification
    for digit1 in range(10):

        # Copy the label data and change the class labels to -1 and +1 for binary classifier
        train_label_binary = numpy.copy(train_label)
        test_label_binary = numpy.copy(test_label)
        train_label_binary = train_label_binary.astype(int)
        test_label_binary = test_label_binary.astype(int)
        #print test_label_binary[0:20]

        train_label_binary[train_label == digit1] =  1
        test_label_binary[test_label == digit1] =  1
        train_label_binary[train_label != digit1] = -1
        test_label_binary[test_label != digit1] = -1
        #print test_label_binary[0:10]
        
        positive_fea = train_fea[train_label_binary ==  1,:]
        negative_fea = train_fea[train_label_binary == -1,:]

        train_feature = numpy.vstack((positive_fea[0:num_train_samples,:], negative_fea[0:num_train_samples,:]))
        train_classes = numpy.hstack((numpy.ones(num_train_samples), -numpy.ones(num_train_samples)))
        
        # Initilize the trainer with 'LutTrainer' or 'StumpTrainer'
        boost_trainer = boosting.Boost(args.trainer_type)

        # Set the parameters for the boosting
        boost_trainer.num_rnds = args.num_rnds             
        boost_trainer.loss_type = args.loss_type        
        boost_trainer.selection_type = args.selection_type
        boost_trainer.num_entries = args.num_entries


        # Perform boosting of the feature set samp 
        machine = boost_trainer.train(train_feature, train_classes)

        for test_digit in range(10):

            # Select the feature and label for a single test digit
            current_feature = test_fea[test_label == test_digit,:]
            current_label = test_label_binary[test_label == test_digit]

            # Classify the test samples (testsamp) using the boosted classifier generated above
            prediction_labels = machine.classify(current_feature)
            #print prediction_labels[0:10]

            # Calculate the accuracy in percentage for the curent classificaiton test
            current_label = current_label[:,numpy.newaxis]
            current_accuracy = 100*float(sum(prediction_labels == 1))/(len(current_label))
            confusion_matrix[digit1,test_digit] = 100*float(sum(prediction_labels == 1))/(len(current_label))
            #print "The accuracy of binary classification test with digits %d and %d is %f " % (digit1, test_digit, accuracy)


    print confusion_matrix
    confusion.display_cm(confusion_matrix, args.title_str)

if __name__ == "__main__":
   main()
