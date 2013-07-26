#!/usr/bin/env python

"""The test script to perform the binary classification on the digits from the MNIST dataset.
    Pixel values of grey scale images are used as features and the available algorithms
    for classification are Lut based Boosting and Stump based Boosting.
   The script test all the possible combination of the two digits which results in 45 different 
   binary classfication tests.


"""


import xbob.db.mnist
import numpy
import sys, getopt
import argparse
import string
from ..core import booster
import xbob.db.mnist

def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', default = 'StumpTrainer',dest = "trainer_type", type = str, choices = {'StumpTrainer', 'LutTrainer'}, help = "This is the type of trainer used for the boosting." )
    parser.add_argument('-r', default = 20, dest = "num_rnds", type = int , help = "The number of round for the boosting")
    parser.add_argument('-l', default = 'exp', dest = "loss_type", type= str, choices = {'log','exp'}, help = "The type of the loss function. Logit and Exponential functions are the avaliable options")
    parser.add_argument('-s', default = 'indep', dest = "selection_type", choices = {'indep', 'shared'}, type = str, help = "The feature selection type for the LUT based trainer. For multivarite case the features can be selected by sharing or independently ")
    parser.add_argument('-n', default = 256, dest = "num_entries", type = int, help = "The number of entries in the LookUp table. It is the range of the feature values, e.g. if LBP features are used this values is 256.")

    args = parser.parse_args()

    # Initializations
    accu = 0
    test_num = 0

    # download the dataset
    db_object = xbob.db.mnist.Database()

    # select the digits to classify
    for digit1 in range(10):
        for digit2 in range(digit1+1,10):
            test_num = test_num +1

            # get the data (features and labels) for the selected digits from the xbob_db_mnist class functions
            fea_train, label_train = db_object.data('train',labels = [digit1,digit2])
            fea_test, label_test = db_object.data('test', labels = [digit1,digit2])


            # Format the label data into int and change the class labels to -1 and +1
            label_train = label_train.astype(int)
            label_test = label_test.astype(int)

            label_train[label_train == digit1] =  1
            label_test[label_test == digit1] =  1
            label_train[label_train == digit2] = -1
            label_test[label_test == digit2] = -1


            # Initilize the trainer with 'LutTrainer' or 'StumpTrainer'
            boost_trainer = booster.Boost(args.trainer_type)

            # Set the parameters for the boosting
            boost_trainer.num_rnds = args.num_rnds             
            boost_trainer.loss_type = args.loss_type        
            boost_trainer.selection_type = args.selection_type
            boost_trainer.num_entries = args.num_entries


            # Perform boosting of the feature set samp 
            machine = boost_trainer.train(fea_train, label_train)

            # Classify the test samples (testsamp) using the boosited classifier generated above
            prediction_labels = machine.classify(fea_test)

            # calculate the accuracy in percentage for the curent classificaiton test
            label_test = label_test[:,numpy.newaxis]
            accuracy = 100*float(sum(prediction_labels == label_test))/(len(label_test))
            print "The accuracy of binary classification test for digits %d and %d is %f " % (digit1, digit2, accuracy)
            accu = accu + accuracy


    accu = accu/test_num

    print "The average accuracy for all the test is %f " % (accu)
    return 0


if __name__ == "__main__":
   main()
