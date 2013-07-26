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
from ..core import booster
import xbob.db.mnist 

def main():

    parser = argparse.ArgumentParser(description = " The arguments for the boosting. ")
    parser.add_argument('-t', default = 'StumpTrainer',dest = "trainer_type", type = string, choices = {'StumpTrainer', 'LutTrainer'}, help = "This is the type of trainer used for the boosting." )
    parser.add_argument('-r', default = 20, dest = "num_rnds", type = string , help = "The number of round for the boosting")
    parser.add_argument('-l', default = 'exp', dest = "loss_type", type= string,choices = {'log','exp'} help = "The type of the loss function. Logit and Exponential functions are the avaliable options")
    parser.add_argument('-s', default = 'indep', dest = "selection_type", choices = {'indep', 'shared'}, type = string, help = "The feature selection type for the LUT based trainer. For multivarite case the features can be selected by sharing or independently ")
    parser.add_argument('-n', default = 256, dest = "num_entries", type = int, help = "The number of entries in the LookUp table. It is the range of the feature values, e.g. if LBP features are used this values is 256.")

    args = parser.parse_args()

    # download the dataset
    db_object = xbob.db.mnist.Database()


    # get the data (features and labels) for the selected digits from the xbob_db_mnist class functions
    fea_train, label_train = db_object.data('train',labels = range(10))
    fea_test, label_test = db_object.data('test', labels = range(10))


    # Format the label data into int and change the class labels to -1 and +1
    label_train = label_train.astype(int)
    label_test = label_test.astype(int)

    # initialize the label data for multivariate case
    train_targets = -np.ones([fea_tr.shape[0],10])
    test_targets = -np.ones([fea_ts.shape[0],10])

    for i in range(10):
        train_targets[label_tr == i,i] = 1
        test_targets[label_ts == i,i] = 1


    # Initilize the trainer with 'LutTrainer' or 'StumpTrainer'
    boost_trainer = booster.Boost(args.trainer_type)

    # Set the parameters for the boosting
    boost_trainer.num_rnds = args.num_rnds     
    boost_trainer.loss_type = args.loss_type        
    boost_trainer.selection_type = args.selection_type
    boost_trainer.num_entries = args.num_entries


    # Perform boosting of the feature set samp 
    machine = boost_trainer.train(fea_train, train_targets)

    # Classify the test samples (testsamp) using the boosited classifier generated above
    prediction_labels = machine.classify(fea_test)

    # Calculate the accuracy in percentage for the curent classificaiton test
    accuracy = 100*float(sum(np.sum(prediction_labels == test_targets,1) == num_op))/float(prediction_labels.shape[0])

    print "The accuracy of binary classification test for digits %d and %d is %f " % (digit1, digit2, accuracy)




if __name__ == "__main__":
   main()
