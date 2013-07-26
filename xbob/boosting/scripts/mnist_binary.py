"""The test script to perform the binary classification on the digits from the MNIST dataset.
   The MNIST data is exported using the xbob.db.mnist module which provide the train and test 
   partitions for the digits. Pixel values of grey scale images are used as features and the
   available algorithms for classification are Lut based Boosting and Stump based Boosting.
   The script test all the possible combination of the two digits which results in 45 different 
   binary classfication tests. 

$ python mnist_binary.py -t <Trainer_type> -r <Number_of_boosting_rounds> -l <Loss_type> -s <selection_type> -n <Number_of_lut_entries>

"""


import xbob.db.mnist
import numpy
import sys, getopt
from ..core import booster

def main(argv):

    opts, args = getopt.getopt(argv,"t:r:l:s:n:")
    for opt, arg in opts:
        if opt == '-t':
            trainer_type = arg
        elif opt == '-r':
            num_rnds = arg
        elif opt == 'l':
            loss_type = arg
        elif opt == 's':
            selection_type = arg
        elif opt == 'n':
            num_entries = arg


    # Initializations
    accu = 0
    test_num = 0

    # download the dataset
    db_object = xbob.db.mnist.Database()

    # select the digits to classify
    for digit1 in range(10):
        for digit2 in range(digit1+2,10):
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
            boost_trainer = booster.Boost('StumpTrainer')

            # Set the parameters for the boosting
            boost_trainer.num_rnds = 10             
            boost_trainer.loss_type = 'exp'        
            boost_trainer.selection_type = 'indep'  
            boost_trainer.num_entries = 256       


            # Perform boosting of the feature set samp 
            model = boost_trainer.train(fea_train, label_train)

            # Classify the test samples (testsamp) using the boosited classifier generated above
            prediction_labels = model.classify(fea_test)

            # calculate the accuracy in percentage for the curent classificaiton test
            label_test = label_test[:,numpy.newaxis]
            accuracy = 100*float(sum(prediction_labels == label_test))/(len(label_test))
            print "The accuracy of binary classification test for digits %d and %d is %f " % (digit1, digit2, accuracy)
            accu = accu + accuracy


    accu = accu/test_num

    print "The average accuracy for all the test is %f %" % (accu)

if __name__ == "__main__":
   main(sys.argv[1:])
