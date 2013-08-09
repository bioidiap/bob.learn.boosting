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
from ..features import local_feature
from ..core import boosting
import matplotlib.pyplot as mpl
 

def main():

    parser = argparse.ArgumentParser(description = " The arguments for the boosting. ")
    parser.add_argument('-r', default = 20, dest = "num_rnds", type = int, help = "The number of round for the boosting")
    parser.add_argument('-l', default = 'exp', dest = "loss_type", type= str, choices = {'log','exp'}, help = "The type of the loss function. Logit and Exponential functions are the avaliable options")
    parser.add_argument('-s', default = 'indep', dest = "selection_type", choices = {'indep', 'shared'}, type = str, help = "The feature selection type for the LUT based trainer. For multivarite case the features can be selected by sharing or independently ")
    parser.add_argument('-n', default = 256, dest = "num_entries", type = int, help = "The number of entries in the LookUp table. It is the range of the feature values, e.g. if LBP features are used this values is 256.")

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
    feature_extractor = local_feature.lbp_feature('lbp')
    scale_y = 4
    scale_x = 4
    
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


    # Perform boosting of the feature set samp
    machine = boost_trainer.train(train_fea, train_targets)

    # Classify the test samples (testsamp) using the boosited classifier generated above
    prediction_labels = machine.classify(test_fea)


    # Calulate the values for confusion matrix
    score = np.zeros([10,10])
    for i in range(num_digits):
        prediction_i = prediction_labels[test_targets[:,i] == 1,:]
        print prediction_i.shape
        for j in range(num_digits):
            score[i,j] = sum(prediction_i[:,j] == 1)
    np.savetxt('conf_mat.out', score, delimiter=',')
    cm = score/np.sum(score,1)
    res = mpl.imshow(cm, cmap=mpl.cm.summer, interpolation='nearest')

    for x in numpy.arange(cm.shape[0]):
      for y in numpy.arange(cm.shape[1]):
          col = 'white'
          if cm[x,y] > 0.5: col = 'black'
          mpl.annotate('%.2f' % (100*cm[x,y],), xy=(y,x), color=col,
              fontsize=8, horizontalalignment='center', verticalalignment='center')

    classes = [str(k) for k in range(10)]

    mpl.xticks(numpy.arange(10), classes)
    mpl.yticks(numpy.arange(10), classes, rotation=90)
    mpl.ylabel("(Your prediction)")
    mpl.xlabel("(Real class)")
    mpl.title("Confusion Matrix (%s set) - in %%" % set_name)
    mpl.show()


    # Calculate the accuracy in percentage for the curent classificaiton test
    accuracy = 100*float(sum(numpy.sum(prediction_labels == test_targets,1) == num_digits))/float(prediction_labels.shape[0])

    print "The average accuracy of classification is %f " % (accuracy)




if __name__ == "__main__":
   main()
