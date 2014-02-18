#!/usr/bin/env python

"""The test script to perform the binary classification on the digits from the MNIST dataset.
The MNIST data is exported using the xbob.db.mnist module which provide the train and test
partitions for the digits. Pixel values of grey scale images are used as features and the
available algorithms for classification are Lut based Boosting and Stump based Boosting.
Thus it conducts only one binary classifcation test.


"""


import xbob.db.mnist
import numpy
import argparse
import xbob.db.mnist

import xbob.boosting


TRAINER = {
  'stump' : xbob.boosting.trainer.StumpTrainer,
  'lut'   : xbob.boosting.trainer.LUTTrainer,
}

LOSS = {
  'exp'   : xbob.boosting.loss.ExponentialLoss,
  'log'   : xbob.boosting.loss.LogitLoss,
  'tan'   : xbob.boosting.loss.TangentialLoss,
}

def command_line_arguments():
  """Defines the command line options."""
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-t', '--trainer-type', default = 'stump', choices = TRAINER.keys(), help = "The type of weak trainer used for boosting." )
  parser.add_argument('-l', '--loss-type', default = 'exp', choices = LOSS.keys(), help = "The type of loss function used in boosting to compute the weights for the weak classifiers.")
  parser.add_argument('-r', '--number-of-boosting-rounds', type = int, default = 20, help = "The number of boosting rounds, i.e., the number of weak classifiers.")

  parser.add_argument('-m', '--multi-variate', action = 'store_true', help = "Perform multi-variate training?")
  parser.add_argument('-s', '--feature-selection-style', default = 'independent', choices = {'indepenent', 'shared'}, help = "The feature selection style (only for multivariate classification with the LUT trainer).")

  parser.add_argument('-d', '--digits', type = int, nargs="+", choices=range(10), default=[5,6], help = "Select the digits you want to compare.")
  parser.add_argument('-n', '--number-of-elements', type = int, help = "For testing purposes: limit the number of training and test examples for each class.")
  parser.add_argument('-c', '--classifier-file', help = "If selected, the strong classifier will be stored in this file (or loaded from it if it already exists).")

  return parser.parse_args()


def main():

  args = command_line_arguments()

  # open connection to the MNIST database
  db = xbob.db.mnist.Database()

  # perform training, if desired
  if args.classifier_file is None or not os.path.exists(args.classifier_file):
    # get the training data
    training_features, training_labels = db_object.data('train', labels = args.digits)

    print training_labels




  fea_test, label_test = db_object.data('test', labels = args.digits)


  # Format the label data into int and change the class labels to -1 and +1
  label_train = label_train.astype(int)
  label_test = label_test.astype(int)

  label_train[label_train == digit1] =  1
  label_test[label_test == digit1] =  1
  label_train[label_train == digit2] = -1
  label_test[label_test == digit2] = -1

  print label_train.shape
  print label_test.shape


  # Initialize the trainer with 'LutTrainer' or 'StumpTrainer'
  boost_trainer = boosting.Boost(args.trainer_type)

  # Set the parameters for the boosting
  boost_trainer.num_rnds = args.num_rnds
  boost_trainer.loss_type = args.loss_type
  boost_trainer.selection_type = args.selection_type
  boost_trainer.num_entries = args.num_entries


  # Perform boosting of the feature set samp
  machine = boost_trainer.train(fea_train, label_train)

  # Classify the test samples (testsamp) using the boosited classifier generated above
  pred_scores, prediction_labels = machine.classify(fea_test)

  # calculate the accuracy in percentage for the curent classificaiton test
  #label_test = label_test[:,numpy.newaxis]
  accuracy = 100*float(sum(prediction_labels == label_test))/(len(label_test))
  print "The accuracy of binary classification test with digits %d and %d is %f " % (digit1, digit2, accuracy)




if __name__ == "__main__":
   main()
