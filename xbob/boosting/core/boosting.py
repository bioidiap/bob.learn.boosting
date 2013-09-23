""" The module consist of the classes to generate a strong boosting classifier and test features using that classifier.
    Boosting algorithms have three main dimensions: weak trainers that are boosting, optimization strategy
    for boosting and loss function that guide the optimization. For each one of these the following
    choices are implemented.

Weak Trainers: StumpTrainer- classifies the features based on a specified threshold
               LutTrainer- Look-Up-Table are used for classification

Optimization Strategy: For StumpTrainer the gradient descent (GradBoost) is used and for LutTrainer the
                optimization is based on Taylor's Boosting framework.
                See following references:
                Saberian et. al.  "Taylorboost: First and second-order boosting algorithms with explicit
                                    margin control."
                Cosmin Atanasoea "Multivariate Boosting with Look-Up Tables for face processing" Phd Thesis

Loss Function: Exponential Loss (Preferred with the StumpTrainer)
               Log Loss (Preferred with LutTrainer)
               Tangent Loss


"""




import numpy
import trainers
import losses
import scipy.optimize
import itertools

import logging
logger = logging.getLogger('bob')

from .. import BoostedMachine

class Boost:

    """ The class to boost the features from  a set of training samples.

    It iteratively adds new trainer models to assemble a strong classifier.
    In each round of iteration a weak trainer is learned
    by optimization of a differentiable function. The following parameters are involved


    Parameters:

    trainer_type:  Type string, Default = 'stump'
                   The type of weak trainer to be learned. Two types of weak trainers are
                   supported currently.

                   'LutTrainer':  It is used for discrete feature types.LUT are used as weak
                                  trainers and Taylor Boost is used as optimization strategy.
                                  Ex.: LBP features, MCT features.

                   'StumpTrainer': Decision Stumps are used as weak trainer and GradBoost is
                                used as optimization strategy.It can be used with both discrete
                                and continuous type of features

    num_rnds:      Type int, Default = 100
                   The number of rounds for boosting. The boosting strategies implemented here
                   (GradBoost and TaylorBoost) are fairly robust to overfitting, so the large
                   number of rounds generally results in a small error rate.

    loss_type:    Type string, Default = 'log'
                  It is the type of loss function to be optimized. Currently we support the
                  following classes of loss function:
                  'log' and 'exp'
                  'exp' loss function is preferred with StumpTrainer and 'log' with LutTrainer.



     num_entries:  Type int, Default = 256
                   This is the parameter for the LutTrainer. It is the
                   number of entries in the LookUp table. It can be determined from the range of
                   feature values. For examples, for LBP features the number of entries in the
                   LookUp table is 256.




     lut_selection: Type string, Default = 'indep'
                  For multivariate classification during the weak trainer selection the best feature can
                  either be shared with all the outputs or it can be selected independently for each output.
                  For feature sharing set the parameter to 'shared' and for independent selection set it to
                  'indep'. See cosmin's thesis for a detailed explanation on the feature selection type.
                  For univariate cases such as face detection this parameter is not relevant.

    Example Usage:

    # Initialize the boosting parameter
    num_rounds = 50
    feature_range = 256
    loss_type = 'log'
    selection_type = 'indep'
    boost_trainer = boosting.Boost('LutTrainer', num_rounds, feature_range, loss_type, selection_type )

    # Train machine using training samples
    machine = boost_trainer.train(train_fea, train_targets)

    # Classify the samples using boosted classifier
    prediction_labels = machine.classify(test_fea)


    """




    def __init__(self, trainer_type, num_rnds = 20, num_entries = 256, loss_type = 'log', lut_selection = 'indep'):
        """ The function to initialize the boosting parameters.

        The function set the default values for the following boosting parameters:
        The number of rounds for boosting: 20
        The number of entries in LUT: 256 (For LBP type features)
        The loss function type: logit
        The LUT selection type: independent

        Inputs:
        trainer_type: The type of trainer for boosting.
                      Type: string
                      Values: LutTrainer or StumpTrainer
        num_rnds:     The number of rounds of boosting
                      Type: int
                      Values: 20 (Default)
        num_entries:  The number of entries for the lookup table
                      Type: int
                      Values: 256 (Default)
        loss_type:    The loss function to be be minimized
                      Type: string
                      Values: 'log' or 'exp'
        lut_selection: The selection type for the LUT based trainers
                       Type: string
                       Values: 'indep' or 'shared'

        """
        self.num_rnds = num_rnds
        self.num_entries = num_entries
        self.loss_type = loss_type
        self.lut_selection = lut_selection
        self.weak_trainer_type = trainer_type





    def train(self, fset, targets, boosted_machine = None):
        """ The function to train a boosting machine.

         The function boosts the discrete features (fset) and returns a strong classifier
         as a combination of weak classifier.

         Inputs:
         fset: features extracted from the samples
               features should be discrete for lutTrainer.
               Type: numpy array (num_sam x num_features)

         labels: class labels of the samples
               Type: numpy array

               Shape for binary classification: #number of samples
               Shape for multivariate classification: #number of samples x #number of outputs

               Examples for 4 classes case (0,1,2,3) and three test samples.
                         [[-1,  1, -1, -1],    #Predicted class is 1
                          [ 1, -1, -1, -1],    #Predicted class is 0
                          [-1, -1, -1,  1]]    #Predicted class is 3
               There can be only single 1 in a row and the index of 1 indicates the class.

         boosted_machine: the machine to add the weak machines to. If not given, a new machine is created.
               Type: BoostMachine (with valid output dimension)

         Return:
         machine: The boosted machine that is combination of the weak classifiers.

        """

        # Initializations
        if(len(targets.shape) == 1):
            targets = targets[:,numpy.newaxis]

        num_op = targets.shape[1]
        machine = BoostedMachine() if boosted_machine is None else boosted_machine
        num_samp = fset.shape[0]
        pred_scores = numpy.zeros([num_samp,num_op])
        loss_class = losses.LOSS_FUNCTIONS[self.loss_type]
        loss_func = loss_class()

        # For lut trainer the features should be integers
        #if(self.weak_trainer_type == 'LutTrainer'):
        #    fset = fset.astype(int)


        # For each round of boosting initialize a new weak trainer
        weak_trainer = {
            'LutTrainer'   : trainers.LutTrainer(self.num_entries, self.lut_selection, num_op),
            'StumpTrainer' : trainers.StumpTrainer(),
#           'GaussTrainer' : trainers.GaussianTrainer(3)
        } [self.weak_trainer_type]


        # Start boosting iterations for num_rnds rounds
        logger.info("Starting %d rounds of boosting" % self.num_rnds)
        for r in range(self.num_rnds):


            # Compute the gradient of the loss function, l'(y,f(x)) using loss_class
            loss_grad = loss_func.update_loss_grad(targets,pred_scores)

            # Select the best weak machine for current round of boosting
            curr_weak_machine = weak_trainer.compute_weak_trainer(fset, loss_grad)

            # Compute the classification scores of the samples based only on the current round weak classifier (g_r)
            curr_pred_scores = numpy.zeros([num_samp,num_op], numpy.float64)
            curr_weak_machine(fset, curr_pred_scores)

            # Initialize the start point for lbfgs minimization
            init_point = numpy.zeros(num_op)


            # Perform lbfgs minimization and compute the scale (alpha_r) for current weak trainer
            lbfgs_struct = scipy.optimize.fmin_l_bfgs_b(loss_func.loss_sum, init_point, fprime = loss_func.loss_grad_sum, args = (targets, pred_scores, curr_pred_scores))
            alpha = lbfgs_struct[0]


            # Update the prediction score after adding the score from the current weak classifier f(x) = f(x) + alpha_r*g_r
            pred_scores = pred_scores + alpha*curr_pred_scores


            # Add the current trainer into the boosting machine
            machine.add_weak_machine(curr_weak_machine, alpha)

            logger.debug("Finished round %d / %r" % (r+1, self.num_rnds))

        return machine








class BoostMachine():
    """ The class to perform the classification using the set of weak trainer """


    def __init__(self, number_of_outputs = 1, hdf5file = None):
        """ Initialize the set of weak trainers and the alpha values (scale)"""
        if hdf5file is not None:
          self.load(hdf5file)
        else:
          self.alpha = []
          self.weak_trainer = []
          self.number_of_outputs = number_of_outputs
          self.selected_indices = set()
          self._update()


    def _update(self):
      """ Initializes internal variables."""
      self.selected_indices = set([weak_trainer.selected_indices[i] for weak_trainer in self.weak_trainer for i in range(self.number_of_outputs)])
      self._weak_results = numpy.ndarray((len(self.weak_trainer),), numpy.float64)


    def add_weak_trainer(self, curr_trainer, curr_alpha):
        """ Function adds a weak trainer and the scale into the list

        Input:
        curr_trainer: the weak trainer learner during a single round of boosting

        curr_alpha: the scale for the curr_trainer
        """
        self.alpha.append(curr_alpha)
        self.weak_trainer.append(curr_trainer)
        self._update()


    def feature_indices(self):
      """Returns the indices of the features that are selected by the weak classifiers."""
      return sorted(list(self.selected_indices))


    def __call__(self, feature):
      """Returns the predicted score for the given single feature, assuming only single output.

      Input: A single feature vector of length No. of total features

      Output: A single floating point number
      """
      # iterate over the weak classifiers
      for index in xrange(len(self.weak_trainer)):
        self._weak_results[index] = self.alpha[index] * self.weak_trainer[index].get_weak_score(feature)
      return numpy.sum(self._weak_results)


    def classify(self, test_features):
        """ Function to classify the test features using a strong trained classifier.

        The function classifies the test features using the boosting machine trained with a
        combination of weak classifiers.

        Inputs:
        test_features: The test features to be classified using the trained machine
                       Type: numpy array (#number of test samples x #number of features)


        Return:
        prediction_scores: The real valued number which are thresholded to determine the prediction classes.

        prediction_labels: The predicted classes for the test samples. It is a binary numpy array where
                         1 indicates the predicted class.
                         Type: numpy array
                         Shape for binary classification: #number of samples


                         Shape for multivariate classification: #number of samples x #number of outputs

                         Examples for 4 classes case (0,1,2,3) and three test samples.
                         [[-1,  1, -1, -1],    #Predicted class is 1
                          [ 1, -1, -1, -1],    #Predicted class is 0
                          [-1, -1, -1,  1]]    #Predicted class is 3
               There can be only single 1 in a row and the index of 1 indicates the class.

        """
        # Initialization
        num_trainer = len(self.weak_trainer)
        num_samp = test_features.shape[0]
        pred_labels = -numpy.ones([num_samp, self.number_of_outputs])
        pred_scores = numpy.zeros([num_samp, self.number_of_outputs])


        # For each round of boosting calculate the weak scores for that round and add to the total
        for i in range(num_trainer):
            curr_trainer = self.weak_trainer[i]
            weak_scores = curr_trainer.get_weak_scores(test_features)
            pred_scores = pred_scores + self.alpha[i] * weak_scores

        # predict the labels for test features based on score sign (for binary case) and score value (multivariate case)
        if(self.number_of_outputs == 1):
            pred_labels[pred_scores >=0] = 1
            pred_labels = numpy.squeeze(pred_labels)
        else:
            score_max = numpy.argmax(pred_scores, axis = 1)
            pred_labels[range(num_samp),score_max] = 1
        return pred_scores, pred_labels


    def save(self, hdf5File):
      hdf5File.set_attribute("version", 0)
      hdf5File.set("Weights", self.alpha)
      hdf5File.set("Outputs", self.number_of_outputs)
      for i in range(len(self.weak_trainer)):
        dir_name = "WeakMachine_%d"%i
        hdf5File.create_group(dir_name)
        hdf5File.cd(dir_name)
        hdf5File.set_attribute("MachineType", self.weak_trainer[i].__class__.__name__)
        self.weak_trainer[i].save(hdf5File)
        hdf5File.cd('..')


    def load(self, hdf5File):
      self.alpha = hdf5File.read("Weights")
      self.number_of_outputs = hdf5File.read("Outputs")
      self.weak_trainer = []
      self.selected_indices = set()
      for i in range(len(self.alpha)):
        dir_name = "WeakMachine_%d"%i
        hdf5File.cd(dir_name)
        weak_machine_type = hdf5File.get_attribute("MachineType")
        weak_machine = {
          "LutMachine"   : trainers.LutMachine(),
          "StumpMachine" : trainers.StumpMachine()
        } [weak_machine_type]
        weak_machine.load(hdf5File)
        self.weak_trainer.append(weak_machine)
        self.selected_indices |= set([weak_machine.selected_indices[i] for i in range(self.number_of_outputs)])
        hdf5File.cd('..')
      self._update()


