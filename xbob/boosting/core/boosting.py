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
							
	

	
	
    def train(self, fset, targets):
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

         Return:
         machine: The boosting machine that is combination of the weak classifiers.

        """

	# Initializations
        if(len(targets.shape) == 1):
            targets = targets[:,numpy.newaxis]

        num_op = targets.shape[1]
        machine = BoostMachine(num_op)
        num_samp = fset.shape[0]
        pred_scores = numpy.zeros([num_samp,num_op])
        loss_class = losses.LOSS_FUNCTIONS[self.loss_type]
        loss_func = loss_class()

        # For lut trainer the features should be integers 
        #if(self.weak_trainer_type == 'LutTrainer'):
        #    fset = fset.astype(int)
	

        # For each round of boosting initialize a new weak trainer
        if self.weak_trainer_type == 'LutTrainer':
            weak_trainer = trainers.LutTrainer(self.num_entries, self.lut_selection, num_op )
        elif self.weak_trainer_type == 'StumpTrainer':
            weak_trainer = trainers.StumpTrainer()
        #elif self.weak_trainer_type == 'GaussTrainer':
        #    weak_trainer = trainers.GaussianTrainer(3)


        # Start boosting iterations for num_rnds rounds
        for r in range(self.num_rnds):

           
            # Compute the gradient of the loss function, l'(y,f(x)) using loss_class
            loss_grad = loss_func.update_loss_grad(targets,pred_scores)

            # Select the best weak trainer for current round of boosting
            curr_weak_trainer = weak_trainer.compute_weak_trainer(fset, loss_grad)

            # Compute the classification scores of the samples based only on the current round weak classifier (g_r)
            curr_pred_scores = curr_weak_trainer.get_weak_scores(fset)
            
            # Initialize the start point for lbfgs minimization
            init_point = numpy.zeros(num_op)


            # Perform lbfgs minimization and compute the scale (alpha_r) for current weak trainer
            lbfgs_struct = scipy.optimize.fmin_l_bfgs_b(loss_func.loss_sum, init_point, fprime = loss_func.loss_grad_sum, args = (targets, pred_scores, curr_pred_scores)) 
            alpha = lbfgs_struct[0]


            # Update the prediction score after adding the score from the current weak classifier f(x) = f(x) + alpha_r*g_r
            pred_scores = pred_scores + alpha*curr_pred_scores 


            # Add the current trainer into the boosting machine
            machine.add_weak_trainer(curr_weak_trainer, alpha)
			
			
        return machine








class BoostMachine():
    """ The class to perform the classification using the set of weak trainer """

    
    def __init__(self, num_op):
        """ Initialize the set of weak trainers and the alpha values (scale)"""
        self.alpha = []
        self.weak_trainer = []
        self.num_op = num_op

    

    def add_weak_trainer(self, curr_trainer, curr_alpha):
        """ Function adds a weak trainer and the scale into the list

        Input: 
        curr_trainer: the weak trainer learner during a single round of boosting
        
        curr_alpha: the scale for the curr_trainer
        """
        self.alpha.append(curr_alpha)
        self.weak_trainer.append(curr_trainer)


    
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
        pred_labels = -numpy.ones([num_samp, self.num_op])
        pred_scores = numpy.zeros([num_samp, self.num_op]) 


        # For each round of boosting calculate the weak scores for that round and add to the total
        for i in range(num_trainer):
            curr_trainer = self.weak_trainer[i]
            weak_scores = curr_trainer.get_weak_scores(test_features)
            pred_scores = pred_scores + self.alpha[i] * weak_scores

        # predict the labels for test features based on score sign (for binary case) and score value (multivariate case)
        if(self.num_op == 1):
            pred_labels[pred_scores >=0] = 1
            pred_labels = numpy.squeeze(pred_labels)
        else:
            score_max = numpy.argmax(pred_scores, axis = 1)
            pred_labels[range(num_samp),score_max] = 1
        return pred_scores, pred_labels



