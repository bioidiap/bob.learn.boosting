import numpy as np
import weaklearner
import StumpTrainer

class boost_machine():

    def __init__(self):
        self.alpha = []
        self.weak_trainer = []

    def add_weak_trainer(self, curr_trainer, curr_alpha):
        self.alpha.append(curr_alpha)
        self.weak_trainer.append(curr_trainer)

    def classify(self, test_features):
        num_trainer = len(self.weak_trainer)
        num_samp = test_features.shape[0]
        pred_scores = np.zeros([num_samp, 1]) 
        for i in range(num_trainer):
            print self.alpha[i]
            #weak_feature = test_features[:,self.weak_trainer[i].selected_indices]
            curr_trainer = self.weak_trainer[i]
            #print np.transpose(curr_trainer.luts)
            #print curr_trainer.selected_indices
            # print type(curr_trainer)
            weak_scores = curr_trainer.get_weak_scores(test_features)
            pred_scores = pred_scores + self.alpha[i] * weak_scores
        pred_labels = np.ones([num_samp,1])
        pred_labels[pred_scores <=0] = -1
        return pred_labels


'''	# Function to classify the features based on the model
	def test(self, test_features,model):
		num_classifier = len(model)
		num_samples = len(test_features)
		test_features = test_features.astype(int)
		num_op = model[0]['lut'].shape[1]
		pred_scores = np.zeros([num_samples,num_op])
		print pred_scores
		for ri in range(num_classifier):
			curr_model = model[ri]
			curr_findex = curr_model['findex']
			curr_lut = curr_model['lut']
			print curr_lut.shape
			for oi in range(curr_lut.shape[1]):
				curr_fea = test_features[:,curr_findex[oi]]
				curr_score =  curr_lut[curr_fea,oi]
				curr_score = curr_score[:,np.newaxis]
		pred_scores = pred_scores + curr_score
		pred_labels = np.ones([num_samples,1])
		print pred_scores.shape
		pred_labels[pred_scores <= 0] = -1
		return pred_labels
'''

