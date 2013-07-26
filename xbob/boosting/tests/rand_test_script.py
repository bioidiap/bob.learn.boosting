import sys
import numpy as np
import random
import tboost
import math


num_fea = 500
num_samples = 10000

# generate the random training and testing samples 
samp1 = np.floor(80*np.random.rand(num_samples,num_fea) + 10*np.ones([num_samples,num_fea]))
samp1[samp1>119] = 119
testsamp1 = np.floor(80*np.random.rand(num_samples,num_fea) + 10*np.ones([num_samples,num_fea]))
testsamp1[testsamp1 >119] = 119
class1 = np.ones([num_samples,1])
testclass1 = np.ones([num_samples,1])

samp2 = np.floor(60*np.random.rand(num_samples,num_fea) + 50 *np.ones([num_samples,num_fea])) 
samp2[samp2 > 119] = 119
testsamp2 = np.floor(60*np.random.rand(num_samples,num_fea)+ 50*np.ones([num_samples,num_fea]))
testsamp2[testsamp2 >119] = 119
class2 = -1*np.ones([num_samples,1])
testclass2 = -1*np.ones([num_samples,1])

testsamp = np.vstack([testsamp1,testsamp2])
testcla = np.vstack([testclass1,testclass2])
samp = np.vstack([samp1,samp2])
cla = np.vstack([class1,class2])
print np.transpose(samp[:,0])

# Set the parameters for the boosting

rnds = 2               # The number of rounds in boosting
bl_type = 'log'        # Type of baseloss functions l(y,f(x)), its can take one of these values ('exp', 'log', 'symexp', 'symlog')
s_type = 'indep'       # It can be 'indep' or 'shared' for details check cosim thesis
num_op = 1             # The number of outputs, for face detection num_op is 1
n_entries = 120        # The number of entries in the LUT, it is the range of the discrete features
loss_type = 'var'      # It can be 'exp' for Expectational loss or 'var' for Variational loss
lamda = 0.4            # lamda value for variational loss

param = {'nrnds':rnds,'baseloss_type':bl_type, 'selection_type':s_type, 'num_op':num_op, 'num_entries':n_entries, 'loss_type':loss_type, 'lamda': lamda }



# Perform boosting of the feature set samp 
model = tboost.LUTBoost_train(samp, cla, param)

# Classify the test samples (testsamp) using the boosited classifier generated above
prediction_labels = tboost.LUTBoost_classify(testsamp,model)

accuracy = float(sum(prediction_labels == testcla))/float(2*num_samples)
print accuracy
