import xbob.db.mnist
import numpy as np
import booster

num_train_samples = 10000
accu = 0
num = 0



# download the training and testing dataset for digits 1 and 2
db_object = xbob.db.mnist.Database()

digit1 = 0
digit2 = 1
digit3 = 2

# select the digits to classify
num = num +1
num_op = 3
# get the data (features and labels) for the selected digits
fea_tr, label_tr = db_object.data('train',labels = [digit1,digit2,digit3])
fea_ts, label_ts = db_object.data('test', labels = [digit1,digit2,digit3])


# Format the label data into int and change the class labels to -1 and +1
label_tr = label_tr.astype(int)
label_ts = label_ts.astype(int)

train_targets = -np.ones([fea_tr.shape[0],3])
test_targets = -np.ones([fea_ts.shape[0],3])

print label_tr[0:10]

for i in range(3):
    train_targets[label_tr == i,i] = 1
    test_targets[label_ts == i,i] = 1

label_tr[label_tr == digit1] =  1
label_ts[label_ts == digit1] =  1
label_tr[label_tr == digit2] = -1
label_ts[label_ts == digit2] = -1



#selected_trc = selected_trc[:,np.newaxis]
#label_ts = label_ts[:,np.newaxis]


boost_trainer = booster.Boost('LutTrainer')

# Set the parameters for the boosting
boost_trainer.num_rnds = 1              # The number of rounds in boosting
boost_trainer.bl_type = 'exp'        # Type of baseloss functions l(y,f(x)), its can take one of these values ('exp', 'log', 'symexp', 'symlog')
boost_trainer.s_type = 'indep'       # It can be 'indep' or 'shared' for details check cosim thesis
boost_trainer.num_entries = 256        # The number of entries in the LUT, it is the range of the discrete features
boost_trainer.loss_type = 'exp'      # It can be 'exp' for Expectational loss or 'var' for Variational loss
boost_trainer.lamda = 0.4            # lamda value for variational loss
#boost.weak_trainer_type = 'LutTrainer'


# Perform boosting of the feature set samp 
model = boost_trainer.train(fea_tr, train_targets)

# Classify the test samples (testsamp) using the boosited classifier generated above
pred_scores, prediction_labels = model.classify(fea_ts)
print prediction_labels[1:10,:]
print test_targets[1:10,:]
accuracy = float(sum(np.sum(prediction_labels == test_targets,1) == num_op))/float(prediction_labels.shape[0])
print accuracy
accu = accu + accuracy
print accu
print num
print accu/num
