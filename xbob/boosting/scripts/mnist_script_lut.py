import xbob.db.mnist
import numpy
import DummyBoost
import boostMachine

num_train_samples = 10000
accu = 0
num = 0

# download the training and testing dataset for digits 1 and 2
db_object = xbob.db.mnist.Database()

# select the digits to classify
for digit1 in range(10):
    for digit2 in range(digit1+1,10):
        num = num +1
        # get the data (features and labels) for the selected digits
        fea_tr, label_tr = db_object.data('train',labels = [digit1,digit2])
        fea_ts, label_ts = db_object.data('test', labels = [digit1,digit2])


        # Format the label data into int and change the class labels to -1 and +1
        label_tr = label_tr.astype(int)
        label_ts = label_ts.astype(int)

        label_tr[label_tr == digit1] =  1
        label_ts[label_ts == digit1] =  1
        label_tr[label_tr == digit2] = -1
        label_ts[label_ts == digit2] = -1


        selected_trs = fea_tr
        selected_trc = label_tr

        print selected_trs.shape
        print selected_trc.shape

        selected_trc = selected_trc[:,numpy.newaxis]
        label_ts = label_ts[:,numpy.newaxis]


        boost_trainer = DummyBoost.Boost('LutTrainer')

        # Set the parameters for the boosting
        boost_trainer.num_rnds = 20               # The number of rounds in boosting
        boost_trainer.bl_type = 'exp'        # Type of baseloss functions l(y,f(x)), its can take one of these values ('exp', 'log', 'symexp', 'symlog')
        boost_trainer.s_type = 'indep'       # It can be 'indep' or 'shared' for details check cosim thesis
        boost_trainer.num_entries = 256        # The number of entries in the LUT, it is the range of the discrete features
        boost_trainer.loss_type = 'exp'      # It can be 'exp' for Expectational loss or 'var' for Variational loss
        boost_trainer.lamda = 0.4            # lamda value for variational loss
        #boost.weak_trainer_type = 'LutTrainer'


        # Perform boosting of the feature set samp 
        model = boost_trainer.train(selected_trs, selected_trc)

        # Classify the test samples (testsamp) using the boosited classifier generated above
        prediction_labels = model.classify(fea_ts)

        accuracy = float(sum(prediction_labels == label_ts))/float(len(label_ts))
        print accuracy
        accu = accu + accuracy
print accu
print num
print accu/num
