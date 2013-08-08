import numpy as np
import sys
import math
from operator import itemgetter

# Function for training the model based on adaboost 
def train_adaboost(samples,classes,rnd):
    num_samples = len(samples)
    Cindex = len(samples[0]) -1
    # D = numpy.array.ones(num_samples,1)*(1/num_samples)
    num_pos = sum(samples[:,Cindex] == 1)
    num_neg = sum(samples[:,Cindex] == 0)

    D = np.zeros([num_samples,1],'float')
    D[classes == 1] =float(1)/float(2*num_pos)
    D[classes == 0] =float(1)/float(2*num_neg)
    alhpa = np.zeros([1,rnd])
    model = {}
    
    
    for t in range(rnd):
        gi = weak_learner(samples,classes,D)
        ei = abs(classes -  gi['pred_label']) 
        err = gi['err_sum']
        beta = err/(1-err)
        D = D*pow(np.transpose(beta),(1-ei))
        D = D/sum(D)
        alpha = math.log(1/max(beta,sys.float_info.epsilon))
        model[t] = {'alpha':alpha,'dim_index':gi['dim_ind'],'thresh':gi['thresh'],'p':gi['p']}
    
    return model
    



# Function to learn weak classifier based on single features
def weak_learner(samp,cla,D):
    num_samples = len(samp) 
    num_features = len(samp[0]) -1 
    Sp = 0
    Sn = 0

    Tp = sum(D[cla == 1])
    Tn = sum(D[cla == 0])

    # Index where weights D are stored in matrix fcd_data (fcd:features, class, D)
    Dindex = num_features +1

    # Index where the classes are stored in the matrix fcd_data
    Cindex = num_features 
    e_min = 1
    p = 1

    fcd_data = np.zeros([num_samples,Dindex+1])
    fcd_data[:,:-1] = samp
        # print D
    fcd_data[:,Dindex] = np.transpose(D) 

    for i in range(num_features):
        sorted_f = fcd_data[np.argsort(fcd_data[:,i])] #sorted(fcd_data,key=itemgetter(i))
        ei_min = 1
        #print fcd_data[:,i]
        for j in range(num_samples):
            curr_samp_class = sorted_f[j,Cindex]
            if curr_samp_class == 1:
                Sp = Sp + sorted_f[j,Dindex]
            if curr_samp_class == 0:
                Sn = Sn + sorted_f[j,Dindex]
            eij = min(Sp+Tn-Sn,Sn+Tp-Sp)
            #print eij

            if eij < ei_min:
                ei_min = eij
                th_i = (sorted_f[j,i] + sorted_f[j-1,i])/2
               # print "ei_min"
               #  print ei_min
                
       # print th_i               
        if ei_min < e_min:
            e_min = ei_min
            thresh = th_i
            dim_index = i
    #print "dimension"
    #print dim_index
    p_check = np.zeros([num_samples,1])
    p_check[fcd_data[:,dim_index] <= thresh] = 1
    #print np.transpose(p_check)
    if float(sum(p_check == cla))/float(num_samples) > 0.5:
        p = 1
    else:
        p = -1
    pred = np.zeros([num_samples,1],'float')
    if p == 1:
        pred[fcd_data[:,dim_index] <= thresh] = 1
    else:
        pred[fcd_data[:,dim_index] > thresh] = 1
    #print "threshold"
    #print thresh
    #print "features"
    #print fcd_data[:,dim_index]
    #print "prediction labels"
    #print np.transpose(pred)
    #print np.transpose(cla)
    err = sum(D*abs(pred - cla))
    #print err

    wl = {'dim_ind':dim_index,'thresh':thresh,'p':p,'err_sum':err,'pred_label':pred}
    return wl

# Function to classify feature based on strong classifier given by model
def test_adaboost(samples,model):
    num_samples = len(samples)
    rnd = len(model)
    sum_response = 0
    sum_alpha = 0
    for t in range(rnd):
        cur_model = model[t]
        cur_alpha = cur_model['alpha']
        cur_dim = cur_model['dim_index']
        cur_p = cur_model['p']
        cur_th = cur_model['thresh']
        weak_response = np.zeros([num_samples,1])
        if cur_p == 1:
            weak_response[samples[:,cur_dim] <= cur_th] = 1
        else:
            weak_response[samples[:,cur_dim] > cur_th] = 1
        sum_response = sum_response + cur_alpha*weak_response
        sum_alpha = sum_alpha + cur_alpha

    test_label = np.zeros([num_samples,1])
    test_label[sum_response >= 0.5*sum_alpha] = 1
    return test_label
