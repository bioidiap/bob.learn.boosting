import numpy as np
import math
import baseloss
import random
from pylab import *
from scipy import optimize

# function LUTBoost_train boosts the discrete features (fset) to provide a strong classifier 
# as a combintaion of weak classifier. Each weak classifier is represented by a feature index
# and a LUT for univariate case and a set of feature indices and LUTs for multivariate case.

# fset- (num_sam x num_features) features extracted from the samples
#        features should be discrete
# labels- class labels of the samples of dimension (#samples x #outputs)
# param- parameters for the boosting

def LUTBoost_train(fset,labels,param):
    # Initializations
    num_samp = len(fset)
    num_rnds = param['nrnds']
    num_entries = param['num_entries']
    num_op = param['num_op']
    s_type = param['selection_type']
    bl_type = param['baseloss_type']
    loss_type = param['loss_type']
    lamda = param['lamda']
    targets = labels
    pred_scores = np.zeros([num_samp,num_op])
    #pred_scores = np.zeros([num_samp])
    strong_learner = {}
    fset = fset.astype(int)
    
    # Start boosting iterations for num_rnds rounds
    for r in range(num_rnds):

        # Compute the baseloss, l(y,f(x)), and its gradient, l'(y,f(x)) using loss type ltype
        [loss, loss_grad] = baseloss.update_loss_grad(targets,pred_scores,bl_type,loss_type)
        # Select the best feature indices and corresponding LUTs for current round of boosting
        [luts, findices] = select_feature(loss_grad,fset,num_entries,s_type)
        # Compute the classification scores of the samples based only on the current round weak classifier
        weak_pred_scores = get_weak_scores(fset,luts,findices)
        # Use the line search method to find the alpha values for current classifier
        ls_res = optimize.fmin_l_bfgs_b(ls_compute_fx, 0, fprime = ls_compute_grad, args = (targets, pred_scores, weak_pred_scores, bl_type, loss_type)) 
        # Scale the LUTs with their corresponding alpha
        luts = scale_luts(luts,ls_res)
       
        # Update the current prediction score f(x) using the strong learner f 
        pred_scores =  update_scores(pred_scores, luts, fset, findices)
        strong_learner[r] = {'lut':luts,'findex':findices}
        print "Boosting done for round number : %d" % (r+1)
        
        
    return strong_learner

def ex_fun():
    print 'working'
        
def scale_luts(luts,ls_res):
    luts = luts*ls_res[0]
    return luts

def get_weak_scores(fset,luts,findices):
    num_samp = len(fset)
    num_op = len(luts[0])
    weak_scores = np.zeros([num_samp,num_op])
    for oi in range(num_op):
        a = luts [fset[:,findices[oi]],oi]
        weak_scores[:,oi] = np.transpose(luts[fset[:,findices[oi]],oi])
    return weak_scores

# linesearch function to compute the gradient of loss function at point x
def ls_compute_grad(x,*args):
    targets = args[0]
    pred_scores = args[1]
    weak_scores = args[2]
    bl_type = args[3]
    loss_type = args[4]
    curr_scores_x = pred_scores + x*weak_scores
    [loss, loss_grad] = baseloss.update_loss_grad(targets, curr_scores_x, bl_type, loss_type )
    cum_grad = np.sum(loss_grad * weak_scores,0)
    #print cum_grad
    return cum_grad

# linesearch function to compute the loss function value at point x
def ls_compute_fx(x, *args):
    targets = args[0]
    pred_scores = args[1]
    weak_scores = args[2]
    bl_type = args[3]
    loss_type = args[4]
    curr_scores_x = pred_scores + x*weak_scores
    [loss, loss_grad] = baseloss.update_loss_grad(targets, curr_scores_x, bl_type, loss_type)
    cum_loss = np.sum(loss,0)
    #print cum_loss
    return cum_loss 

def update_scores(pscores, luts, fset, selected_indices):
    for oi in range(luts.shape[1]):
        #print np.transpose(pscores)
        #print np.transpose(luts[:,fset[:,selected_indices[oi]],oi])
        pscores[:,oi] = pscores[:,oi] + np.transpose(luts[fset[:,selected_indices[oi]],oi])
    return pscores


def select_feature(loss_grad,fset,num_entries,selection_type):
    # sum_loss = np.zeros([num_fea,1])
    #print np.transpose(loss_grad)
    num_op = loss_grad.shape[1]
    fea_grad = np.zeros([num_entries,num_op])
    sum_loss = compute_fgrad(loss_grad,fset,num_entries)
    selected_indices = np.zeros([num_op,1],'int16')
    if selection_type == 'shared':
        selected_findices = [np.argmin(col) for col in np.transpose(sum_loss)]
        for oi in range(num_op):
            selected_indices[oi] = sum_loss[:,oi].argmin()
            fea_grad[:,oi] = compute_hgrad(loss_rgad[:,oi],fset[:,selected_findices[oi]],num_entries)
    elif selection_type == 'indep':
        accum_loss = np.sum(sum_loss,1)
        selected_findex = accum_loss.argmin()
        selected_findices = selected_findex*np.ones([num_op,1],'int16')
        for oi in range(num_op):
            fea_grad[:,oi] = compute_hgrad(loss_grad[:,oi],fset[:,selected_findex],num_entries)
    #print np.transpose(fea_grad)
    luts = np.ones([num_entries,num_op])
     
    # print np.transpose(fea_grad)
    luts[fea_grad <= 0.0] = -1
    #print np.transpose(lut)
    return luts, selected_findices
    

def compute_fgrad(loss_grad,fset,num_entries):
    num_fea = len(fset[0])
    num_samp = len(fset)
    num_op = len(loss_grad[0])
    sum_loss = np.zeros([num_fea,num_op])
    for fi in range(num_fea):

        for oi in range(num_op):
            hist_grad = compute_hgrad(loss_grad[:,oi],fset[:,fi],num_entries)
            sum_loss[fi,oi] = - sum(abs(hist_grad))
        #for u in range(num_entries):
        #    sum_loss[fi] = sum_loss[fi] - abs(hist_grad[u])

    return sum_loss


def compute_hgrad(loss_grado,fval,num_entries):
    num_samp = len(fval)
    hist_grad = np.zeros([num_entries])
    #print 'computing hgrad'
    for hi in range(num_entries):
        hist_grad[hi] = sum(loss_grado[fval == hi])
    return hist_grad


# Function to classify the features based on the model
def LUTBoost_classify(test_features,model):
    num_classifier = len(model)
    num_samples = len(test_features)
    test_features = test_features.astype(int)
    num_op = model[0]['lut'].shape[1]
    pred_scores = np.zeros([num_samples,num_op])
    for ri in range(num_classifier):
        curr_model = model[ri]
        curr_findex = curr_model['findex']
        curr_lut = curr_model['lut']
        for oi in range(curr_lut.shape[1]):
            curr_fea = test_features[:,curr_findex[oi]]
            pred_scores = pred_scores + curr_lut[curr_fea,oi]
    pred_labels = np.ones([num_samples,1])
    pred_labels[pred_scores <= 0] = -1
    return pred_labels
