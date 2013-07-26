import numpy as np

class StumpTrainer():
    def  __init__(self):
        self.th = 0
        self.p = 0
        self.selected_indices = 0

    def compute_weak_trainer(self, fea, loss_grad):
        labels = - np.sign(loss_grad)
        numSamp = fea.shape[0]
        numFea = fea.shape[1]

        th = np.zeros([numFea])
        p = np.zeros([numFea])
        g = np.zeros([numFea])

        for i in range(numFea):
            p[i],th[i], g[i] = self.compute_thresh(fea[:,i],loss_grad,labels)

        min_id = g.argmax()
        self.th = th[min_id]
        self.p = p[min_id]
        self.selected_indices = min_id
        print min_id
        print self.th
        print self.p
        print self.selected_indices
        return self

    # Function to compute the threshold for a single feature. The threshold is computed for 
    # the given feature values using the weak learner algorithm given in the Voila Jones Robust Face classification
    # Inputs:
    # f: The feature values for a single index, array of dimension N = No. of training Samples
    # loss_grad: The loss gradient values for the training samples. Array of dimension N.
    # labels: The class of the training samples. Array of dimension N.
    # Return:(weak stump classifier for given feature)
    # th: threshold which minimizes the error
    # p: the polarity or the direction used for stump classification
    # err: error of the classifier
    def compute_thresh(self, f,loss_grad,labels):
        # The weights for Adaboost are negative of exponential loss gradient
        wt = -loss_grad/labels
        loss_grad = -loss_grad
        wt = wt/sum(wt)
        num_samp = f.shape[0]
        wt_p = np.empty_like(wt)
        wt_n = np.empty_like(wt)

        # Sort the feature and rearrange the corresponding labels and wt values
        sorted_id = np.argsort(f)
        f = f[sorted_id] 
        labels = labels[sorted_id]
        loss_grad = loss_grad[sorted_id]
        wt = wt[sorted_id]
        grad_cs =  np.cumsum(loss_grad)
        #print np.transpose(grad_cs[1:40])
        grad_sum = grad_cs[-1]
        #print grad_cs
        g = (grad_sum - grad_cs)
        #print np.transpose(g[1:40])
        #print g[-1]
        opt_id = np.argmax(np.absolute(g))
        err = np.absolute(g[opt_id])
        if(opt_id == num_samp-1):
            th = f[opt_id]
        else:
            th = float(f[opt_id] + f[opt_id+1])/2
        #print opt_id
        #print th
        #print err
        if(err == g[opt_id]):
            p = -1
        else:
            p = 1
        return p, th, err

        '''# Compute Tp and Tn values as mentioned in the paper
        tn = sum(wt[labels == -1])
        tp = sum(wt[labels == 1])
        wt_p[:] = wt
        temp = labels == -1
        wt_p[labels == -1] = 0
        sp =  np.cumsum(wt_p)
        wt_n[:] = wt
        wt_n[labels == 1] = 0
        sn = np.cumsum(wt_n)
        
        # Compute the error for the thresholds
        err_vec = np.minimum(sp + tn - sn, sn + tp - sp)
        min_id = np.argmin(err_vec)
        err = err_vec[min_id]

        if(min_id != 0):
            thresh = float(f[min_id] + f[min_id-1])/2
        else:
            thresh = f[min_id]

        # Compute the polarity for the 
        p_check = np.ones([num_samp,1])
        p_check[f <= thresh] = -1
         #print np.transpose(p_check)
        cur_err = float(sum(p_check == labels))/float(num_samp) 
        if float(sum(p_check == labels))/float(num_samp) > 0.5:
            p = 1
        else:
            p = -1
        return p, thresh, err'''

    ''' Function to compute the threshold for a single feature f. The threshold th
    # minimizes the value sum() 
    def compute_thresh(self, f,loss_grad,labels):
        # 
        numSamp = f.shape[0]
        sorted_ids = np.argsort(f)
        f = f[sorted_ids]
        f = f[:,np.newaxis]
        loss_grad = loss_grad[sorted_ids]
        wts = -loss_grad
        labels = labels[sorted_ids]
        th_mat = np.tile(np.transpose(f),(numSamp,1))
        f_mat = np.tile(f,numSamp)
        wts_mat = np.tile(wts,numSamp)
        polarity_vec = np.ones([1,numSamp])
        label_mat = np.tile(labels,numSamp)
        pred_mat = np.ones([numSamp,numSamp])
        pred_mat[f_mat >th_mat] = -1
        pred_sum = np.zeros([numSamp,1],'float')
        pred_sum = (sum(pred_mat == label_mat,0))
        err_vec = pred_sum/float(numSamp)
        err_vec = (err_vec[:,np.newaxis])
        print err_vec.shape
        print polarity_vec.shape 
        polarity_vec[err_vec <0.5] = -1
        err_vec[err_vec <0.5] = 1 - err_vec[err_vec <0.5]
        polarity_mat = np.tile(polarity_vec,(numSamp,1))
        g = sum(wts_mat*polarity_mat*pred_mat,0)
        'g = np.zeros([numSamp,1])
        wts = -loss_grad
        polarity = np.ones([numSamp,1])
        print numSamp
        for i in range(numSamp):
            if(np.remainder(i,100) == 0):
                print i
            pred = np.ones([numSamp,1])
            pred[f>f[i]] = -1
            err = sum(pred == labels)/numSamp
            if err < 0.5:
                err = 1-err
                polarity[i] = -1
            g[i] = sum(wts*polarity*pred)'

        opt_ind = np.argmax(g)
        p_opt = polarity_vec[:,opt_ind]
        g_opt = g[opt_ind]
        if(opt_ind == numSamp-1):
            th = f[opt_ind]
        else:
            th = (f[opt_ind] + f[opt_ind+1])/2
        return p_opt, th, g_opt'''


    def get_weak_scores(self,test_features):
        numSamp = test_features.shape[0]
        weak_scores = np.ones([numSamp,1])
        weak_features = test_features[:,self.selected_indices]
        weak_scores[weak_features < self.th] = -1
        weak_scores = self.p *weak_scores
        return weak_scores


