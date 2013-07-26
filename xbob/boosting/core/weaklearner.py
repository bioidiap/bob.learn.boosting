import numpy as np


class LutTrainer():
 
    def __init__(self, num_entries, s_type, num_op):
        self.num_entries = num_entries
        self.luts = np.ones((num_entries, num_op), dtype = np.int)
        self.s_type = s_type
        self.selected_indices = np.zeros([num_op,1], 'int16')


    def compute_weak_trainer(self, fea, loss_grad):
        # sum_loss = np.zeros([num_fea,1])
        #print np.transpose(loss_grad)
        num_op = loss_grad.shape[1]
        fea_grad = np.zeros([self.num_entries,num_op])
        sum_loss = self.compute_fgrad(loss_grad, fea)
        self.luts = np.ones((self.num_entries, num_op), dtype = np.int)
        print self.s_type
        if self.s_type == 'shared':
            selected_indices = [np.argmin(col) for col in np.transpose(sum_loss)]
            for oi in range(num_op):
                curr_id = sum_loss[:,oi].argmin()
                fea_grad[:,oi] = self.compute_hgrad(loss_grad[:,oi],fea[:,curr_id])
                self.selected_indices[oi] = curr_id
        elif self.s_type == 'indep':
            accum_loss = np.sum(sum_loss,1)
            selected_findex = accum_loss.argmin()
            self.selected_indices = selected_findex*np.ones([num_op,1],'int16')
            for oi in range(num_op):
                fea_grad[:,oi] = self.compute_hgrad(loss_grad[:,oi],fea[:,selected_findex])
        #print np.transpose(fea_grad)
     
        # print np.transpose(fea_grad)
        #print self.selected_indices
        self.luts[fea_grad <= 0.0] = -1
        #print np.transpose(self.luts)
        return self
    

    def compute_fgrad(self, loss_grad, fea):
        num_fea = len(fea[0])
        num_samp = len(fea)
        num_op = len(loss_grad[0])
        sum_loss = np.zeros([num_fea,num_op])
        for fi in range(num_fea):
            for oi in range(num_op):
                hist_grad = self.compute_hgrad(loss_grad[:,oi],fea[:,fi])
                sum_loss[fi,oi] = - sum(abs(hist_grad))
        #for u in range(num_entries):
        #    sum_loss[fi] = sum_loss[fi] - abs(hist_grad[u])

        return sum_loss


    def compute_hgrad1(self, loss_grado,fval):
        num_samp = len(fval)
        hist_grad = np.zeros([self.num_entries])
        for hi in range(self.num_entries):
            hist_grad[hi] = sum(loss_grado[fval == hi])
        return hist_grad
  
    def compute_hgrad(self, loss_grado,fval):
        num_samp = len(fval)
        hist_grad = np.zeros([self.num_entries])
        for hi in range(self.num_entries):
            hist_grad[hi] = sum(loss_grado[fval == hi])
        return hist_grad

	# Function computes classification results according to current weak classifier
    def get_weak_scores(self, fset):
		num_samp = len(fset)
		num_op = len(self.luts[0])
		weak_scores = np.zeros([num_samp,num_op])
		for oi in range(num_op):
			a = self.luts[fset[:,self.selected_indices[oi]],oi]
			weak_scores[:,oi] = np.transpose(self.luts[fset[:,self.selected_indices[oi]],oi])
		return weak_scores


'''    def ls_compute_grad(x,*args):
        targets = args[0]
        pred_scores = args[1]
        weak_scores = args[2]
        bl = args[3]
        print x
        curr_scores_x = pred_scores + x*weak_scores
        [loss, loss_grad] = bl.update_loss_grad(targets, curr_scores_x )
        cum_grad = np.sum(loss_grad * weak_scores,0)
        #print cum_grad
        return cum_grad

    # linesearch function to compute the loss function value at point x
    def ls_compute_fx(x, *args):
        targets = args[0]
        pred_scores = args[1]
        weak_scores = args[2]
        bl = args[3]
        print x
        print weak_scores.shape()
        curr_scores_x = pred_scores + x*weak_scores
        [loss, loss_grad] = bl.update_loss_grad(targets, curr_scores_x)
        cum_loss = np.sum(loss,0)
        #print cum_loss
        return cum_loss '''

