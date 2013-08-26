
class GaussianMachine():

    def __init__(self, num_classes):
        self.means = numpy.zeros(num_classes)
        self.variance = numpy.zeros(num_classes)
        self.selected_index = 0


    def get_weak_scores(self, features):
        num_classes = self.means.shape[0]
        

        for i in range(num_classes):
            mean_i = means[i]
            variance_i = variance[i]
            feature_i = features[:,i]
            demon = numpy.sqrt(2*numpy.pi*variance_i)
            scores[i] = numpy.exp(-(((feature_i - mean_i)**2)/2*variance_i))

        return scores
             
class GaussianTrainer():

    def __init__(self, num_classes):
        self.num_classes = num_classes


    def compute_weak_trainer(self, features, loss_grad):

        num_features = features.shape[1]
        means = numpy.zeros([num_features,self.num_classes])
        variances = numpy.zeros([num_features,self.num_classes])
        summed_loss = numpy.zeros(num_features)
        gauss_machine = GaussianMachine()

        for feature_index in range(num_features)
            single_feature = features[:,feature_index]
            means[i,;], variances[i,:], loss[i] = compute_current_loss(single_feature, classes, loss_grad)

        gauss_machine.selected_index = numpy.argmin(loss)
        gauss_machine.means = means[optimum_index,:]
        gauss_machine.variance = variances[optimum_index,:]
        
        

    def compute_current_loss(feature, classes, loss_grad):
        num_samples = feature.shape[0]
        scores = numpy.zeros([num_samples, self.num_classes])

        for class_index in range(self.num_classes):
            samples_i = feature[loss_grad[:,class_index] < 0]
            mean[i] = numpy.mean(samples_i)
            variance[i] = numpy.std(samples_i)**2
            scores[:,class_index] = numpy.exp(-(((feature_i - mean_i)**2)/2*variance_i))
            

        scores_sum = numpy.sum(scores)

        return mean, variance, scores_sum

