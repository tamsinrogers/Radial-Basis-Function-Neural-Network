'''rbf_net.py
Radial Basis Function Neural Network
TAMSIN ROGERS
CS 251: Data Analysis Visualization, Spring 2021
'''
import numpy as np
import kmeans
import scipy.linalg


class RBF_Net:
    def __init__(self, num_hidden_units, num_classes):
        '''RBF network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset

        TODO:
        - Define number of hidden units as an instance variable called `k` (as in k clusters)
            (You can think of each hidden unit as being positioned at a cluster center)
        - Define number of classes (number of output units in network) as an instance variable
        '''
        # prototypes: Hidden unit prototypes (i.e. center)
        #   shape=(num_hidden_units, num_features)
        self.prototypes = None
        # sigmas: Hidden unit sigmas: controls how active each hidden unit becomes to inputs that
        # are similar to the unit's prototype (i.e. center).
        #   shape=(num_hidden_units,)
        #   Larger sigma -> hidden unit becomes active to dissimilar inputs
        #   Smaller sigma -> hidden unit only becomes active to similar inputs
        self.sigmas = None
        # wts: Weights connecting hidden and output layer neurons.
        #   shape=(num_hidden_units+1, num_classes)
        #   The reason for the +1 is to account for the bias (a hidden unit whose activation is always
        #   set to 1).
        self.wts = None

        self.k = num_hidden_units
        self.num_classes = num_classes

    '''returns the hidden layer prototypes (centers)'''
    def get_prototypes(self):
        
        return self.prototypes

    '''returns the number of hidden layer prototypes (centers/"hidden units")'''
    def get_num_hidden_units(self):

        return self.k

    '''returns the number of output layer units'''
    def get_num_output_units(self):

        return self.num_classes
        
    '''compute the average distance between each cluster center and data points that are assigned to it'''
    def avg_cluster_dist(self, data, centroids, cluster_assignments, kmeans_obj):

        sigma = []
        
        for i in range(len(centroids)):
            found = []
            find = np.where(cluster_assignments == i)[0]
           
            for j in find:
                found.append(data[j,:])
            
            founds = np.array(found)
            
            dat = founds - centroids[i]
            square = np.square(dat)
            summed = np.sum(square,axis=1)
            sqrt = np.sqrt(summed)
            me = np.mean(sqrt)
            sigma.append(me)

        sigmas = np.array(sigma)
        return sigmas

    '''initializes hidden unit centers using K-means clustering and initialize sigmas using the average distance within each cluster'''
    def initialize(self, data):
        
        kMeansObj = kmeans.KMeans(data)
        kMeansObj.cluster_batch(k=self.k, n_iter=5)
        self.prototypes = kMeansObj.centroids
        clusterAssignments = kMeansObj.data_centroid_labels
        self.sigmas = self.avg_cluster_dist(data=data, centroids=self.prototypes, cluster_assignments=clusterAssignments, kmeans_obj=kMeansObj)
    
    '''performs linear regression'''
    def linear_regression(self, A, y):
        
        Ahat = np.hstack([A, np.ones((len(A), 1))])			# handles the intercept/homogenous coordinate
        temp = np.linalg.lstsq(Ahat, y, rcond=None)[0]
        return temp

    '''computes the activation of the hidden layer units'''
    def hidden_act(self, data):
        
        alphas = []
        Num_Nodes = self.k
        act_mat = np.zeros((data.shape[0], Num_Nodes))
        
        for i in range(len(self.sigmas)):
        	a = 1/(2 * (self.sigmas[i] ** 2) + 0.00000001)
        	alphas.append(a)
        alphas = np.array(alphas)
        
        for i in range(data.shape[0]):
        	for j in range(self.k):
        		sq = (((data[i,:] - self.prototypes[j,:])**2).sum())
        		m = alphas[j] * sq
        		e = np.exp(-m)
        		act_mat[i][j] = e
        
        return act_mat

    '''computes the activation of the output layer units'''
    def output_act(self, hidden_acts):
        
        ones = np.ones((1, hidden_acts.shape[0]))
        h = np.hstack((hidden_acts, ones.T))
        out = h@self.wts
        return out

    '''trains the radial basis function network'''
    def train(self, data, y):
        
        self.initialize(data)
        act_mat = self.hidden_act(data)
        
        new = np.empty((y.shape[0], self.num_classes))
        weights = []
        
        for j in range(self.num_classes):
        	for i in range(y.shape[0]):
        		if y[i] == j:
        			new[i][j] = 1
        		else:
        			new[i][j] = 0
        
        for i in range(self.num_classes):
        	c = self.linear_regression(act_mat, new[:, i])
        	weights.append(c)
        
        self.wts = np.array(weights)
        self.wts = self.wts.T

	'''classifies each sample in the given data'''
    def predict(self, data):
        
        y_pred = []
        h_a = self.hidden_act(data)
        o_p = self.output_act(h_a)
        
        for i in range(o_p.shape[0]):
        	c_num = np.argmax(o_p[i])
        	y_pred.append(c_num)
        
        return np.array(y_pred)

	'''computes accuracy based on percent correct: Proportion of predicted class labels `y_pred` that match the true values `y`'''
    def accuracy(self, y, y_pred):
        
        a = np.sum(y == y_pred) / y.shape[0]
        return a
