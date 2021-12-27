'''kmeans.py
Performs K-Means clustering
TAMSIN ROGERS
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors
from scipy.cluster.vq import kmeans2

class KMeans():

    '''KMeans constructor'''
    def __init__(self, data=None):
    
        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        # Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    '''replaces the data instance variable with given data'''
    def set_data(self, data):

        self.data = data
        self.num_samps = data.shape[0]
        self.num_features = data.shape[1]

    '''returns a copy of the data'''
    def get_data(self):

        dat = self.data.copy()
        return dat

    '''returns the stored k-means centroids'''
    def get_centroids(self):
        
        return self.centroids

    '''returns the data-to-cluster assignments'''
    def get_data_centroid_labels(self):
        
        return self.data_centroid_labels

    '''computes  the Euclidean distance between data samples pt_1 and pt_2'''
    def dist_pt_to_pt(self, pt_1, pt_2):
        
        d = np.linalg.norm(pt_1 - pt_2)
        return float(d)

    '''computes the Euclidean distance between data sample pt and all the other cluster centroids'''
    def dist_pt_to_centroids(self, pt, centroids):
       
        sq = np.square(np.subtract(centroids, pt))
        result = np.sqrt(np.sum(sq, axis=1))
        return result
    
    '''initializes K-means by setting the initial centroids (means) to K unique randomly selected data samples'''
    def initialize(self, k):
    
        '''self.k = k
        cent,_ = kmeans2(self.data, self.k, minit="random")
        self.centroids = cent
        
        return self.centroids'''
        
        self.k = k
        
        idxs = np.random.choice( self.num_samps, k, replace=False )
        
        self.centroids = self.data[idxs,:]
        
        return self.data[idxs,:]

    '''performs K-means clustering on the data'''
    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False):
        
        self.k = k
        self.num_samps = self.data.shape[0]
        self.num_features = self.data.shape[1]
        centroids = self.initialize(k)
        labels = self.update_labels(self.centroids)
        new_centroids, centroid_diff = self.update_centroids(self.k, labels, centroids)
    
        i = 0
        
        while i < max_iter and (centroid_diff > tol).all():
            labels = self.update_labels(new_centroids)
            new_centroids, centroid_diff = self.update_centroids(self.k, labels, new_centroids)
            i+=1
        
        self.centroids = new_centroids
        self.data_centroid_labels = labels
        self.inertia = self.compute_inertia()
        
        if verbose:
            print("error")
        
        return (self.inertia, i)

    '''runs K-means mutiple times with different initial conditions & sets variables based on best K-means run'''
    def cluster_batch(self, k=2, n_iter=1, verbose=False):
       
        cents = np.zeros((n_iter,k,self.num_features))
        labs = np.zeros((n_iter, self.num_samps))
        inerts = np.zeros((n_iter,))

        for i in range(n_iter):
            inert, itera = self.cluster(k=k)
            cents[i] = self.centroids
            labs[i] = self.data_centroid_labels
            inerts[i] = self.inertia
          
        best = np.argmin(inerts)
        self.centroids = cents[best]
        self.data_centroid_labels = labs[best]
        self.inertia = inerts[best]
        
        if verbose:
            print("error")

    '''assigns each data sample to the nearest centroid'''
    def update_labels(self, centroids):

        index = []
        for i in range(self.data.shape[0]):
            hold = self.dist_pt_to_centroids(self.data[i,:], centroids)
            best = min(hold)
            bestindex = np.where(hold == best)
            index.append(bestindex[0][0].astype('int'))
        return np.array(index)

    '''computes each of the K centroids (means) based on the data assigned to each cluster'''
    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        
        new_centroids = np.zeros((k,self.num_features))
        centroid_diff = np.zeros((k,self.num_features))
        
        for i in range(k):
            points = self.data[data_centroid_labels == i] 
            for j in points:
                new_centroids[i] = np.mean(points, axis = 0)
        centroid_diff = new_centroids - prev_centroids
        
        return (new_centroids, centroid_diff)

    '''computes the mean squared distance between every data sample and its assigned (nearest) centroid'''
    def compute_inertia(self):
        
        centroids_array = self.centroids[self.data_centroid_labels.astype('int')] 
        d = self.data - centroids_array
        sq = d * d
        s = np.sum(sq, axis=1)
        sumsqdist = np.sum(s)
        meansqdist = sumsqdist / self.data.shape[0]
        
        return meansqdist

    '''creates a scatter plot of the data color-coded by cluster assignment'''
    def plot_clusters(self):
       
        colors = cartocolors.qualitative.Safe_10.mpl_colors             # use the colorblind friendly colors library 
        
        for i in range(self.k):
            dat = self.data[self.data_centroid_labels == i] 
            plot = plt.scatter(dat[:,0], dat[:,1],color = colors[i])    # pick a new color for each cluster
            
        plt.plot(self.centroids[:,0], self.centroids[:,1], "x", c="black")

    '''creates an elbow plot cluster number (k) on x axis, inertia on y axis'''
    def elbow_plot(self, max_k):
        
        inertiaList = np.zeros((max_k,))

        for i in range (1, max_k + 1):
            inertia, iterations = self.cluster(k = i)
            inertiaList[i-1] = inertia

        fig, axes = plt.subplots(1,1)
        axes.plot(inertiaList)
        axes.set_xticks(np.arange(1, max_k+1))
        axes.set_title("Elbow Plot: k vs. inertia")
        axes.set_xlabel('k')
        axes.set_ylabel('inertia')
        plt.xticks(np.arange(0, max_k), np.arange(1, max_k + 1).astype('str'))

    '''replaces each RGB pixel in self.data (flattened image) with the closest centroid value'''
    def replace_color_with_centroid(self):
        
        for i in range(self.num_samps):
            index = self.data_centroid_labels[i]
            self.data[i, :] = self.centroids[index, :]