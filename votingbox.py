import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM, SVC
from sklearn.neighbors import NearestNeighbors

"""
Anomaly detector.
"""
class AnomalyDetector:
    def __init__(self):
        pass
        
    def __densest_cloud__(self, x, idx):
        """
        Among n subsets of the data, returns the indices corresponding to the subsets by
        the density of the subsets.
        idx determines which subset each entry of x is in.
        """
        unique, counts = np.unique( idx, return_counts=True )
        dets = []
        for ii in unique:
            dets.append( np.linalg.det( np.cov(x[idx == ii,:].transpose()) ) )
        # Sort indices based on density of subsets
        sorted_idx = np.argsort( dets )
        return [unique[ii] for ii in sorted_idx]

    def __knn_move_outliers__(self, p, knn, knn_iters=1):
        """
        Move some of the outliers if they are close to a point in the main set
        """
        nbrs = NearestNeighbors(n_neighbors=knn+1, algorithm='ball_tree').fit(self.X_train)
        for kk in range(knn_iters):
            # Put outliers that have a neighbor in the main class into the main class
            distances, indices = nbrs.kneighbors(self.X_train)
            convert_to_main = []
            for ii in range(self.X_train.shape[0]):
                if p[ii] == self.main_class: continue
                if self.main_class in p[indices[ii,1:]]:
                    convert_to_main.append( ii )
            for ii in convert_to_main:
                p[ii] = self.main_class
        return p

    def fit(self, X, gamma0='auto', gamma1='auto', knn=2, C=1.0,
            knn_iters=2, nu=0.3):
        """
        Identifies the outliers in a dataset        
        Arguments:
          knn (int): if a positive integer, looks at the k nearest neighbors of each of the
            points in the outlier set; if any of them are in the non-outlier set then those
            points are no longer classified as being outliers. This can be used to reduce
            some of the overfitting caused by the one-class SVM.

          knn_iters (int): the number of times to apply the outlier moving process to the
            data.

        Return:
          outliers (np.ndarray): the predicted outliers in the dataset.
          dat (np.ndarray): the predicted non-outliers.
        """
        assert type(knn) is int and knn >= 0
        assert type(knn_iters) is int and knn_iters >= 0
        
        self.X_train = X
        self.oneclass = OneClassSVM( kernel='rbf', gamma=gamma0, nu=nu )
        self.oneclass.fit( self.X_train )
        p = self.oneclass.predict( self.X_train )
        
        # Determine which class has more points
        idx = self.__densest_cloud__( self.X_train, p )
        self.main_class, self.outlier_class = idx[0], idx[1]
        
        # Put some of the outliers in the main class of points
        if knn > 0:
            p = self.__knn_move_outliers__(p, knn, knn_iters=knn_iters)

        # Now fit an SVM to the labels learned by the one-class SVM
        self.svm = SVC( kernel='rbf', C=C, gamma=gamma1 )
        self.svm.fit( self.X_train, p )
        
    def predict( self, x ):
        p = self.svm.predict( x )
        outliers = (p == self.outlier_class)
        main     = (p == self.main_class)
        return (x[outliers,:], x[main,:])

    def fit_predict(self, X, **kwargs):
        self.fit(X, **kwargs)
        return self.predict(self.X_train)

"""
TESTING PURPOSES ONLY
"""
if __name__=="__main__":
    def gen_circle_data(N, r):
        """
        Generates random, circular data. Angles and radii are sampled from
        a uniform distribution.
        """
        thetas = np.random.uniform(-np.pi, np.pi, size=(N,1))
        rs = np.random.uniform(0, r, size=(N,1))
        x, y = rs * np.cos(thetas), rs * np.sin(thetas)
        return np.concatenate( [x, y], axis=1 )

    # Generate some circles, and then overlay some random points
    x1 = gen_circle_data(200, 1) + np.array([5, 5])
    x2 = gen_circle_data(200, 1) + np.array([-3, -1])
    x3 = gen_circle_data(200, 1) + np.array([2,1])
    noise = np.concatenate( [np.random.uniform(-3,5,size=(50,1)),
                             np.random.uniform(-1,5,size=(50,1))], axis=1 )
    data = np.concatenate( [x1, x2, x3, noise] )

    # First anomaly detector
    box = AnomalyDetector()
    outliers, main = box.fit_predict( data )
    plt.subplot( 121 )
    plt.scatter( outliers[:,0], outliers[:,1], label="Outliers" )
    plt.scatter( main[:,0], main[:,1], label="Main data" )
    plt.legend()
    
    # Second anomaly detector
    box = AnomalyDetector()
    outliers, main = box.fit_predict( data, C=0.2, nu=0.2 )
    plt.subplot( 122 )
    plt.scatter( outliers[:,0], outliers[:,1], label="Outliers" )
    plt.scatter( main[:,0], main[:,1], label="Main data" )
    plt.legend()
    
    plt.show()
