import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM, SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator

"""
Anomaly detector when data is clustered into multiple groups.
"""
class AnomalyDetector(BaseEstimator):

    def __init__(self, gamma0='auto', gamma1='auto', knn=2, C=1.0,
                 knn_iters=2, nu=0.3):
        assert type(knn) is int and knn >= 0
        assert type(knn_iters) is int and knn_iters >= 0

        self.gamma0    = gamma0
        self.gamma1    = gamma1
        self.knn       = knn
        self.C         = 1.0
        self.knn_iters = knn_iters
        self.nu        = nu
        
    def __densest_cloud__(self, x, idx):
        """
        Among n subsets of the data, returns the indices corresponding to the subsets by
        the density of the subsets.
        idx determines which subset each entry of x is in.
        """
        unique, counts = np.unique( idx, return_counts=True )
        dets = []
        for ii in unique:
            # Use Cholesky decomposition to calculate log determinant. Note that covariance matrix
            # is symmetric positive semidefinite, so if Cholesky composition doesn't exist then
            # the log determinant is -Inf.
            try:
                L = np.linalg.cholesky( np.cov(x[idx == ii,:].transpose()) )
                det = np.sum([np.log(val**2) for val in np.diag(L)])
            except:
                det = -float("inf")
            #_, det = np.linalg.slogdet( np.cov(x[idx == ii,:].transpose()) )
            dets.append( det )
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

    def fit(self, X):
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
        self.X_train = X
        self.oneclass = OneClassSVM( kernel='rbf', gamma=self.gamma0, nu=self.nu )
        self.oneclass.fit( self.X_train )
        p = self.oneclass.predict( self.X_train )
        
        # Determine which class has more points
        idx = self.__densest_cloud__( self.X_train, p )
        self.main_class, self.outlier_class = idx[0], idx[1]
        
        # Put some of the outliers in the main class of points
        if self.knn > 0:
            p = self.__knn_move_outliers__(p, self.knn, knn_iters=self.knn_iters)
            # Fit a new svm to the labels learned by the one-class SVM and k-nearest neighbors
            # Now fit an SVM to the labels learned by the one-class SVM
            self.svm = SVC( kernel='rbf', C=self.C, gamma=self.gamma1 )
            self.svm.fit( self.X_train, p )
        else:
            self.svm = self.oneclass


    def predict( self, x ):
        p = self.svm.predict( x )
        outliers = (p == self.outlier_class)
        main     = (p == self.main_class)
        return (x[outliers,:], x[main,:])

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(self.X_train)

"""
TESTING PURPOSES ONLY
"""
if __name__=="__main__":
    import time
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
    DEFAULT_NX = 1000
    DEFAULT_NR = 200
    x1_gen     = lambda NX: gen_circle_data(NX, 1) + np.array([5, 5])
    x2_gen     = lambda NX: gen_circle_data(NX, 1) + np.array([-3, -1])
    x3_gen     = lambda NX: gen_circle_data(NX, 1) + np.array([2,1])
    noise_gen  = lambda NR: np.concatenate( [np.random.uniform(-3,5,size=(NR,1)),
                                         np.random.uniform(-1,5,size=(NR,1))], axis=1 )
    data_x = np.concatenate( [x1_gen(DEFAULT_NX), x2_gen(DEFAULT_NX),
                              x3_gen(DEFAULT_NX)] )
    data = np.concatenate( [data_x, noise_gen(DEFAULT_NR)] )

    # First anomaly detector
    box = AnomalyDetector()
    t0 = time.time()
    outliers, main = box.fit_predict( data )
    print( "Time to fit first model:", time.time() - t0 )
    plt.subplot( 121 )
    plt.scatter( outliers[:,0], outliers[:,1], label="Outliers" )
    plt.scatter( main[:,0], main[:,1], label="Main data" )
    plt.legend()
    
    # Second anomaly detector
    box = AnomalyDetector( C=0.2, nu=0.2, knn=2 )
    t0 = time.time()
    outliers, main = box.fit_predict( data )
    print( "Time to fit second model:", time.time() - t0 )
    plt.subplot( 122 )
    plt.scatter( outliers[:,0], outliers[:,1], label="Outliers" )
    plt.scatter( main[:,0], main[:,1], label="Main data" )
    plt.legend()
    plt.show()

    # Fit anomaly detector to main data, then do predictions on randomized data
    box = AnomalyDetector( C=0.2, nu=0.2, knn=3, knn_iters=3 )
    box.fit( data_x )
    outliers, main = box.predict( data )
    plt.scatter( outliers[:,0], outliers[:,1], label="Outliers" )
    plt.scatter( main[:,0], main[:,1], label="Main data" )
    plt.legend()
    plt.title( 'Performance after fitting only on "good" data' )
    plt.show()
    
    # Determine how the time complexity increases as the size of the data increases
    box = AnomalyDetector( C=0.2, nu=0.2, knn=0 )
    NX_vals = 10**np.linspace(2, 3.5, num=30)
    NX_actual = np.asarray( np.round( NX_vals ), dtype=np.int64 )
    times = []
    for (ii, NX) in enumerate(NX_actual):
        if ii % 10 == 0:
            print( "Iteration", ii )
        data_x = np.concatenate( [x1_gen(NX), x2_gen(NX), x3_gen(NX)] )
        t0 = time.time()
        _, _ = box.fit_predict( data_x )
        times.append( time.time() - t0 )
    plt.plot( 3 * NX_actual, times )
    plt.title( "Time to fit data" )
    plt.xlabel( "Number of data points" )
    plt.ylabel( "Time to fit (seconds)" )
    plt.show()
