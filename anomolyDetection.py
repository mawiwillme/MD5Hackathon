
#import pyshark
from sklearn import linear_model
from sklearn.neighbors import DistanceMetric
import matplotlib.pylab as plt
import numpy as np

def regressionOnData(data):

    data[9] = 2
    std = data.std()
    mean = data.mean()
    print("mean {}".format(mean))
    print("std:{}".format(std))
    for i in range(len(data)):
        #print(mean + std)
        #print(mean - std)
        if((mean + 2*std) < data[i] or (mean - 2*std) > data[i]):
            print(data[i])

    '''
    # Generate some 2D points from a multivariate standard normal distribution
    x = np.random.normal( 0, 1, size=(100, 2))
    mean = np.mean( x, axis=0 ).reshape( (1,2) )  # Reshape so that we're (1,2) instead of (2,)
    # Calculate sample covariance matrix
    # Need to reshape x temporarily to 2 x 100 so that the covariance matrix is 2 x 2 and
    # not 
    V = np.cov( x.reshape(2,100) )  # Calculate sample covariance matrix
    metric = DistanceMetric.get_metric( "mahalanobis", V=V )
    
    dist = metric.pairwise( x, mean ).reshape( (-1,) )  
    print( dist.shape )


    # Plot the data
    plt.subplot( 121 )
    plt.scatter( x[:,0], x[:,1] )
    plt.xlim( -3, 3 )
    plt.ylim( -3, 3 )
    plt.title( "Original data" )
    
    # Remove top 20% of outliers from the data using the mahalanobis distances
    idx = np.argsort( dist )
    keep_size = int(round(x.shape[0] * 0.8))
    x_cleaned = np.empty( shape=(keep_size, x.shape[1]) )
    for ii in range(keep_size):
        x_cleaned[ii,:] = x[idx[ii],:]
    plt.subplot( 122 )
    plt.scatter( x_cleaned[:,0], x_cleaned[:,1] )
    plt.xlim( -3, 3 )
    plt.ylim( -3, 3 )
    plt.title( "Data with top 20% of outliers removed" )
    plt.show()
    '''

def y_noisy(a,b,x,var):
    noise = np.random.normal(0,var,len(x))
    return a+x*b+noise

def simulateVotes(var=0.5,alpha=3.15,beta=0,n_points=20):
    alpha2 = np.random.choice([2.5,4.0])    
    x = np.array(np.linspace(0,20,n_points))
    y = np.array(y_noisy(alpha,beta,x,var))

    regressionOnData(x,y)
    regressionOnData
    '''
    fig, ax = plt.subplots()
    scatter = ax.scatter(x,y,marker='o',label="y=x+0.5+noise")
    line = ax.plot(x,beta*x+alpha,label="y=x+0.5")
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    plt.legend()
    plt.show()
    '''
   

simulateVotes()
