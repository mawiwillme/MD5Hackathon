
#import pyshark
from sklearn import *
import matplotlib.pylab as plt
import numpy as np

def y_noisy(a,b,x,var):
    noise = np.random.normal(0,var,len(x))
    return a+x*b+noise

def simulateVotes(var=0.05,alpha=0.5,beta=0,n_points=20):
    alpha2 = np.random.choice([0.25,0.75])    
    x = np.linspace(0,20,n_points)
    y = y_noisy(alpha2,beta,x,var)
    fig, ax = plt.subplots()
    scatter = ax.scatter(x,y,marker='o',label="y=x+0.5+noise")
    line = ax.plot(x,beta*x+alpha,label="y=x+0.5")
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    plt.legend()
    plt.show()

    

def classifier():
    print("classifying")
    # Runs scikit leanr classifier to learn patterns in data sets

simulateVotes()
