# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 10:39:18 2014
Implement EM fitting of a mixture of gaussians on the two-dimensional data 
set points.dat. You should try different numbers of mixtures, as well as 
tied vs. separate covariance matrices for each gaussian. 

IN EITHER CASE Use the final 1/10 of the data for dev. 
Plot likelihood on train and dev vs iteration for different numbers of mixtures.
@author: Md. Iftekhar Tanveer (itanveer@cs.rochester.edu)
"""
import numpy as np
import matplotlib.pyplot as matlab
import matplotlib.mlab as mlab


# Note: X and mu are assumed to be column vector
def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        norm_const = 1.0/(np.math.pow((2*np.pi), float(size)/2) * np.math.pow(det, 1.0/2))
        x_mu = np.matrix(x - mu)
        inv_ = np.linalg.inv(sigma)
        result = np.math.pow(np.math.e, -0.5 * (x_mu.T * inv_ * x_mu))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")
        return -1

# N = Number of data points
# M = Dimension of data points. In this homework, it is always 2
# K = Number of Gaussians in the mixture. In other words, number of "clusters"
def initEM(dataSet,K):
    # The weight matrix is an NxK matrix. I am initializing it by assigning the N points in K clusters
    # This assignment is arbitrary. So I am doing it based on the indices of the points. This process assigns
    # same number of points for each cluster
    (N, M) = np.shape(dataSet)
    W = np.zeros([N, K])
    nPerK = N/K
    for k in range(K):
        W[np.floor(k*nPerK):np.floor((k+1)*nPerK), k] = 1
    # Then MU, SIGMA and ALPHA are calculated by applying an M-step
    Alpha,Mu,Sigma = Mstep(dataSet,W)
    return W, Alpha, Mu, Sigma

def Mstep(dataSet,W):
    (N, M) = np.shape(dataSet)
    K = np.size(W,1)
    # Each column of MU represents the mean of a cluster. 
    # So, for K clusters, there will be K columns of MU
    # Each column,
    # mu_k = (1/N_k)*sum_{1}^{N}{w_{ik}*x_i} 
    N_k = np.sum(W,0)
    Alpha = N_k/np.sum(N_k)
    Mu = dataSet.T.dot(W).dot(np.diag(np.reciprocal(N_k)))
    # SIGMA is a 3-dimensional matrix of size MxMxK. 
    # It contains K covariances for each cluster
    Sigma = np.zeros([M,M,K])
    for k in range(K):
        datMeanSub = dataSet.T - Mu[0:,k][None].T.dot(np.ones([1,N]))
        Sigma[:,:,k] = (datMeanSub.dot(np.diag(W[0:,k])).dot(datMeanSub.T))/N_k[k]
    return Alpha,Mu,Sigma

def Estep(dataSet,Alpha,Mu,Sigma):
    # We will calculate the membership weight matrix W here. W is an
    # NxK matrix where (i,j)th element represents the probability of
    # ith data point to be a member of jth cluster given the parameters
    # Alpha, Mu and Sigma
    N = np.size(dataSet,0)
    K = np.size(Alpha)
    W = np.zeros([N,K])
    for k in range(K):
        for i in range(N):
            W[i,k] = Alpha[k]*norm_pdf_multivariate(dataSet[i,:][None].T, \
                     Mu[:,k][None].T,Sigma[:,:,k])
    # Normalize W row-wise because each row represents a pdf. In other words,
    # probability of a point to be any one of the K clusters is equal to 1.
    W = W*np.reciprocal(np.sum(W,1)[None].T)
    return W
    
def logLike(dataSet,Alpha,Mu,Sigma):
    K = len(Alpha)
    N,M = np.shape(dataSet)
    # P is an NxK matrix where (i,j)th element represents the likelihood of 
    # the ith datapoint to be in jth Cluster (i.e. when z_k = 1)
    P = np.zeros([N,K])
    for k in range(K):
        for i in range(N):
            P[i,k] = norm_pdf_multivariate(dataSet[i,:][None].T,Mu[0:,k][None].T,Sigma[:,:,k])
    return np.sum(np.log(P.dot(Alpha)))

def main():
    # Reading the data file
    input_file = open('points.dat')
    lines = input_file.readlines()
    allData = np.array([line.strip().split() for line in lines]).astype(np.float)
    (m, n) = np.shape(allData)

    # Separating out dev and train set
    devSet = allData[np.math.ceil(m*0.9):m, 0:]
    trainSet = allData[:np.math.floor(m*0.9), 0:]
    N = np.size(trainSet, 0)
    N_dev = np.size(devSet, 0)

    # Setting up initial settings. Change the cluster size manually
    # because otherwise the program becomes very slow
    tiedCov = False
    K = 16

    # Initialize the variables
    (W, Alpha, Mu, Sigma) = initEM(trainSet,K)

    # Temporary variables. X, Y mesh for plotting
    nx = np.arange(-4.0, 4.0, 0.1)
    ny = np.arange(-4.0, 4.0, 0.1)
    ax, ay = np.meshgrid(nx, ny)

    iter = 0
    prevll = -999999

    matlab.figure(2)
    matlab.clf()
    while(True):    
        # Iterate with E-Step and M-step
        # If the covariance is tied, use the sum of all cov
        if(tiedCov):
            SigmaSum = np.sum(Sigma,2)
            for k in range(K):
                Sigma[:,:,k] = SigmaSum
        W = Estep(trainSet,Alpha,Mu,Sigma)
        Alpha,Mu,Sigma = Mstep(trainSet,W)
        ll_train = logLike(trainSet,Alpha,Mu,Sigma)
        ll_dev = logLike(devSet,Alpha,Mu,Sigma)
        iter = iter + 1

        # For first window
        matlab.figure(1)
        # Plot the log-likelihood of the training data
        matlab.subplot(211)
        matlab.scatter(iter,ll_train,c='b')
        matlab.hold(True)
        matlab.xlabel('Iteration')
        matlab.ylabel('Log Likelihood of Training Data')

        # Plot the log likelihood of Development Data
        matlab.subplot(212)        
        matlab.scatter(iter,ll_dev,c='r')        
        matlab.hold(True)
        matlab.xlabel('Iteration')
        matlab.ylabel('Log Likelihood of Development Data')

        # Render these        
        matlab.draw()
        matlab.pause(0.01)

        # Plot the scatter plots and clusters
        matlab.figure(2)
        # Plot scatter plot of training data and corresponding clusters
        matlab.subplot(211)
        matlab.scatter(trainSet[0:,0],trainSet[0:,1])
        matlab.hold(True)
        for k in range(0, K):
            az = mlab.bivariate_normal(ax, ay, Sigma[0, 0, k], Sigma[1, \
                1, k], Mu[0,k], Mu[1,k], Sigma[1, 0, k])
            try:
                matlab.contour(ax, ay, az)
            except:
                continue
        matlab.hold(False)
        
        # Render these
        matlab.draw()
        matlab.pause(0.01)
        
                
        matlab.subplot(212)
        matlab.scatter(devSet[0:,0],devSet[0:,1])
        matlab.hold(True)
        for k in range(0, K):
            az = mlab.bivariate_normal(ax, ay, Sigma[0, 0, k], Sigma[1, \
                1, k], Mu[0,k], Mu[1,k], Sigma[1, 0, k])
            try:
                matlab.contour(ax, ay, az)
            except:
                continue
        matlab.hold(False)

        # Render these
        matlab.draw()
        matlab.pause(0.01)
        
        if(iter>150 or abs(ll_train - prevll)< 0.01):
            break
        print abs(ll_train - prevll)
        prevll = ll_train

        

if __name__ == '__main__':
    main()