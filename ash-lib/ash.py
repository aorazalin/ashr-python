from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt

def pct_delta(x, y):
    if isinstance(x, np.ndarray):
        return (np.abs(x-y).sum())/((np.abs(x).sum() + np.abs(y).sum())/2)
    elif isinstance(x, float) or isinstance(x, np.float):
        return np.abs(x - y) / ((np.abs(x) + np.abs(y))/2)

def ash(betahat, sebetahat, mode="n", debug=False, ret_pi=False):
    logsigma_min = np.log(np.min(sebetahat) / 10)
    if (sebetahat**2 - betahat**2).max() > 0:
        logsigma_max = np.log(np.sqrt(2*(betahat**2 - sebetahat**2).max()))
    else:
        logsigma_max = np.log(8) + logsigma_min
    m = np.log(np.sqrt(2))
    logsigma_grid = np.arange(logsigma_min, logsigma_max, m)
    sigma = np.exp(logsigma_grid)
    
    n = len(betahat)
    K = len(sigma)
    
    # EM algorithm
    # initialization
    pi = np.r_[1-K/n, np.ones(K)*1/n]
    pi_old = None
    max_iters = 1000
    n_iters = 0
    eps = 1e-6
    lam = np.r_[10, np.ones(K)]
    
    
    # pre-compute l[k, j] := p(betahat[j] | sebetahat, regime[j]=k)
    l = np.zeros((K+1, n))
    for k in range(K+1):
        for j in range(n):
            if k == 0:
                l[k, j] = norm.pdf(betahat[j], loc=0, scale=sebetahat[j])
            else:
                l[k, j] = norm.pdf(betahat[j], loc=0, scale=np.sqrt(sebetahat[j]**2+sigma[k-1]**2))
        
    def loss_func(pi):                    
        return np.log((pi.reshape((-1, 1)) * l).sum(axis=0)).sum() + (lam-1)@np.log(pi)
    
    losses = []
    while (n_iters == 0 or pct_delta(loss_func(pi), loss_func(pi_old)) >= eps) and n_iters < max_iters:
        pi_old = pi
        w = (pi.reshape((K+1, 1)) * l) / (pi.reshape((K+1, 1)) * l).sum(axis=0).reshape((1, n))
        # E-step
        cnts = w.sum(axis=1) + lam - 1
        # M-step
        pi = cnts / cnts.sum()
        n_iters += 1
        losses.append(loss_func(pi))
        
    if debug:
        plt.plot(losses)
        plt.title("EM loss")
        plt.show()
        
    if ret_pi:
        return pi
        
    if n_iters == max_iters:
        print("WARNING: EM algorithm hasn't converged in", n_iters, "iterations!")
        
    # compute lfdr
    lfdr = pi[0]*l[0,:]/(pi.reshape((K+1, 1)) * l).sum(axis=0)
    return lfdr