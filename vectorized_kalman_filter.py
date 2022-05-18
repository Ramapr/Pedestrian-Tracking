# -*- coding: utf-8 -*-
"""
Created on Tue May 17 23:38:48 2022

@author: roma
"""

import numpy as np
from copy import deepcopy
from numba import njit, vectorize
from numba import float64


#%%
@njit(parallel=True, fastmath=True)
def _update(z, x, P, R, H):
    Hx = np.dot(H, x)
    # error (residual) between measurement and prediction
    y = z - Hx
    
    PHT = np.dot(P, H.T)
    S = np.dot(H, PHT) + R

    # project system uncertainty into measurement space
    #S = np.dot(np.dot(H, P), H.T) + R
    SI = np.linalg.inv(S) 
    
    # map system uncertainty into kalman gain
    K = np.dot(PHT, SI)
    #np.dot(np.dot(P, H.T), np.linalg.inv(S))

    # predict new x with residual scaled by the kalman gain
    x_ = x + np.dot(K, y)

    # P = (I-KH)P(I-KH)' + KRK'
    KH = np.dot(K, H)

    I_KH = np.eye(KH.shape[0]) - KH
    P_ = np.dot(np.dot(I_KH, P), I_KH.T) + np.dot(np.dot(K, R), K.T)
    return x_, P_

@njit(parallel=True, fastmath=True)
def _predict(x, P, F, Q, alpha):
    x_ = np.dot(F, x)
    P_ = alpha * np.dot(np.dot(F, P), F.T) + Q
    #P = (alpha * alpha) * np.dot(np.dot(F, P), F.T) + Q
    return x_, P_
#%%
"""
необходимо проверить !
"""

@vectorize([float64[:](float64[:], float64[:])])
def x_pred(x, F):
    return np.dot(F, x)

@vectorize([(float64[:])(float64[:], float64[:], float64[:], float64)])
def p_pred(P, F, Q, alpha):
    return alpha * np.dot(np.dot(F, P), F.T) + Q

def vec_predict(x, P, F, Q, alpha):
    x_ = x_pred(F, x)
    P_ = p_pred(P, F, Q, alpha)
    return x_, P_

@vectorize([(float64[:])(float64[:], float64[:])])
def x_update(x, H):
    return np.dot(H, x)

@vectorize([(float64[:])(float64[:], float64[:])])
def y_update(z, Hx):
    return z - Hx

@vectorize([(float64[:])(float64[:], float64[:], float64[:])])
def xx(x, K, y):
    return x + np.dot(K, y)

@vectorize([(float64[:])(float64[:], float64[:], float64[:])])
def k_update(P, H, R):
    PHT = np.dot(P, H.T)
    S = np.dot(H, PHT) + R
    # project system uncertainty into measurement space
    #S = np.dot(np.dot(H, P), H.T) + R
    SI = np.linalg.inv(S) 
    # map system uncertainty into kalman gain
    return np.dot(PHT, SI) # K

@vectorize([(float64[:])(float64[:], float64[:])])
def ikh_update(K, H):
    KH = np.dot(K, H)
    return np.eye(KH.shape[0]) - KH

@vectorize([(float64[:])(float64[:], float64[:], float64[:], float64[:])])
def p_update(P, I_KH, K, R):
    return np.dot(np.dot(I_KH, P), I_KH.T) + np.dot(np.dot(K, R), K.T)

def vec_update(z, x, P, R, H):
    Hx = x_update(H, x)
    # error (residual) between measurement and prediction
    y = y_update(z, Hx)
    K = k_update(P, H, R)
    #PHT = np.dot(P, H.T)
    #S = np.dot(H, PHT) + R
    # project system uncertainty into measurement space
    #S = np.dot(np.dot(H, P), H.T) + R
    #SI = np.linalg.inv(S)  
    # map system uncertainty into kalman gain
    #K = np.dot(PHT, SI)
    ##np.dot(np.dot(P, H.T), np.linalg.inv(S))

    # predict new x with residual scaled by the kalman gain
    x_ = xx(x, K, y) #x + np.dot(K, y)

    ## P = (I-KH)P(I-KH)' + KRK'
    #KH = np.dot(K, H)
    #I_KH = np.eye(KH.shape[0]) - KH
    I_KH = ikh_update(K, H)
    P_ = p_update(P, I_KH, K, R)
    #np.dot(np.dot(I_KH, P), I_KH.T) + np.dot(np.dot(K, R), K.T)
    return x_, P_


#%%

class KF:
    def __init__(self, dimx, dimz, mode=False):
      # 
      #
      self.x = np.zeros((dimx, 1)) 
      self.F = np.eye(dimx)
      self.H = np.eye(dimz, dimx)
      self.R = np.eye(dimz) 
      self.P = np.eye(dimx)  
      self.Q = np.eye(dimx)
      
      #self.M = np.zeros((dim_x, dim_z)) # process-measurement cross correlation
      self.z = np.array([[None]*dimz]).T
      self.alpha_sq = 1.
      # these will always be a copy of x,P after predict() is called
      self.x_prior = self.x.copy()
      self.P_prior = self.P.copy()
      
      self.P_post = self.P.copy()
      self.x_post = self.x.copy()
      self.mode = mode
      #predict_func = vec_predict() if self.mode else _predict()
      # переделать с ифа на ламбду
      
    def predict(self):
      # x = (4, 1)
      if self.mode:
          self.x, self.P = vec_predict(self.x, self.P, self.F, self.Q, self.alpha_sq)
      else:    
          self.x, self.P = _predict(self.x, self.P, self.F, self.Q, self.alpha_sq)
      #self.x_prior = x_pr.copy()
      #self.P_prior = P_pr.copy()
      return self.x
    
    def update(self, z):
      # z : (dim_z, 1): array_like
      if self.mode:
          self.x, self.P = vec_update(z, self.x, self.P, self.R, self.H)
      else:
          self.x, self.P = _update(z, self.x, self.P, self.R, self.H)
      self.z = deepcopy(z)
      self.x_post = self.x.copy()
      self.P_post = self.P.copy()
      #return self.x, self.P
      
            
#%%

dimx = 7
dimz = 4

kf = KF(dimx, dimz)
kf.F += np.eye(dimx, k=dimx-3)
#kf.F 
kf.H = np.eye(dimz, dimx)
kf.R = 10. * np.eye(dimz)
kf.R[2:,2:] *= 10.
kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
kf.P *= 10.
kf.Q[-1,-1] *= 0.01
kf.Q[4:,4:] *= 0.01
# @njit([(float64[:])(float64, float64, float64[:], float64[:])])

#%%
y = np.array([[1.,2.,3.,4.]]).reshape((4, 1)) #.shape

print(y)
kf.update(y)
kf.predict()

#y = np.array([[1.4,1.2,1.3,1.4]]).reshape((4, 1))#
#kf.update(y)      
#_update(x, kf.P, y, kf.R, kf.H)
#
#%%
# njit 0.07978177070617676
# python 0.01

import time 
np.random.seed(10)
yy = np.random.normal(size=(100, 4, 1))

start = time.time()
for i in yy:
    _ = kf.update(i)
    _ = kf.predict()

end = time.time()
print("Elapsed = %s" % (end - start))

_predict.parallel_diagnostics(level=1)
_update.parallel_diagnostics(level=1)

#%%
