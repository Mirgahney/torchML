import torch
from torch import matmul, inverse
from LinearRegression import LinearRegression
import numpy as np

class NonLinearRegression:
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=None, penality = 'l2', l = 0.1, transformer = 'poly', K = 3, length = 0.1, mu_range = (0,1)):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.penality = penality
        self.l = l
        self.transformer = transformer
        self.K = K
        self.length = length
        self.mu_range = mu_range
        self.theta = None
        self.X = None
        self.y = None
        self.fitted = False
        
        self.lr = LinearRegression(self.fit_intercept, self.normalize, self.copy_X, self.n_jobs, self.penality, self.l)
  
        
    def predict(self,X):
       
    
    def fit(self, X,y):
        
        
    def get_report(self,X,y):
        
        
    def MSE(self, y, y_pred):
        return matmul((y - y_pred).view(-1,1).t(), (y - y_pred).view(-1,1)).mean() 
