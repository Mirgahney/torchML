import torch
from torch import matmul, inverse

class LinearRegression:
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=None):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.theta = None
        self.X = None
        self.y = None
        self.fitted = False
  
        
    def predict(self,X):
        if not self.fitted:
            print('Need to fit the model fisrt use model.fit')
            pass
        else:
            return matmul(X, self.theta)
    
    def fit(self, X,y):
        self.X = X
        self.y = y
        
        if self.copy_X:
            self.X = X.clone()
            
        if self.normalize:
            # normalize
            std = self.X.std()
            self.X -=self.X.mean()
            self.X /= std
            
        self.theta =  matmul(inverse(matmul(self.X.t(),self.X)), matmul(self.X.t(),self.y))
        self.fitted = True
        
    def get_report(self,X,y):
        y_perd = self.predict(X)
        mse = self.MSE(y, y_perd)
        pass
        
    def MSE(self, y, y_pred):
        return matmul((y - y_pred).view(-1,1).t(), (y - y_pred).view(-1,1)).mean() 
