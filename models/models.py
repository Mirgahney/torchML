import torch
from torch import matmul, inverse

class LinearRegression:
    """
    Ordinary least squares Linear Regression.

    Parameters
    ----------
    fit_intercept : boolean, optional, default True
        whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, optional, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit`` on
        an estimator with ``normalize=False``.

    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This will only provide
        speedup for n_targets > 1 and sufficient large problems.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    coef_ : array, shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    intercept_ : array
        Independent term in the linear model.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> # y = 1 * x_0 + 2 * x_1 + 3
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.score(X, y)
    1.0
    >>> reg.coef_
    array([1., 2.])
    >>> reg.intercept_ # doctest: +ELLIPSIS
    3.0000...
    >>> reg.predict(np.array([[3, 5]]))
    array([16.])

    Notes
    -----
    
    """
    
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=None, penality = 'l2', l = 0.1):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.penality = penality
        self.l = l
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
        """
        Fit linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data

        y : array_like, shape (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape [n_samples]
            Individual weights for each sample

            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.

        Returns
        -------
        self : returns an instance of self.
        """
        
        self.X = X
        self.y = y
        
        if self.copy_X:
            self.X = X.clone()
            
        if self.normalize:
            # normalize
            std = self.X.std()
            self.X -=self.X.mean()
            self.X /= std
        if self.penality == 'l2':
            self.theta =  matmul(inverse(matmul(self.X.t(),self.X) + self.l*torch.eye(self.X.shape[1]).double()), matmul(self.X.t(),self.y))
            self.fitted = True
        
    def get_report(self,X,y):
        y_perd = self.predict(X)
        mse = self.MSE(y, y_perd)
        pass
        
    def MSE(self, y, y_pred):
        return matmul((y - y_pred).view(-1,1).t(), (y - y_pred).view(-1,1)).mean() 
