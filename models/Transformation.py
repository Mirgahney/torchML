import numpy as np

def point_feature(x,k):
    feature = []
    for i in range(k+1):
        feature.append(x**i)
    return np.array(feature)

def poly_features(X, K):
    
    #X: inputs of size N x 1
    #K: degree of the polynomial
    # computes the feature matrix Phi (N x (K+1))
    
    N, D = X.shape
    
    #initialize Phi
    Phi = np.zeros((N, (K+1)*D))
    
    # Compute the feature matrix in stages
    Phi = np.apply_along_axis(point_feature,0,X, K).T 
    return np.reshape(Phi, (N,-1))

def trig_point_feature(x,k):
    feature = []
    feature.append(1)
    for i in range(1,k+1):
        feature.append(np.sin(2*np.pi*i*x))
        feature.append(np.cos(2*np.pi*i*x))
    return np.array(feature, dtype=float)

def trigonometric_features(X, K):
    
    #X: inputs of size N x 1
    #K: degree of the polynomial
    # computes the feature matrix Phi (N x (K+1))
    
    N, D = X.shape
    X = X.flatten()
    
    #initialize Phi
    Phi = np.zeros((D*N, (2*K+1)))
    
    # Compute the feature matrix in stages
    #Phi = np.apply_along_axis(trig_point_feature,0,X, K).T # np.zeros((N, K+1))
    for i in range(N):
        Phi[i,:] = trig_point_feature(X[i],K)
    return np.reshape(Phi, (N,-1))

def gaussian_feature(x, l, k, mu_range = (0,1)):
    feature = []
    feature.append(1)
    mu = 0
    mu_step = (mu_range[1] - mu_range[0])/k
    for i in range(1,k+1):
        mu = i * mu_step
        phi = np.exp(-((x - mu)**2)/(2*(l**2)))
        feature.append(phi)
    return np.array(feature)

def gaussian_features(X, K,l, mu_range = (0,1)):
    
    #X: inputs of size N x 1
    #K: degree of the polynomial
    # computes the feature matrix Phi (N x (K+1))
    
    N, D = X.shape
    X = X.flatten()
    
    #initialize Phi
    Phi = np.zeros((D*N, K+1))
    
    # Compute the feature matrix in stages
    #Phi = np.apply_along_axis(point_feature,0,X, K).T # np.zeros((N, K+1)) ## <-- EDIT THIS LINE
    for i in range(N):
        Phi[i,:] = gaussian_feature(X[i], l, K, mu_range)
    return np.reshape(Phi, (N,-1))