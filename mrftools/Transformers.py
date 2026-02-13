from sklearn.base import BaseEstimator, TransformerMixin  # This function just makes sure that the object is fitted
from sklearn.utils.validation import check_is_fitted
import numpy as np
try:
    import cupy as cp
except:
    pass
try:
    import matplotlib.pyplot as plt
except:
    pass 

class PCAComplex(BaseEstimator,TransformerMixin):

    def __init__(self, n_components_=None):
        self.n_components_=n_components_

    def fit(self, X):
        mean_X = np.mean(X, axis=0)
        cov_X = np.matmul(np.transpose((X - mean_X).conj()), (X - mean_X))

        if self.n_components_ is None:
            self.n_components_=cov_X.shape[0]

        X_val, X_vect = np.linalg.eigh(cov_X)

        sorted_index_X = np.argsort(X_val)[::-1]
        X_val = X_val[sorted_index_X]
        X_vect = X_vect[:, sorted_index_X]

        explained_variance_ratio = np.cumsum(X_val ** 2) / np.sum(X_val ** 2)
        if self.n_components_ is None:
            self.n_components_ = cov_X.shape[0]
        else :
            if self.n_components_<1:
                self.n_components_ = np.sum(explained_variance_ratio < self.n_components_) + 1
            else:
                self.n_components_=self.n_components_

        self.components_ = X_vect[:, :self.n_components_]
        self.singular_values_ = X_val[:self.n_components_]

        self.explained_variance_ratio_=explained_variance_ratio[:self.n_components_]
        self.mean_ = mean_X
        self.n_features_=X.shape[1]
        self.n_samples_=X.shape[0]

        return self


    def transform(self, X):
        # try:
        #    xp=cp.get_array_module(X)
        # except :
        #    print("Not using cupy in PCA transform")
        #    xp=np
        #    cp=None

        xp = cp.get_array_module(X)
        
        check_is_fitted(self,'explained_variance_ratio_')

        #X = X.copy()  # This is so we do not make changes to the

        if xp==cp:
            components = cp.asarray(self.components_)
        else:
            components = self.components_

        return xp.matmul(X, components.conj())

    def plot_retrieved_signal(self,X,i=0,len=None,figsize=(15,10)):
        X_trans = self.transform(X)
        retrieved_X = np.matmul(X_trans,np.transpose(self.components_))
        plt.figure(figsize=figsize)
        if len is None:
            len = X.shape[-1]
        plt.plot(np.abs(X[i,:len]),label="Original")
        plt.plot(np.abs(retrieved_X[i, :len]), label="Retrieved")
        plt.legend()
        plt.show()


