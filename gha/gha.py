import numpy as np

class GHA(object):
	def __init__(self, n_features, n_input, dtype=None):
		self.n_features = n_features
		self.n_input = n_input
		self._W = np.array(np.random.rand(n_features, n_input)*2-1,dtype=dtype)
		self.error = None

	def fit(self, X):
		return self.partial_fit(X)

	def partial_fit(self, X, nu, remember_dW = False):
		Y = self.transform(X)
		dW = (np.outer(Y,X) - np.dot(np.tril(np.outer(Y,Y)),self._W))*nu
		if remember_dW:
			self._dW = dW
		self._W += dW
		return self

	def transform(self, X):
		return np.dot(self._W, X)

	def inverse_transform(self, Y):
		return np.dot(self._W.transpose(), Y)
