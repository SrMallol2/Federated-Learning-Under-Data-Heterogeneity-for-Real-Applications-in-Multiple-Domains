"""
Simple federated averaging demo using sklearn's load_breast_cancer dataset.

Usage: run the file. It will simulate multiple clients training locally
and the server averaging their weights each round, printing test accuracy.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


class SimpleLogistic:
	def __init__(self, n_features):
		self.w = np.zeros(n_features, dtype=float)
		self.b = 0.0

	def set_weights(self, w, b):
		self.w = w.copy()
		self.b = float(b)

	def predict_proba(self, X):
		z = X.dot(self.w) + self.b
		return 1.0 / (1.0 + np.exp(-z))

	def predict(self, X):
		return (self.predict_proba(X) >= 0.5).astype(int)

	def train(self, X, y, epochs=1, lr=0.1, init_weights=None):
		if init_weights is not None:
			self.set_weights(init_weights['w'], init_weights['b'])

		n = X.shape[0]
		for _ in range(epochs):
			p = self.predict_proba(X)
			grad_w = X.T.dot(p - y) / n
			grad_b = (p - y).mean()
			self.w -= lr * grad_w
			self.b -= lr * grad_b

		return {'w': self.w.copy(), 'b': float(self.b)}


def partition_data(X, y, num_parts, seed=0):
	rng = np.random.default_rng(seed)
	idx = np.arange(X.shape[0])
	rng.shuffle(idx)
	parts = np.array_split(idx, num_parts)
	return [(X[p], y[p]) for p in parts]


def federated_averaging(num_clients=3, rounds=10, local_epochs=5, lr=0.1):
	data = load_breast_cancer()
	X, y = data.data, data.target

	# train/test split and scale
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=1, stratify=y
	)
	scaler = StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	clients = partition_data(X_train, y_train, num_clients, seed=42)

	n_features = X_train.shape[1]
	# initialize global weights
	global_w = np.zeros(n_features, dtype=float)
	global_b = 0.0

	for r in range(1, rounds + 1):
		client_weights = []
		client_sizes = []

		for (Xc, yc) in clients:
			model = SimpleLogistic(n_features)
			init = {'w': global_w, 'b': global_b}
			cw = model.train(Xc, yc, epochs=local_epochs, lr=lr, init_weights=init)
			client_weights.append(cw)
			client_sizes.append(Xc.shape[0])

		# weighted average
		total = sum(client_sizes)
		avg_w = np.zeros_like(global_w)
		avg_b = 0.0
		for cw, sz in zip(client_weights, client_sizes):
			avg_w += cw['w'] * (sz / total)
			avg_b += cw['b'] * (sz / total)

		global_w = avg_w
		global_b = avg_b

		# evaluate
		server = SimpleLogistic(n_features)
		server.set_weights(global_w, global_b)
		preds = server.predict(X_test)
		acc = metrics.accuracy_score(y_test, preds)
		print(f"Round {r:2d}: test accuracy = {acc:.4f}")

	print("Done. Final accuracy:")
	server = SimpleLogistic(n_features)
	server.set_weights(global_w, global_b)
	preds = server.predict(X_test)
	print(metrics.classification_report(y_test, preds, digits=4))


if __name__ == '__main__':
	federated_averaging(num_clients=4, rounds=8, local_epochs=3, lr=0.2)

