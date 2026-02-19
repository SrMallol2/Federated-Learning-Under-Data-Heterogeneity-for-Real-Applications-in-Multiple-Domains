
"""
Simple knowledge distillation demo using sklearn's load_breast_cancer dataset.

This script trains a stronger "teacher" logistic model on the dataset,
then trains a smaller "student" logistic to match a mixture of hard labels
and the teacher's soft probabilities (distillation). Prints accuracies.
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

	def train(self, X, y_target, epochs=100, lr=0.1, init_weights=None):
		if init_weights is not None:
			self.set_weights(init_weights['w'], init_weights['b'])
		n = X.shape[0]
		for _ in range(epochs):
			p = self.predict_proba(X)
			grad_w = X.T.dot(p - y_target) / n
			grad_b = (p - y_target).mean()
			self.w -= lr * grad_w
			self.b -= lr * grad_b
		return {'w': self.w.copy(), 'b': float(self.b)}


def knowledge_distillation(num_teacher_epochs=300, num_student_epochs=120,
						   student_lr=0.2, teacher_lr=0.15, distill_lambda=0.7):
	data = load_breast_cancer()
	X, y = data.data, data.target

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=1, stratify=y
	)
	scaler = StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	n_features = X_train.shape[1]

	# Train teacher on true labels (stronger: more epochs, lower lr)
	teacher = SimpleLogistic(n_features)
	teacher.train(X_train, y_train, epochs=num_teacher_epochs, lr=teacher_lr)
	t_preds = teacher.predict(X_test)
	t_acc = metrics.accuracy_score(y_test, t_preds)
	print(f"Teacher accuracy = {t_acc:.4f}")

	# Teacher soft probabilities (to be used as targets)
	teacher_probs = teacher.predict_proba(X_train)

	# Student trained on hard labels only (baseline)
	student_baseline = SimpleLogistic(n_features)
	student_baseline.train(X_train, y_train, epochs=num_student_epochs, lr=student_lr)
	sb_preds = student_baseline.predict(X_test)
	sb_acc = metrics.accuracy_score(y_test, sb_preds)
	print(f"Student (hard labels) accuracy = {sb_acc:.4f}")

	# Student trained with distillation: combine hard labels and teacher probs
	# y_distill = (1 - lambda)*hard + lambda*teacher_soft
	y_distill_targets = (1.0 - distill_lambda) * y_train + distill_lambda * teacher_probs

	student = SimpleLogistic(n_features)
	student.train(X_train, y_distill_targets, epochs=num_student_epochs, lr=student_lr)
	s_preds = student.predict(X_test)
	s_acc = metrics.accuracy_score(y_test, s_preds)
	print(f"Student (distilled) accuracy = {s_acc:.4f}")

	print("\nFinal classification reports:\n")
	print("Teacher:\n", metrics.classification_report(y_test, t_preds, digits=4))
	print("Student (hard):\n", metrics.classification_report(y_test, sb_preds, digits=4))
	print("Student (distilled):\n", metrics.classification_report(y_test, s_preds, digits=4))


if __name__ == '__main__':
	knowledge_distillation()

