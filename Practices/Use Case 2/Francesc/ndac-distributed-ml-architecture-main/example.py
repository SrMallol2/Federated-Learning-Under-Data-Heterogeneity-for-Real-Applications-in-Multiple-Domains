import matplotlib.pyplot as plt
import torch
from scipy.optimize import curve_fit
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from pylab import concatenate, normal
from torchmetrics.regression import MeanAbsolutePercentageError

mean_abs_percentage_error = MeanAbsolutePercentageError()

print(mean_abs_percentage_error(torch.from_numpy(np.array([0.0001]).reshape(-1, 1)),
                                torch.from_numpy(np.array([0.000001]).reshape(-1, 1))))

y1 = np.array([1,1,1,1])
y2 = np.array([9,9,9,9])

# weighted arithmetic mean (corrected - check the section below)
mean1 = np.mean(y1)
mean2 = np.mean(y2)
std1 = np.std(y1)
std2 = np.std(y2)


# Generate n1 samples from the first normal distribution and n2 samples from the second normal distribution
X = concatenate([normal(mean1, std1, 4), normal(mean2, std2, 4)]).reshape(-1, 1)

# Determine parameters mu1, mu2, sigma1, sigma2, w1 and w2
gm = GaussianMixture(n_components=2, random_state=0).fit(X)

print(f'mu1={gm.means_[0]}, mu2={gm.means_[1]}')
print(f'sigma1={np.sqrt(gm.covariances_[0])}, sigma2={np.sqrt(gm.covariances_[1])}')
print(f'w1={gm.weights_[0]}, w2={gm.weights_[1]}')
print(f'n1={int(4 * gm.weights_[0])} n2={int(4 * gm.weights_[1])}')

# mean, std = norm.fit(y1)
# print(mean)
# print(mean1)
# print(std)
# print(std1)

# print(aggregated_mean)
# print(np.mean(np.hstack((y1, y2))))
#
# print(aggregated_std)
# print(np.std(np.hstack((y1, y2))))