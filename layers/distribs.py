import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import laplace
import math


if __name__ == "__main__":
    mu = 0
    sigma = 40
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-')
    plt.plot(x, laplace.pdf(x, mu, sigma), 'k-')
    plt.show()