import numpy as np
import matplotlib.pyplot as plt


## Pre task work
mean = 0
variance = 1

def norm_func(mean, variance, size, x_lim=5):
    x = np.linspace(-x_lim*np.sqrt(variance), x_lim*np.sqrt(variance), size)
    p = (1/(2*np.pi*variance))*np.exp(-np.square(x-mean)/np.square(2*variance))

    return p, x

norm_val, x_val = norm_func(mean, variance, 100, 5)

plt.figure()
plt.plot(x_val, norm_val)
plt.show()

## 1d)

sigmasq = 1.0
D = np.linspace(1e-6, 2*sigmasq,10000)
R = np.maximum(0.5*np.log(sigmasq/D),0)

plt.figure()
plt.plot(D, R)

plt.show()
