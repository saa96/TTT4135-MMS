from random import sample
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

norm_samples = np.random.normal(0,1,100)
print(len(norm_samples))
print(norm_samples)

plt.figure()
plt.plot(x_val, norm_val)
plt.show()

## 1d)



sigmasq = 1.0
D = np.linspace(1e-6, 2*sigmasq,10000)
R = np.maximum(0.5*np.log(sigmasq/D),0)

# plt.figure()
# plt.plot(D, R)

# plt.show()

## 1e)
def uniform_quantizer(samples,nbits):
    quantizedSample = []
    sorted_samples = np.sort(samples)
    minVal = sorted_samples[0]
    maxVal = sorted_samples[-1]
    ValRange = maxVal - minVal
    quantInterval = ValRange/(2**nbits)
    print(quantInterval)
    for sample in samples:
        curr_interval = minVal + quantInterval/2
        while(sample>curr_interval):
            curr_interval += quantInterval
        quantizedSample.append(curr_interval-quantInterval/2)

    return quantizedSample

def test_quantizer(samples):

    errors = []
    print("testing")
    quantized_samples = np.sort(uniform_quantizer(samples,3))
    sorted_samples = np.sort(samples)
    for x in range(0,len(sorted_samples)):
        print(quantized_samples[x],sorted_samples[x])
        errors.append(abs(quantized_samples[x]-sorted_samples[x]))
    print("error",np.max(errors))
    ind = np.argmax(errors)
    print(ind)
    print(quantized_samples[ind],sorted_samples[ind])

test_quantizer(norm_samples)


##def quantize(nbits):
  ##  for()
    
