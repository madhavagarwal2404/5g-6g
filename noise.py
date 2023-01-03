import numpy as np
import matplotlib.pyplot as plt

N = 1024 # number of samples to simulate, choose any number you want
# x = np.random.randn(N) --command to generate noraml noise

# to generate and plot complex noise
x = (np.random.randn(N) + 1j*np.random.randn(N))/np.sqrt(2)
plt.plot(np.real(x), '.-')
plt.plot(np.imag(x), '.-')
plt.legend(['real','imag'])
plt.show()
# ploting  IQ graph for gaussian noise
plt.plot(np.real(x),np.imag(x),'.')
plt.grid(True, which='both')
plt.axis([-2, 2, -2, 2])
plt.show()


X = np.fft.fftshift(np.fft.fft(x))
X = X[N//2:] # only look at positive frequencies.  remember // is just an integer divide
plt.plot(np.real(X), '.-')
plt.show()