import matplotlib.pyplot as plt
import numpy as np
Fs = 1e6
N = 1024
t = np.arange(100)
s = np.sin(0.15*2*np.pi*t)
s = s*np.hamming(100)
S = np.fft.fft(s)
S_mag = np.abs(S)
S_phase = np.angle(S)
plt.figure(0)
plt.plot(t,S_mag,'.-')
plt.figure(1)
plt.plot(t,S_phase,'.-')
plt.show()

