import librosa
import numpy as np
import librosa.display as ld
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

s, sr2 = librosa.load("test_audio.wav")

# fig = plt.figure(figsize=(10,25))
# plt.subplot()

# D = librosa.core.amplitude_to_db(np.abs(librosa.stft(file))**2, ref=np.max)
# ld.specshow(D,x_axis='time' ,y_axis='log')
# plt.title('graph 1')


# file, sr2 = librosa.load("test_audio.wav")
# fig = plt.figure()
# plt.subplot()
# ld.waveshow(file,sr = sr2)
# plt.title('graph 2')
# plt.show()
# plt.show()

sample_rate = 1e6

# Generate tone plus noise
t = np.arange(1024*1000)/sample_rate
print(type(t))
# S = np.fft.fft(s)
S =s 
S_mag = np.abs(S)
S_phase = np.angle(S)
plt.figure(0)
plt.plot(t,S_mag,'.-')
plt.figure(1)
plt.plot(t,S_phase,'.-')
plt.show()