import numpy as np

EDV, HZV = np.genfromtxt('content/data.txt', unpack=True)

x = np.sum(EDV) / len(EDV)
dx = np.std(EDV, ddof=1) / np.sqrt(len(EDV))

y=np.sum(HZV) / len(HZV)
dy=np.std(HZV, ddof=1) / np.sqrt(len(HZV))


print(x)
print(y)
print("Mittelwert=", x, '+-', dx)
print("Mittelwert=", y, '+-', dy)
