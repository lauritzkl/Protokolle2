import numpy as np
from uncertainties import ufloat

x = ([4.60, 4.53, 4.77, 4.10, 4.60, 5.00, 5.62])

y = ufloat(4.746, 0.437)

print(np.std(x))
print(np.std(x, ddof=1) / np.sqrt(len(x)))
print((6.6e-34 * 299792458) / (y * 1.6e-19))
