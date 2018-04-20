import matplotlib as mpl
mpl.use('pgf')
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
mpl.rcParams.update({
'font.family': 'serif',
'text.usetex': True,
'pgf.rcfonts': False,
'pgf.texsystem': 'lualatex',
'pgf.preamble': r'\usepackage{unicode-math}\usepackage{siunitx}',
})



y, x_1 = np.genfromtxt('content/data1.txt', unpack=True)


plt.plot(x_1, y, r'rx', label=r'Messwerte')
#plt.axvline(9.84, color='black', linestyle=':', label='Maximum')

def f(x_1, m, b):
    y = m*x_1+b
    return y

params, covariance_matrix = curve_fit(f, x_1, y)

errors = np.sqrt(np.diag(covariance_matrix))

plt.plot(x_1, f(x_1, *params), 'k-', label='Regression')
plt.legend()
plt.grid()
plt.xlabel(r'$Z$')
plt.ylabel(r'$\sqrt E / eV$')
plt.tight_layout()
plt.savefig('build/plot1.pdf')

print('m=', params[0], '+-', errors[0])
print('b=', params[1], '+-', errors[1])
