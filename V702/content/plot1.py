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

U = 2.5
dU = 0.1768
x, y_2 = np.genfromtxt('content/data1.txt', unpack=True)

y_1 = (y_2-U)
y = np.log(y_1)

yerr = np.array(dU/y_1)
x_1 = x[23:]
y_11 = y[23:]

def f(x_1, m, b):
    y_11 = m*x_1+b
    return y_11

params, covariance_matrix = curve_fit(f, x_1, y_11)

errors = np.sqrt(np.diag(covariance_matrix))

plt.plot(x, y, r'rx', label='Messwerte')
plt.plot(x_1, f(x_1, *params), 'k-', label='Regression')
plt.errorbar(x, y, yerr, xerr=None, fmt='none', ecolor='0.2')
plt.legend()
plt.grid()
plt.ylabel(r'$\ln(N_{\Delta t})$')
plt.xlabel(r'$t \, / \, \si{\second}$')
plt.tight_layout()
plt.savefig('plot1.pdf')
print('m=', params[0], '+-', errors[0])
print('b=', params[1], '+-', errors[1])
