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

U = 60
dU = 4.243
x, y_1 = np.genfromtxt('content/data2.txt', unpack=True)

y_2 = (y_1 - U)


y = np.log(y_2)

yerror = np.array(dU/y_2)

def f(x, m, b):
    y = m*x+b
    return y

params, covariance_matrix = curve_fit(f, x, y)

errors = np.sqrt(np.diag(covariance_matrix))

plt.plot(x, y, r'rx', label='Messwerte')
plt.plot(x, f(x, *params), 'k-', label='Regression')
plt.errorbar(x, y, yerr=yerror, xerr=None, fmt='none', ecolor='0.2')
plt.legend()
plt.grid()
plt.ylabel(r'$\ln(N_{\Delta t})$')
plt.xlabel(r'$t \, / \, \si{\second}$')
plt.tight_layout()
plt.savefig('plot2.pdf')
print('m=', params[0], '+-', errors[0])
print('b=', params[1], '+-', errors[1])
