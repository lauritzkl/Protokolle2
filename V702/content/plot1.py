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


x, y_1 = np.genfromtxt('content/data1.txt', unpack=True)

y = np.log(y_1)

#def f(x, m, b):
#    y = m*x+b
#    return y

#params, covariance_matrix = curve_fit(f, x, y)

#errors = np.sqrt(np.diag(covariance_matrix))

plt.plot(x, y, r'rx', label='Messwerte')
#plt.plot(x, f(x, *params), 'k-', label='Regression')
plt.legend()
plt.grid()
plt.ylabel(r'$N_{\Delta t}$')
plt.xlabel(r'$t \, / \, \si{\second}$')
plt.tight_layout()
plt.savefig('plot1.pdf')

#print('m=', params[0], '+-', errors[0])
#print('b=', params[1], '+-', errors[1])
