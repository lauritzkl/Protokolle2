import matplotlib as mpl
mpl.use('pgf')
import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy
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

y = y_1**2

def f(x, a, b):
   y = a + b/(x**2)
   return y

params, covariance_matrix = curve_fit(f, x, y)
errors = np.sqrt(np.diag(covariance_matrix))

def g(x, c, d):
   y = c - d * x**2
   return y

params1, covariance_matrix1 = curve_fit(g, x, y)
errors1 = np.sqrt(np.diag(covariance_matrix1))

x_plot = np.linspace(min(x), max(x), 1000)

plt.plot(x, y, r'rx', label='Messwerte')
plt.plot(x_plot, f(x_plot, *params), 'k-', label='Regression für $\lambda >>\lambda_1$')
plt.plot(x_plot, g(x_plot, *params1), 'k-', label='Regression für $\lambda << \lambda_1$')
plt.legend()
plt.grid()
plt.ylabel(r'$n^2$')
plt.xlabel(r'$\lambda \, / \, nm$')
plt.savefig('plot1.pdf')
plt.close()
print('a=', params[0], '+-', errors[0])
print('b=', params[1], '+-', errors[1])
print('c=', params1[0], '+-', errors1[0])
print('d=', params1[1], '+-', errors1[1])


#Berechnung der Abweichungsquadrate

S1= (1/5)*np.sum((y**2-2.952+46604/x**2)**2)
#S2= (1/5)*np.sum((y**2-))
print(S1)
