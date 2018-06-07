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

x, y = np.genfromtxt('content/data1.txt', unpack=True)

def f(x, a, b):
   y = np.sqrt(b+a/x**2)
   return y

params, covariance_matrix = curve_fit(f, x, y)
errors = np.sqrt(np.diag(covariance_matrix))

plt.plot(x, y, r'rx', label='Messwerte')
plt.plot(x, f(x, *params), 'k-', label='Regression')
plt.legend()
plt.grid()
plt.ylabel(r'$n$')
plt.xlabel(r'$Wellenlänge \, / \, nm$')
plt.savefig('plot1.pdf')
plt.close()
print('a=', params[0], '+-', errors[0])
print('b=', params[1], '+-', errors[1])


def g(x, c, d):
   y = np.sqrt(c-d*x**2)
   return y

params, covariance_matrix1 = curve_fit(g, x, y)
errors = np.sqrt(np.diag(covariance_matrix1))
plt.plot(x, y, r'rx', label='Messwerte')
plt.plot(x, f(x, *params), 'k-', label='Regression')
plt.legend()
plt.grid()
plt.ylabel(r'$n$')
plt.xlabel(r'$Wellenlänge \, / \, nm$')
plt.savefig('plot2.pdf')
plt.close()
print('c=', params[0], '+-', errors[0])
print('d=', params[1], '+-', errors[1])

#Berechnung der Abweichungsquadrate

S1= (1/5)*np.sum((y**2-2.952+46604/x**2)**2)
#S2= (1/5)*np.sum((y**2-))
print(S1)
