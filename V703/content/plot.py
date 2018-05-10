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


x, y, z = np.genfromtxt('content/data1.txt', unpack=True)

x_1=x[9:25]
y_1=y[9:25]

def f(x_1, m, b):
    y_1 = m*x_1+b
    return y_1

params, covariance_matrix = curve_fit(f, x_1, y_1)

errors = np.sqrt(np.diag(covariance_matrix))

s=ufloat(params[0], errors[0])
e=ufloat(params[1], errors[1])



plt.plot(x, y, r'rx', label='Messwerte')
plt.plot(x_1, f(x_1, *params), 'g-', label='Regressionsgerade')
plt.errorbar(x, y, yerr = np.sqrt(y) , xerr=None, fmt='none', ecolor='0.2')
plt.axvline(400, color='blue', linestyle=':', label='Plateau-Bereich')
plt.axvline(550, color='blue', linestyle=':')
plt.legend()
plt.grid()
plt.ylabel(r'$N$')
plt.xlabel(r'$U \, / \, V$')
plt.tight_layout()
plt.savefig('plot1.pdf')

print('m=', params[0], '+-', errors[0])
print('b=', params[1], '+-', errors[1])
print('m in % =', ((s*600+e)/(s*500+e)) * 100 - 100)


dN1= ufloat(13100, np.sqrt(13100))
dN2= ufloat(1393, np.sqrt(1393))
dN12=ufloat(14658, np.sqrt(14658))

print('T=', (dN1 + dN2 - dN12)/(2*dN1*dN2))

dy = np.sqrt(y)

N = unumpy.uarray(y, dy)
I = z * 10**(-6)

n = (I* 60)/(N * 1.6e-19)

print(n*10**(-10))
