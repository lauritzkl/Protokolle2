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

I_0, I, L = np.genfromtxt('content/data1.txt', unpack=True)

y_1 = np.log(I/I_0)

def f(L, m, b):
    y_1 = m*L+b
    return y_1

params, covariance_matrix = curve_fit(f, L, y_1)

errors = np.sqrt(np.diag(covariance_matrix))

plt.plot(L, y_1, r'rx', label='Messwerte')
plt.plot(L, f(L, *params), 'k-', label='Regression')
plt.legend()
plt.grid()
plt.ylabel(r'$\ln(I/I_0)$')
plt.xlabel(r'$L \, / \, \si{\meter}$')
plt.tight_layout()
plt.savefig('plot1.pdf')
plt.close()

print('m=', params[0], '+-', errors[0])
print('b=', params[1], '+-', errors[1])

x, y = np.genfromtxt('content/data2.txt', unpack=True)

c = (2*x) / y
v = np.sum(c) / len(c)
dv = np.std(c, ddof=1) / np.sqrt(len(c))

print('Schallgeschwindigkeiten Echo:')
print(c)
print("Mittelwert=", v, '+-', dv)


x_1, y_2 = np.genfromtxt('content/data3.txt', unpack=True)

c_1 = x_1 / y_2
v_1 = np.sum(c_1) / len(c_1)
dv_1 = np.std(c_1, ddof=1) / np.sqrt(len(c_1))

print('Schallgeschwindigkeiten Durchschallung:')
print(c_1)
print("Mittelwert=", v_1, '+-', dv_1)
