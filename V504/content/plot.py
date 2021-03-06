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
x2, y2 = np.genfromtxt('content/data2.txt', unpack=True)
x3, y3 = np.genfromtxt('content/data3.txt', unpack=True)
x4, y4 = np.genfromtxt('content/data4.txt', unpack=True)
x5, y5 = np.genfromtxt('content/data5.txt', unpack=True)

plt.plot(x, y, 'rx', label='$I = 1.8A$')
plt.plot(x2, y2, 'kx', label='$I = 1.9A$')
plt.plot(x3, y3, 'bx', label='$I = 2.0A$')
plt.plot(x4, y4, 'gx', label='$I = 2.2A$')
plt.plot(x5, y5, 'yx', label='$I = 2.4A$')

plt.axhline(1.550, linestyle='dashed', color='yellow')
plt.axhline(0.438, linestyle='dashed', color='green')
plt.axhline(0.102, linestyle='dashed', color='blue')
plt.axhline(0.044, linestyle='dashed', color='black')
plt.axhline(0.018, linestyle='dashed', color='red')

plt.legend()
plt.grid()
plt.ylabel(r'$I \, / \, \text{mA}$')
plt.xlabel(r'$U \, / \, \text{V}$')
plt.tight_layout()
plt.savefig('plot1.pdf')
plt.close()



x_1, y_1 = np.genfromtxt('content/data5.txt', unpack=True)

x = x_1[0:19]
y = y_1[0:19]

def f(x, a, c, d):
    y = d * x**a + c
    return y

params, covariance_matrix = curve_fit(f, x, y)
errors = np.sqrt(np.diag(covariance_matrix))
x_plot = np.linspace(min(x), max(x), 1000)
plt.plot(x, y, 'rx', label='Messwerte')
plt.plot(x_plot, f(x_plot, *params), 'k-', label='Regression')
plt.legend()
plt.grid()
plt.ylabel(r'$I \, / \, \text{mA}$')
plt.xlabel(r'$U \, / \, \text{V}$')
plt.tight_layout()
plt.savefig('plot2.pdf')
plt.close()

print('b=', params[0], '+-', errors[0])
print('a=', params[2], '+-', errors[2])

x_2, y_1 = np.genfromtxt('content/data6.txt', unpack=True)
#y = y_1 * 10**(-9)
e = 1.602e-19
k = 1.38e-23
y = y_1[2:]
x_1 = x_2[2:]
x = x_1 - 10**(-3) * y


def f(x, b, a):
    y = a * np.exp(-b * x)
    return y

params, covariance_matrix = curve_fit(f, x, y)
errors = np.sqrt(np.diag(covariance_matrix))

b_1 = ufloat(params[0], errors[0])

T = e / (k*b_1)

x_plot = np.linspace(min(x), max(x), 1000)
plt.plot(x, y, 'rx', label='Messwerte')
plt.plot(x_plot, f(x_plot, *params), 'k-', label='Regression')
plt.legend()
plt.grid()
plt.ylabel(r'$I \, / \, \text{nA}$')
plt.xlabel(r'$U \, / \, \text{V}$')
plt.tight_layout()
plt.savefig('plot3.pdf')
plt.close()

print('a=', params[1], '+-', errors[1])
print('b=', params[0], '+-', errors[0])
print('T=', T)

U, I = np.genfromtxt('content/data7.txt', unpack=True)

N = 1
f = 0.35
n = 0.28
o = 5.7e-12

T = ((I*U - N)/(f*n*o))**(1/4)

print('T=', T)

I_s = np.array([0.018, 0.044, 0.102, 0.438, 1.550])

e = 1.602e-19
k = 1.381e-23
h = 6.626e-34
m = 9.109e-31

A = - (k * T)/ e * np.log((I_s * h**3)/(4 * f * np.pi * e * m * k**2 * T**2))


print('A=', A)
print('Mittelwert:')
print('A=', np.mean(A), '+-', np.std(A, ddof=1) / np.sqrt(len(A)))
