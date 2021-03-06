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
plt.plot(x_plot, g(x_plot, *params1), 'b-', label='Regression für $\lambda << \lambda_1$')
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

a_1 = ufloat(params[0], errors[0])
b_1 = ufloat(params[1], errors[1])
c_1 = ufloat(params1[0], errors1[0])
d_1 = ufloat(params1[1], errors1[1])
W = np.array([656, 589, 486])
#Berechnung der Abweichungsquadrate

S1= (y - a_1 - b_1/x**2)**2
S2= (y - c_1 + d_1 * x**2)**2

L1 = unumpy.sqrt(b_1 / (a_1 - 1))

n = unumpy.sqrt(a_1 + (b_1/W**2))

A = ((3e7)*b_1)/(W**3 * unumpy.sqrt(a_1 + (b_1/W**2)))
Z = (n[1] - 1)/(n[2] - n[0])
print('Berechnung der Abweichungsquadrate:')
print('S_1=', 1/5 * sum(S1))
print('S_2=', 1/5 * sum(S2))
print('Fehler der Absorptionstelle=', L1)
print('Brechungsindices=', n)
print('Auflösungsvermögen', A)
print('Abbesche Zahl =', Z)

#Fehlerrechnung
p, e = np.genfromtxt('content/data2.txt', unpack=True)

phi=np.mean(p)
eta=np.mean(e)
dphi=np.std(p, ddof=1) / np.sqrt(len(p))
deta=np.std(e, ddof=1) / np.sqrt(len(e))


print('Winkelfehler')
print('phi=', phi, 'dphi=', dphi)
print('eta=', eta, 'deta=', deta)
