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

U_1 = 2.5
U = ufloat(2.5, 0.1768)
x, y_2 = np.genfromtxt('content/data1.txt', unpack=True)


y_1 = unumpy.log(y_2 - U)
y = unumpy.nominal_values(y_1)
dy = unumpy.std_devs(y_1)


x_1 = x[24:]
y_11 = y[24:]
x_2 = x[:12]
y_22 = y_2[:12]

def f(x_1, m, b):
    y_11 = m*x_1+b
    return y_11

params, covariance_matrix = curve_fit(f, x_1, y_11)

errors = np.sqrt(np.diag(covariance_matrix))

m_1 = ufloat(params[0], errors[0])
b_1 = ufloat(params[1], errors[1])

N_lang = unumpy.exp(m_1 * x_2 + b_1)

N_kurz = unumpy.log((y_22 - U) - N_lang)
N_k = unumpy.nominal_values(N_kurz)
dN_k = unumpy.std_devs(N_kurz)


def f(x_2, m_2, b_2):
    N_k = m_2*x_2+b_2
    return N_k

params1, covariance_matrix1 = curve_fit(f, x_2, N_k)
errors1 = np.sqrt(np.diag(covariance_matrix1))

m_21 = ufloat(params1[0], errors1[0])
b_21 = ufloat(params1[1], errors1[1])

plt.plot(x, y, r'rx', label='$\ln(N_{gesamt})$')
plt.plot(x_1, f(x_1, *params), 'k-', label='Regression von $\ln(N_{lang})$')
plt.errorbar(x, y, yerr = dy, xerr=None, fmt='none', ecolor='0.2')

plt.plot(x_2, N_k, r'bx', label='$\ln(N_{kurz})$')
plt.plot(x_2, f(x_2, *params1), 'g-', label='Regression von $\ln(N_{kurz})$')
plt.errorbar(x_2, N_k, yerr = dN_k, xerr=None, fmt='none', ecolor='0.6')

plt.legend()
plt.grid()
plt.ylabel(r'$\ln(N_{\Delta t})$')
plt.xlabel(r'$t \, / \, \si{\second}$')
plt.tight_layout()
plt.savefig('plot1.pdf')
plt.close()

print('Silber:')
print('lange Zerfallszeit:')
print('m=', params[0], '+-', errors[0])
print('b=', params[1], '+-', errors[1])
print('T_1=', -unumpy.log(2)/m_1)
print('kurze Zerfallszeit:')
print('m_2=', params1[0], '+-', errors1[0])
print('b_2=', params1[1], '+-', errors1[1])
print('T_2=', -unumpy.log(2)/m_21)


N_s = np.exp(params[0]*x+params[1]) + np.exp(params1[0]*x+params1[1])


plt.plot(x, N_s, r'k-', label='Summenkurve')
plt.plot(x, (y_2-U_1), r'rx', label='Messwerte')
plt.legend()
plt.grid()
plt.ylabel(r'$N_{\Delta t}$')
plt.xlabel(r'$t \, / \, \si{\second}$')
plt.tight_layout()
plt.savefig('plot2.pdf')
plt.close()


U = 60
dU = 4.24
x, y_1 = np.genfromtxt('content/data2.txt', unpack=True)

y_2 = (y_1 - U)


y = np.log(y_2)

yerr = np.array(dU/y_2)


def f(x, m, b):
    y = m*x+b
    return y

params, covariance_matrix = curve_fit(f, x, y)

errors = np.sqrt(np.diag(covariance_matrix))

m_3 = ufloat(params[0], errors[0])
b_3 = ufloat(params[1], errors[1])

plt.plot(x, y, r'rx', label='Messwerte')
plt.plot(x, f(x, *params), 'k-', label='Regression')
plt.errorbar(x, y, yerr, xerr=None, fmt='none', ecolor='0.2')
plt.legend()
plt.grid()
plt.ylabel(r'$\ln(N_{\Delta t})$')
plt.xlabel(r'$t \, / \, \si{\second}$')
plt.tight_layout()
plt.savefig('plot3.pdf')
plt.close()

print('Indium:')
print('m_3=', params[0], '+-', errors[0])
print('b_3=', params[1], '+-', errors[1])
print('T_3=', -unumpy.log(2)/m_3)
print('exp(b_3)=', unumpy.exp(b_3))
