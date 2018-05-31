import matplotlib as mpl
mpl.use('pgf')
import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.misc import factorial
mpl.rcParams.update({
'font.family': 'serif',
'text.usetex': True,
'pgf.rcfonts': False,
'pgf.texsystem': 'lualatex',
'pgf.preamble': r'\usepackage{unicode-math}\usepackage{siunitx}',
})

#Berechnung der mittleren Reichweite bei 2,5cm:

p, N, CH = np.genfromtxt('content/data.txt', unpack=True)

x = 2.5 * (p/1013)
E = (CH/258) * 4

x_1 = x[10:]
N_1 = N[10:]

def f(x_1, m, b):
    N_1 = m*x_1+b
    return N_1

params, covariance_matrix = curve_fit(f, x_1, N_1)

errors = np.sqrt(np.diag(covariance_matrix))

m_1 = ufloat(params[0], errors[0])
b_1 = ufloat(params[1], errors[1])

R_m = (53554-b_1)/m_1

plt.plot(x, N, r'rx', label='Messwerte')
plt.plot(x_1, f(x_1, *params), 'k-', label='Regression')

plt.legend()
plt.grid()
plt.ylabel(r'$N$')
plt.xlabel(r'$x \, / \, cm$')
plt.tight_layout()
plt.savefig('plot1.pdf')
plt.close()

print('Zählrate bei 2,5cm:')
print('m=', params[0], '+-', errors[0])
print('b=', params[1], '+-', errors[1])
print('R_m=', R_m)
print('E_m=', (R_m/3.1)**(2/3))

#Berechnung des Energieverlustes bei 2,5cm:

def f(x, m_2, b_2):
    E = m_2*x+b_2
    return E

params1, covariance_matrix1 = curve_fit(f, x, E)

errors1 = np.sqrt(np.diag(covariance_matrix1))


plt.plot(x, E, r'rx', label='Messwerte')
plt.plot(x, f(x, *params1), 'k-', label='Regression')

plt.legend()
plt.grid()
plt.ylabel(r'$E \, / \, MeV$')
plt.xlabel(r'$x \, / \, cm$')
plt.tight_layout()
plt.savefig('plot2.pdf')
plt.close()


print('Energie bei 2,5cm:')
print('m_2=', params1[0], '+-', errors1[0])
print('b_2=', params1[1], '+-', errors1[1])


#Berechnung der mittleren Reichweite bei 2cm:

p, N, CH = np.genfromtxt('content/data2.txt', unpack=True)

x = 2.5 * (p/1013)
E = (CH/608) * 4

x_1 = x[13:]
N_1 = N[13:]

def f(x_1, m, b):
    N_1 = m*x_1+b
    return N_1

params, covariance_matrix = curve_fit(f, x_1, N_1)

errors = np.sqrt(np.diag(covariance_matrix))

m_1 = ufloat(params[0], errors[0])
b_1 = ufloat(params[1], errors[1])

R_m = (42517-b_1)/m_1

plt.plot(x, N, r'rx', label='Messwerte')
plt.plot(x_1, f(x_1, *params), 'k-', label='Regression')

plt.legend()
plt.grid()
plt.ylabel(r'$N$')
plt.xlabel(r'$x \, / \, cm$')
plt.tight_layout()
plt.savefig('plot3.pdf')
plt.close()

print('Zählrate bei 2cm:')
print('m=', params[0], '+-', errors[0])
print('b=', params[1], '+-', errors[1])
print('R_m=', R_m)
print('E_m=', (R_m/3.1)**(2/3))

#Berechnung des Energieverlustes bei 2cm:

def f(x, m_2, b_2):
    E = m_2*x+b_2
    return E

params1, covariance_matrix1 = curve_fit(f, x, E)

errors1 = np.sqrt(np.diag(covariance_matrix1))


plt.plot(x, E, r'rx', label='Messwerte')
plt.plot(x, f(x, *params1), 'k-', label='Regression')

plt.legend()
plt.grid()
plt.ylabel(r'$E \, / \, MeV$')
plt.xlabel(r'$x \, / \, cm$')
plt.tight_layout()
plt.savefig('plot4.pdf')
plt.close()


print('Energie bei 2cm:')
print('m_2=', params1[0], '+-', errors1[0])
print('b_2=', params1[1], '+-', errors1[1])


#Statistik des Radioaktiven Zerfalls:

N = np.genfromtxt('content/data3.txt', unpack=True)

N_1 = N/100

plt.figure(1)
result = plt.hist(N_1, bins=100, label='Messwerte')
plt.xlim((min(N_1), max(N_1)))

mean = np.mean(N_1)
variance = np.var(N_1)
sigma = np.sqrt(variance)
x = np.linspace(min(N_1), max(N_1), 100)
dx = result[1][1] - result[1][0]
scale = len(N_1)*dx
plt.plot(x, mlab.normpdf(x, mean, sigma)*scale)

plt.legend()
#plt.ylabel(r'$E \, / \, MeV$')
plt.xlabel(r'$\Delta N$')
plt.tight_layout()
plt.savefig('plot5.pdf')
plt.close()

print('N_m=', mean)
print('N_v=', variance)

n, bins, patches = plt.hist(N_1, bins=100)

def poisson(k, v):
    return (v**k/factorial(k)) * np.exp(-v)

params, covariance_matrix = curve_fit(poisson, bins, n)

x_1 = np.linspace(min(N_1), max(N_1), 1000)
plt.plot(x_1, poisson(x_1, *params), 'r-')
#plt.legend()
#plt.ylabel(r'$E \, / \, MeV$')
plt.xlabel(r'$\Delta N$')
plt.tight_layout()
plt.savefig('plot6.pdf')
plt.close()
