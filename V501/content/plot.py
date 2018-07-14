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
#Versuch 1
#Berechnung zur Empfindlichkeit
x, y = np.genfromtxt('content/data1.txt', unpack=True)
def f(x, a, b):
    y = a*x + b
    return y
params, covariance_matrix = curve_fit(f, x, y)
errors = np.sqrt(np.diag(covariance_matrix))
x_plot = np.linspace(min(x), max(x), 1000)
plt.plot(x, y, 'rx',   label='$U_b = 200V$')
plt.plot(x_plot, f(x_plot, *params),'r-', label='Regression')
plt.legend()
plt.grid()
plt.xlabel(r'$U_d \, / \, \si{\volt}$')
plt.ylabel(r'$D \, / \, \si{\centi\meter}$')
plt.tight_layout()
plt.savefig('plot1.pdf')
plt.close()


x2, y2 = np.genfromtxt('content/data2.txt', unpack=True)
def g(x2, c, d):
    y2 = c*x2 + d
    return y2
params1, covariance_matrix1 = curve_fit(g, x2, y2)
errors1 = np.sqrt(np.diag(covariance_matrix1))
x2_plot = np.linspace(min(x2), max(x2), 1000)
plt.plot(x2, y2, 'rx',   label='$U_b = 250V$')
plt.plot(x2_plot, f(x2_plot, *params1), 'r-', label='Regression')
plt.legend()
plt.grid()
plt.xlabel(r'$U_d \, / \, \si{\volt}$')
plt.ylabel(r'$D \, / \, \si{\centi\meter}$')
plt.tight_layout()
plt.savefig('plot2.pdf')
plt.close()

x3, y3 = np.genfromtxt('content/data3.txt', unpack=True)
def h(x3, e, f):
    y3 = e*x3 + f
    return y3
params2, covariance_matrix2 = curve_fit(h, x3, y3)
errors2 = np.sqrt(np.diag(covariance_matrix2))
x3_plot = np.linspace(min(x3), max(x3), 1000)
plt.plot(x3, y3, 'rx',   label='$U_b = 300V$')
plt.plot(x3_plot, f(x3_plot, *params2), 'r-', label='Regression')
plt.legend()
plt.grid()
plt.xlabel(r'$U_d \, / \, \si{\volt}$')
plt.ylabel(r'$D \, / \, \si{\centi\meter}$')
plt.tight_layout()
plt.savefig('plot3.pdf')
plt.close()

x4, y4 = np.genfromtxt('content/data4.txt', unpack=True)

def j(x4, g, h):
    y4 = g*x4 + h
    return y4

params3, covariance_matrix3 = curve_fit(j, x4, y4)
errors3 = np.sqrt(np.diag(covariance_matrix3))
x4_plot = np.linspace(min(x4), max(x4), 1000)
plt.plot(x4, y4, 'rx',   label='$U_b = 400V$')
plt.plot(x4_plot, f(x4_plot, *params3), 'r-', label='Regression')
plt.legend()
plt.grid()
plt.xlabel(r'$U_d \, / \, \si{\volt}$')
plt.ylabel(r'$D \, / \, \si{\centi\meter}$')
plt.tight_layout()
plt.savefig('plot4.pdf')
plt.close()
x5, y5 = np.genfromtxt('content/data5.txt', unpack=True)
def k(x5, i, j):
    y5 = i*x5 + j
    return y5

params4, covariance_matrix4 = curve_fit(k, x5, y5)
errors4 = np.sqrt(np.diag(covariance_matrix4))
x5_plot = np.linspace(min(x5), max(x5), 1000)
plt.plot(x5, y5, 'rx',   label='$U_b = 500V$')
plt.plot(x5_plot, f(x5_plot, *params4), 'r-', label='Regression')
plt.legend()
plt.grid()
plt.xlabel(r'$U_d \, / \, \si{\volt}$')
plt.ylabel(r'$D \, / \, \si{\centi\meter}$')
plt.tight_layout()
plt.savefig('plot5.pdf')
plt.close()
print('a=', params[0], '+-', errors[0])
print('b=', params[1], '+-', errors[1])
print('c=', params1[0], '+-', errors1[0])
print('d=', params1[1], '+-', errors1[1])
print('e=', params2[0], '+-', errors2[0])
print('f=', params2[1], '+-', errors2[1])
print('g=', params3[0], '+-', errors3[0])
print('h=', params3[1], '+-', errors3[1])
print('i=', params4[0], '+-', errors4[0])
print('j=', params4[1], '+-', errors4[1])
#Berechnung zur Konstruktions
x, y = np.genfromtxt('content/data6.txt', unpack=True)
def f(x, a, b):
    y = a*x + b
    return y
params, covariance_matrix = curve_fit(f, x, y)
errors = np.sqrt(np.diag(covariance_matrix))
m3 = ufloat(params[0], errors[0])
Ud= (1.905/m3)*300
x_plot = np.linspace(min(x), max(x), 1000)
plt.plot(x, y, 'rp', label='Messwerte')
plt.plot(x_plot, f(x_plot, *params), 'k-', label='Regression')
plt.legend()
plt.grid()
plt.xlabel(r'$\frac{1}{U_b} \, / \, \si{\per\volt}$')
plt.ylabel(r'$\frac{D}{U_d} \, / \, \si{\centi\meter\per\volt}$')
plt.tight_layout()
plt.savefig('plot6.pdf')
plt.close()
print('Darstellung der Konstruktion:')
print('Steigung=', params[0], '+-', errors[0])
print('b=', params[1], '+-', errors[1])
print('Ablenkung=', Ud)
#Mittelwert bestimmen
x=np.array([87.99, 99.98, 79.32, 79.32])
print('Fehler=', np.mean(x), '+-', np.std(x, ddof=1) / np.sqrt(len(x)))
D = np.array([1.59e11, 1.58e11, 1.61e11, 1.63e11, 1.62e11])
print('Mittelwert der spezifischen Ladung:')
print('D=', np.mean(D), '+-', np.std(D, ddof=1) / np.sqrt(len(D)))
#Versuch 2
x, y = np.genfromtxt('content/data7.txt', unpack=True)
def f(x, a, b):
    y = a*x + b
    return y
params, covariance_matrix = curve_fit(f, x, y)
errors = np.sqrt(np.diag(covariance_matrix))

d1 = ufloat(params[0], errors[0])
D1= (8*250*d1**2 )/ 10**(-6)
x_plot = np.linspace(min(x), max(x), 1000)
plt.plot(x, y,   'rx',   label='$U_b = 250V$')
plt.plot(x_plot, f(x_plot, *params),'r-', label='Regression')
plt.legend()
plt.grid()
plt.xlabel(r'$B \, / \, \si{\milli\tesla}$')
plt.ylabel(r'$\frac{D}{L^2+D^2} \, / \, \si{\per\meter}$')
plt.tight_layout()
plt.savefig('plot7.pdf')
plt.close()

x2, y2 = np.genfromtxt('content/data8.txt', unpack=True)
def g(x2, c, d):
    y2 = c*x2 + d
    return y2
params1, covariance_matrix1 = curve_fit(g, x2, y2)
errors1 = np.sqrt(np.diag(covariance_matrix1))

d2 = ufloat(params1[0], errors1[0])
D2= (8*300*d2**2 )/ 10**(-6)

x2_plot = np.linspace(min(x2), max(x2), 1000)
plt.plot(x2, y2,   'rx',   label='$U_b = 300V$')
plt.plot(x2_plot, f(x2_plot, *params1),'r-', label='Regression')
plt.legend()
plt.grid()
plt.xlabel(r'$B \, / \, \si{\milli\tesla}$')
plt.ylabel(r'$\frac{D}{L^2+D^2} \, / \, \si{\per\meter}$')
plt.tight_layout()
plt.savefig('plot8.pdf')
plt.close()

x3, y3 = np.genfromtxt('content/data9.txt', unpack=True)
def h(x3, e, f):
    y3 = e*x3 + f
    return y3
params2, covariance_matrix2 = curve_fit(h, x3, y3)
errors2 = np.sqrt(np.diag(covariance_matrix2))
d3 = ufloat(params2[0], errors2[0])
D3= (8*350*d3**2 )/ 10**(-6)

x3_plot = np.linspace(min(x3), max(x3), 1000)
plt.plot(x3, y3,   'rx',   label='$U_b = 350V$')
plt.plot(x3_plot, f(x3_plot, *params2),'r-', label='Regression')
plt.legend()
plt.grid()
plt.xlabel(r'$B \, / \, \si{\milli\tesla}$')
plt.ylabel(r'$\frac{D}{L^2+D^2} \, / \, \si{\per\meter}$')
plt.tight_layout()
plt.savefig('plot9.pdf')
plt.close()

x4, y4 = np.genfromtxt('content/data10.txt', unpack=True)
def j(x4, g, h):
    y4 = g*x4 + h
    return y4
params3, covariance_matrix3 = curve_fit(j, x4, y4)
errors3 = np.sqrt(np.diag(covariance_matrix3))
d4 = ufloat(params3[0], errors3[0])
D4= (8*400*d4**2 )/ 10**(-6)

x4_plot = np.linspace(min(x4), max(x4), 1000)
plt.plot(x4, y4,   'rx',   label='$U_b = 400V$')
plt.plot(x4_plot, f(x4_plot, *params3),'r-', label='Regression')
plt.legend()
plt.grid()
plt.xlabel(r'$B \, / \, \si{\milli\tesla}$')
plt.ylabel(r'$\frac{D}{L^2+D^2} \, / \, \si{\per\meter}$')
plt.tight_layout()
plt.savefig('plot10.pdf')
plt.close()
x5, y5 = np.genfromtxt('content/data11.txt', unpack=True)
def k(x5, i, j):
    y5 = i*x5 + j
    return y5
params4, covariance_matrix4 = curve_fit(k, x5, y5)
errors4 = np.sqrt(np.diag(covariance_matrix4))
d5 = ufloat(params4[0], errors4[0])
D5= (8*430*d5**2 )/ 10**(-6)
x5_plot = np.linspace(min(x5), max(x5), 1000)
plt.plot(x5, y5,   'rx',   label='$U_b = 430V$')
plt.plot(x5_plot, f(x5_plot, *params4),'r-', label='Regression')
plt.legend()
plt.grid()
plt.xlabel(r'$B \, / \, \si{\milli\tesla}$')
plt.ylabel(r'$\frac{D}{L^2+D^2} \, / \, \si{\per\meter}$')
plt.tight_layout()
plt.savefig('plot11.pdf')
plt.close()





x2_plot = np.linspace(min(x2), max(x2), 1000)
x3_plot = np.linspace(min(x3), max(x3), 1000)
x4_plot = np.linspace(min(x4), max(x4), 1000)
x5_plot = np.linspace(min(x5), max(x5), 1000)

plt.plot(x, y,   'rx',   label='$U_b = 250V$')
plt.plot(x2, y2, 'kx',   label='$U_b = 300V$')
plt.plot(x3, y3, 'bx', label='$U_b = 350V$')
plt.plot(x4, y4, 'gx', label='$U_b = 400V$')
plt.plot(x5, y5, 'yx', label='$U_b = 430V$')
plt.plot(x_plot, f(x_plot, *params),    'r-', label='Regression')
plt.plot(x2_plot, g(x2_plot, *params1), 'k-', label='Regression')
plt.plot(x3_plot, h(x3_plot, *params2), 'b-', label='Regression')
plt.plot(x4_plot, j(x4_plot, *params3), 'g-', label='Regression')
plt.plot(x5_plot, k(x5_plot, *params4), 'y-', label='Regression')
plt.legend()
plt.grid()
plt.xlabel(r'$B \, / \, mT$')
plt.ylabel(r'$\frac{D}{L^2+D^2} \, / \, \frac{1}{m}$')
plt.tight_layout()
plt.savefig('plot7.pdf')
plt.close()

print('Darstellung der Steigung der 5.Messungen:')
print('a=', params[0], '+-', errors[0])
print('b=', params[1], '+-', errors[1])
print('c=', params1[0], '+-', errors1[0])
print('d=', params1[1], '+-', errors1[1])
print('e=', params2[0], '+-', errors2[0])
print('f=', params2[1], '+-', errors2[1])
print('g=', params3[0], '+-', errors3[0])
print('h=', params3[1], '+-', errors3[1])
print('i=', params4[0], '+-', errors4[0])
print('j=', params4[1], '+-', errors4[1])
print('Fehlerdarstellung:')
print('1=', D1)
print('2=', D2)
print('3=', D3)
print('4=', D4)
print('5=', D5)

# Winkelberechnung
x=np.array([1.43, 1.27, 0.96, 0.96])
y = np.mean(x)
dy = np.std(x, ddof=1) / np.sqrt(len(x))
dx=ufloat(y, dy)
B= 8.035/unumpy.cos(dx)
print('Winkelfehler=', y, '+-', dy)
print('B=', B)
