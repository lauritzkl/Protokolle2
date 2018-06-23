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

#Messung am Spalt 1
z, w = np.genfromtxt('content/data.txt', unpack=True)

x= 23.75 - z  #Abzug vom Hauptmaxima
y= w - 0.0093 #Abzug vom Dunkelstrom

p = x / 1000 #Winkel phi = x /L
I= y * 1000
l= 635 * 10**(-9)

x_l = x[1:30]
y_l = y[1:30]
x_r = x[30:]
y_r = y[30:]

def f(x_l, a, b, c, d):
   y = a * ((l*b/(np.pi * np.sin(x_l - d)))**2) * (np.sin((np.pi * b * np.sin(x_l - d))/l)**2) + c
   return y

params, covariance_matrix = curve_fit(f, p, y)
errors = np.sqrt(np.diag(covariance_matrix))
t = np.linspace(min(p), max(p), 1000)

plt.plot(p, I, r'rx', label='Messwerte')
plt.plot(t, f(x_l, *params), 'k-', label='Regression')
plt.legend()
plt.grid()
plt.ylabel('Strom proportional zur Intensität / nA')
plt.xlabel(r'$\varphi$ / rad')
plt.savefig('plot1.pdf')
plt.close()

#print('a =', params[0], '+/-', err[0])
#print('b =', params[1], '+/-', err[1])
#print('c =', params[2], '+/-', err[2])
#print('d =', params[3], '+/-', err[3])


#Messung am Spalt 2

k, l = np.genfromtxt('content/data2.txt', unpack=True)
x= (24.3 - k)
y= (l - 0.0093)*10**(-6)

p = x / 1000 #Winkel phi

#def f(x, a, b, c, d):
#   y = a * (((635 * 10**(-9))/(np.pi * np.sin(x - d)))**2) * ((np.sin((np.pi * b * np.sin(x - d))/(635 * 10**(-9))))**2) + c
#   return y
#
#params, covariance_matrix = curve_fit(f, x, y)
#errors = np.sqrt(np.diag(covariance_matrix))
#t = np.linspace(min(x), max(x), 1000)

plt.plot(p, y, r'rx', label='Messwerte')
#plt.plot(t, f(x, *params), 'k-', label='Regression')
plt.legend()
plt.grid()
plt.ylabel('Strom proportional zur Intensität / nA')
plt.xlabel(r'$\varphi$ / rad')
plt.savefig('plot2.pdf')
plt.close()
#print('a =', params[0], '+/-', err[0])
#print('b =', params[1], '+/-', err[1])
#print('c =', params[2], '+/-', err[2])
#print('d =', params[3], '+/-', err[3])



#Messung am Doppelspalt

k, l = np.genfromtxt('content/data3.txt', unpack=True)
x= (24.2 - k)
y= (l - 0.0093)*10**(-6)

p = x / 1000 #Winkel phi

#def f(x, a, s, b, d):
#  a * ((np.cos((np.pi * s * np.sin(x - d))/(635 * 10**(-9))))**2) * (((635 * 10**(-9))/(np.pi * b * np.sin(x - d)))**2) * ((np.sin((np.pi * b * np.sin(x - d))/(635 * 10**(-9))))**2)
#   return y
#
#params, covariance_matrix = curve_fit(f, x, y)
#errors = np.sqrt(np.diag(covariance_matrix))
#t = np.linspace(min(x), max(x), 1000)

plt.plot(p, y, r'rx', label='Messwerte')
#plt.plot(t, f(x, *params), 'k-', label='Regression')
plt.legend()
plt.grid()
plt.ylabel('Strom proportional zur Intensität / nA')
plt.xlabel(r'$\varphi$ / rad')
plt.savefig('plot3.pdf')
plt.close()
#print('a =', params[0], '+/-', err[0])
#print('b =', params[1], '+/-', err[1])
#print('c =', params[2], '+/-', err[2])
#print('d =', params[3], '+/-', err[3])

#Messung Vergleichen
