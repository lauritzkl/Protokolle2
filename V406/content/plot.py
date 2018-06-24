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
z, y = np.genfromtxt('content/data.txt', unpack=True)

x= 23.75 - z  #Abzug vom Hauptmaxima
#y= w - 0.0093 #Abzug vom Dunkelstrom

p = x / 1000 #Winkel phi = x /L
I= y
l= 635 * 10**(-9)
np.savetxt('content/Winkel1.txt', np.column_stack([p]), header="Der Winkel von Messung 1")

def f(p, a, b, c, d):
   y = a * ((l/(np.pi * np.sin(p - d)))**2) * (np.sin((np.pi * b * np.sin(p - d))/l)**2) + c
   return y

params, covariance_matrix = curve_fit(f, p, y, p0=((10**10), (0.15*10**(-3)), 1, (-0.0002)))
errors = np.sqrt(np.diag(covariance_matrix))
t = np.linspace(min(p), max(p), 1000)

plt.plot(p, I, r'rx', label='Messwerte')
plt.plot(t, f(t, *params), 'k-', label='Regression')
plt.legend()
plt.grid()
plt.ylabel('Strom proportional zur Intensität /µA ')
plt.xlabel(r'$\varphi$ / rad')
plt.savefig('plot1.pdf')
plt.close()

print('Spalt 0,15mm:')
print('a =', params[0], '+/-', errors[0])
print('b =', params[1], '+/-', errors[1])
print('c =', params[2], '+/-', errors[2])
print('d =', params[3], '+/-', errors[3])


#Messung am Spalt 2

k, y = np.genfromtxt('content/data2.txt', unpack=True)
x= (24.3 - k)
l= 635 * 10**(-9)


p = x / 1000 #Winkel phi
np.savetxt('content/Winkel2.txt', np.column_stack([p]), header="Der Winkel von Messung 2")
def f(p, a, b, c, d):
   y = a * ((l/(np.pi * np.sin(p - d)))**2) * (np.sin((np.pi * b * np.sin(p - d))/l)**2) + c
   return y

params, covariance_matrix = curve_fit(f, p, y, p0=((10**10), (0.075*10**(-3)), 1, (-0.0002)))
errors = np.sqrt(np.diag(covariance_matrix))
t = np.linspace(min(p), max(p), 1000)

plt.plot(p, y, r'rx', label='Messwerte')
plt.plot(t, f(t, *params), 'k-', label='Regression')
plt.legend()
plt.grid()
plt.ylabel('Strom proportional zur Intensität / µA')
plt.xlabel(r'$\varphi$ / rad')
plt.savefig('plot2.pdf')
plt.close()

print('Spalt 0,075mm:')
print('a =', params[0], '+/-', errors[0])
print('b =', params[1], '+/-', errors[1])
print('c =', params[2], '+/-', errors[2])
print('d =', params[3], '+/-', errors[3])



#Messung am Doppelspalt

k, y = np.genfromtxt('content/data3.txt', unpack=True)
x= (24.2 - k)
l= 635 * 10**(-9)
p = x / 1000 #Winkel phi
np.savetxt('content/Winkel3.txt', np.column_stack([p]), header="Der Winkel von Messung 3")

def f(p, a, s, b, d):
  y = a * ((np.cos((np.pi * s * np.sin(p - d))/l))**2) * ((l/(np.pi * b * np.sin(p - d)))**2) * ((np.sin((np.pi * b * np.sin(p - d))/l))**2)
  return y

params, covariance_matrix = curve_fit(f, p, y, p0=((6*10**3), (0.5*10**(-3)), (0.15*10**(-3)), (- 0.0002)))
errors = np.sqrt(np.diag(covariance_matrix))
t = np.linspace(min(p), max(p), 1000)

plt.plot(p, y, r'rx', label='Messwerte')
plt.plot(t, f(t, *params), 'k-', label='Regression')
plt.legend()
plt.grid()
plt.ylabel('Strom proportional zur Intensität / µA')
plt.xlabel(r'$\varphi$ / rad')
plt.savefig('plot3.pdf')
plt.close()
print('Doppelspalt:')
print('a =', params[0], '+/-', errors[0])
print('s =', params[1], '+/-', errors[1])
print('b =', params[2], '+/-', errors[2])
print('d =', params[3], '+/-', errors[3])

#Messung Vergleichen

k, y = np.genfromtxt('content/data3.txt', unpack=True)
x= (24.2 - k)
l= 635 * 10**(-9)
p = x / 1000 #Winkel phi

np.savetxt('content/Winkel3.txt', np.column_stack([p]), header="Der Winkel von Messung 3")

def f(p, a, s, b, d):
  y = a * ((np.cos((np.pi * s * np.sin(p - d))/l))**2) * ((l/(np.pi * b * np.sin(p - d)))**2) * ((np.sin((np.pi * b * np.sin(p - d))/l))**2)
  return y

params, covariance_matrix = curve_fit(f, p, y, p0=((6*10**3), (0.5*10**(-3)), (0.15*10**(-3)), (- 0.0002)))
errors = np.sqrt(np.diag(covariance_matrix))

def d(p):
    return 6.3e7 * ((l/(np.pi * np.sin(p - params[3])))**2) * (np.sin((np.pi * params[2] * np.sin(p - params[3]))/l)**2)


t = np.linspace(min(p), max(p), 1000)

plt.plot(p, y, r'rx', label='Messwerte')
plt.plot(t, f(t, *params), 'k-', label='Doppelspalt')
plt.plot(t, d(t), 'b-', label='Einzelspalt')
plt.legend()
plt.grid()
plt.ylabel('Strom proportional zur Intensität / µA')
plt.xlabel(r'$\varphi$ / rad')
plt.savefig('plot4.pdf')
plt.close()

#print(params1[0])
