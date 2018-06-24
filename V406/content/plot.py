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

x, I = np.genfromtxt('content/data3.txt', unpack=True)

p = x / 1000     # x / L
l = 635 * 10**(-9)
shiftdop = -0.000180171874364
shiftp = p - shiftdop

plt.plot(shiftp, I, 'kx', label='Messwerte Doppelspalt')

A0dop = 1.6775075197717908
sdop = 0.0005084773951663262
bdop = 0.0001633336719143504

A0sing = 20106583.17271117
bsing = 0.00014944404494694484
csing = 0.006199038950450398

def d(x, a, s, b):
    y = a * (np.cos((np.pi * s * np.sin(x))/l)**2) * ((l/(np.pi * b * np.sin(x)))**2) * (np.sin((np.pi * b * np.sin(x))/l)**2)
    return y

def e(x, a, b, c):
    y = a * ((l/(np.pi * np.sin(x)))**2) * (np.sin((np.pi * b * np.sin(x))/l)**2) + c
    return y

t = np.linspace(min(p), max(p), 1000)
plt.plot(t, d(t, A0dop, sdop, bdop), 'b-', label='Doppelspalt')
plt.plot(t, e(t, A0sing, bsing, csing), 'r-', label='Einzelspalt')
plt.xlabel(r'$\varphi$ / rad')
plt.ylabel('Strom proportional zur Intensität / nA')
plt.legend(loc='best')
plt.grid()
plt.tight_layout()
plt.savefig('plot4.pdf')
plt.show()


print(d((0.000001 / 1000), A0dop, sdop, bdop))
