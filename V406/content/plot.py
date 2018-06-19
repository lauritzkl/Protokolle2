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
#Berechung von lambda:

N_1 = np.genfromtxt('content/data.txt', unpack=True)

y = (2/N_1) * (5e-3/5.017)
y_1 = np.mean(y)
dy = np.sqrt(np.var(y, ddof=1))/np.sqrt(len(y))

W = ufloat(y_1, dy)

print('Wellenlängen=', y)
print('Gemittelte Wellenlängen=', W)


#Berechnung von n:

N_2 = np.genfromtxt('content/data2.txt', unpack=True)

dn = (N_2 * 635e-9) / (2 * 50e-3)

n = 1 + dn * (1013.2e-3/0.8)

print('n=', n)
print('Gemitteltes n:')
print('n=', np.mean(n), '+-', np.sqrt(np.var(n, ddof=1))/np.sqrt(len(n)))
