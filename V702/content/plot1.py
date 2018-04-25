import matplotlib as mpl
mpl.use('pgf')
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
mpl.rcParams.update({
'font.family': 'serif',
'text.usetex': True,
'pgf.rcfonts': False,
'pgf.texsystem': 'lualatex',
'pgf.preamble': r'\usepackage{unicode-math}\usepackage{siunitx}',
})

U = 2.5
x, y_2 = np.genfromtxt('content/data1.txt', unpack=True)

y_1 = (y_2-U)
y = np.log(y_1)


plt.plot(x, y, r'rx', label='Messwerte')
plt.legend()
plt.grid()
plt.ylabel(r'$\ln(N_{\Delta t})$')
plt.xlabel(r'$t \, / \, \si{\second}$')
plt.tight_layout()
plt.savefig('plot1.pdf')
