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


x_1, y_1 = np.genfromtxt('content/data2.txt', unpack=True)

plt.plot(x_1, y_1, r'rx', label=r'Steigung')

plt.legend()
plt.grid()
plt.ylabel(r'$a$')
plt.xlabel(r'$U_a \, / \, \si{\volt}$')
plt.tight_layout()
plt.savefig('plot2.pdf')
