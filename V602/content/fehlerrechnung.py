import numpy as np

Z = 83
R = 2.1798e-18
a = 7.297e-3
E = 1.37787e-15


print(Z-((4/a)*np.sqrt(E/R)-(5*E/R))**(1/2)*(1+(19/32)*(a**2)*(E/R))**(1/2))
